#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Layercast benchmark: measure model load latency across HF, NFS, and NIXL scenarios.

Deploys each scenario on Kubernetes, captures structured logs via stern, collects
hardware context from the cluster, and produces a markdown table, JSON results
file, and comparison charts.

Usage:
    ./scripts/benchmark.py                          # run all scenarios
    ./scripts/benchmark.py --skip-hf                # skip HF cold (slow)
    ./scripts/benchmark.py --skip-nfs               # skip NFS
    ./scripts/benchmark.py --skip-nixl              # skip NIXL P2P
    ./scripts/benchmark.py --skip-nixl-scaling      # skip NIXL 3-replica scaling
    ./scripts/benchmark.py --skip-populate          # assume model already on PVC
    ./scripts/benchmark.py --model Qwen/Qwen2.5-3B  # use a smaller model
    ./scripts/benchmark.py --cleanup                # tear everything down
    ./scripts/benchmark.py --from-logs <dir>        # re-process existing logs (no deploy)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HardwareContext:
    """Hardware details for the node a pod ran on."""
    node_name: str = ""
    gpu_product: str = ""
    gpu_count: int = 0
    ib_device: str = ""

    @classmethod
    def from_pod(cls, pod_name: str, kubectl: list[str]) -> HardwareContext:
        """Collect hardware context from the node where a pod is scheduled."""
        ctx = cls()
        try:
            ctx.node_name = _kubectl_jsonpath(
                kubectl, "pod", pod_name, "{.spec.nodeName}"
            )
        except SubprocessError:
            return ctx

        if not ctx.node_name:
            return ctx

        try:
            ctx.gpu_product = _kubectl_jsonpath(
                kubectl, "node", ctx.node_name,
                r"{.metadata.labels.nvidia\.com/gpu\.product}",
            )
        except SubprocessError:
            pass

        try:
            raw = _kubectl_jsonpath(
                kubectl, "node", ctx.node_name,
                r"{.status.capacity.nvidia\.com/gpu}",
            )
            ctx.gpu_count = int(raw) if raw else 0
        except (SubprocessError, ValueError):
            pass

        try:
            ctx.ib_device = _kubectl_jsonpath(
                kubectl, "node", ctx.node_name,
                r"{.metadata.labels.nvidia\.com/gpu\.machine}",
            )
        except SubprocessError:
            pass

        return ctx


@dataclass
class TransferMetrics:
    """Metrics extracted from structured log events."""
    weight_load_s: float | None = None
    transfer_gbps: float | None = None
    transfer_bytes: int | None = None
    checksum_s: float | None = None
    compile_cache_gets: int | None = None
    compile_cache_hits: int | None = None


@dataclass
class ScenarioResult:
    """Complete result for a single benchmark scenario."""
    scenario: str
    method: str
    ttr_s: float | None = None
    metrics: TransferMetrics = field(default_factory=TransferMetrics)
    hardware: HardwareContext = field(default_factory=HardwareContext)
    pod_name: str = ""
    log_file: str = ""


@dataclass
class BenchmarkRun:
    """Top-level benchmark run with all results and metadata."""
    model: str
    cluster: str
    timestamp: str
    vllm_image: str
    log_dir: str
    results: list[ScenarioResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

class SubprocessError(Exception):
    pass


def _run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: int = 300,
) -> str:
    """Run a command, return stdout. Raises SubprocessError on failure if check=True."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SubprocessError(f"timeout ({timeout}s): {' '.join(cmd)}") from exc
    if check and r.returncode != 0:
        stderr = r.stderr.strip() if r.stderr else ""
        raise SubprocessError(f"{' '.join(cmd[:4])}... failed: {stderr[:200]}")
    return (r.stdout or "").strip()


def _kubectl_jsonpath(
    kubectl: list[str], resource: str, name: str, jsonpath: str
) -> str:
    return _run([*kubectl, "get", resource, name, "-o", f"jsonpath={jsonpath}"])


# ---------------------------------------------------------------------------
# K8s operations
# ---------------------------------------------------------------------------

class K8sRunner:
    """Wraps kubectl and stern for benchmark orchestration."""

    def __init__(self, context: str, namespace: str) -> None:
        self.context = context
        self.namespace = namespace
        self.kubectl = ["kubectl", "--context", context, "-n", namespace]
        self._stern_procs: list[subprocess.Popen[bytes]] = []

    def ensure_namespace(self) -> None:
        try:
            _run([*self.kubectl, "get", "ns", self.namespace], timeout=10)
        except SubprocessError:
            _run(["kubectl", "--context", self.context, "create", "ns", self.namespace])

    def apply_template(
        self,
        yaml_path: str,
        *,
        model: str,
        vllm_image: str,
    ) -> None:
        """envsubst + kustomize image override, then kubectl apply."""
        vllm_name, vllm_tag = _split_image(vllm_image)

        with tempfile.TemporaryDirectory() as tmp:
            # Expand model variables
            raw = Path(yaml_path).read_text()
            raw = raw.replace("${BENCH_MODEL}", model)
            raw = raw.replace("${BENCH_MODEL_PATH}", model)
            Path(tmp, "resource.yaml").write_text(raw)

            # Kustomization for image overrides
            kustomization = (
                "apiVersion: kustomize.config.k8s.io/v1beta1\n"
                "kind: Kustomization\n"
                "resources:\n"
                "  - resource.yaml\n"
                "images:\n"
                f"  - name: vllm-plugin\n"
                f"    newName: {vllm_name}\n"
                f'    newTag: "{vllm_tag}"\n'
            )
            Path(tmp, "kustomization.yaml").write_text(kustomization)
            _run([*self.kubectl, "apply", "-k", tmp])

    def delete_template(
        self,
        yaml_path: str,
        *,
        model: str,
        vllm_image: str,
    ) -> None:
        """Delete resources from a templated YAML."""
        vllm_name, vllm_tag = _split_image(vllm_image)

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(yaml_path).read_text()
            raw = raw.replace("${BENCH_MODEL}", model)
            raw = raw.replace("${BENCH_MODEL_PATH}", model)
            Path(tmp, "resource.yaml").write_text(raw)

            kustomization = (
                "apiVersion: kustomize.config.k8s.io/v1beta1\n"
                "kind: Kustomization\n"
                "resources:\n"
                "  - resource.yaml\n"
                "images:\n"
                f"  - name: vllm-plugin\n"
                f"    newName: {vllm_name}\n"
                f'    newTag: "{vllm_tag}"\n'
            )
            Path(tmp, "kustomization.yaml").write_text(kustomization)
            try:
                _run([*self.kubectl, "delete", "--ignore-not-found", "-k", tmp])
            except SubprocessError:
                pass

    def get_pod(self, scenario_label: str) -> str | None:
        """Get the first pod name matching a bench/scenario label."""
        try:
            name = _run([
                *self.kubectl, "get", "pods",
                "-l", f"bench/scenario={scenario_label}",
                "-o", "jsonpath={.items[0].metadata.name}",
            ])
            return name or None
        except SubprocessError:
            return None

    def wait_for_ready(self, scenario_label: str, timeout: int = 900) -> float | None:
        """Poll until pod is Ready. Returns elapsed seconds or None on timeout."""
        info(f"Waiting for bench/{scenario_label} to be Ready (timeout {timeout}s)...")
        start = time.monotonic()

        while (elapsed := time.monotonic() - start) < timeout:
            pod = self.get_pod(scenario_label)
            if pod:
                try:
                    ready = _kubectl_jsonpath(
                        self.kubectl, "pod", pod,
                        '{.status.conditions[?(@.type=="Ready")].status}',
                    )
                    if ready == "True":
                        total = time.monotonic() - start
                        info(f"{scenario_label} ready in {total:.1f}s")
                        return total
                except SubprocessError:
                    pass

                # Progress
                try:
                    statuses = _run([
                        *self.kubectl, "get", "pod", pod,
                        "-o", "jsonpath={range .status.containerStatuses[*]}{.name}={.ready} {end}",
                    ])
                except SubprocessError:
                    statuses = "pending"
                print(f"  ... {statuses} ({elapsed:.0f}s)", file=sys.stderr)
            else:
                print(f"  ... waiting for pod ({elapsed:.0f}s)", file=sys.stderr)

            time.sleep(10)

        warn(f"Timed out waiting for {scenario_label}")
        return None

    def get_k8s_ttr(self, pod_name: str) -> float | None:
        """Compute creation-to-ready duration from K8s timestamps."""
        try:
            created = _kubectl_jsonpath(
                self.kubectl, "pod", pod_name,
                "{.metadata.creationTimestamp}",
            )
            ready_time = _kubectl_jsonpath(
                self.kubectl, "pod", pod_name,
                '{.status.conditions[?(@.type=="Ready")].lastTransitionTime}',
            )
        except SubprocessError:
            return None

        if not created or not ready_time:
            return None

        try:
            t_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            t_ready = datetime.fromisoformat(ready_time.replace("Z", "+00:00"))
            return (t_ready - t_created).total_seconds()
        except ValueError:
            return None

    def start_stern(self, scenario_label: str, log_path: Path) -> None:
        """Start stern in the background to capture pod logs."""
        proc = subprocess.Popen(
            [
                "stern", "--context", self.context, "-n", self.namespace,
                "-l", f"bench/scenario={scenario_label}",
                "--output", "raw", "--no-follow=false",
            ],
            stdout=log_path.open("w"),
            stderr=subprocess.DEVNULL,
        )
        self._stern_procs.append(proc)
        info(f"stern capturing {scenario_label} (pid {proc.pid}) -> {log_path}")

    def stop_stern(self) -> None:
        """Kill all running stern processes."""
        for proc in self._stern_procs:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        self._stern_procs.clear()

    def scrape_compile_cache_stats(self) -> dict[str, int]:
        """Fetch compile cache stats via port-forward to the metadata server."""
        import socket

        # Find a free local port
        with socket.socket() as s:
            s.bind(("", 0))
            local_port = s.getsockname()[1]

        pf = subprocess.Popen(
            [*self.kubectl, "port-forward", "svc/layercast-metadata-server",
             f"{local_port}:8081"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(2)
            raw = _run(["curl", "-sf", f"http://127.0.0.1:{local_port}/compile-cache-stats"])
            return json.loads(raw) if raw else {}
        except (SubprocessError, json.JSONDecodeError):
            return {}
        finally:
            pf.kill()
            pf.wait()

    def apply_raw(self, path: str) -> None:
        try:
            _run([*self.kubectl, "apply", "-f", path])
        except SubprocessError:
            pass

    def delete_raw(self, resource: str, name: str) -> None:
        try:
            _run([*self.kubectl, "delete", resource, name, "--ignore-not-found"])
        except SubprocessError:
            pass

    def wait_job(self, name: str, timeout: int = 3600) -> bool:
        try:
            _run(
                [*self.kubectl, "wait", f"--for=condition=complete", f"job/{name}",
                 f"--timeout={timeout}s"],
                timeout=timeout + 30,
            )
            return True
        except SubprocessError:
            return False


def _split_image(image: str) -> tuple[str, str]:
    """Split 'registry/repo:tag' into (name, tag)."""
    if ":" in image:
        name, tag = image.rsplit(":", 1)
        return name, tag
    return image, "latest"


# ---------------------------------------------------------------------------
# Log extraction
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def extract_log_field(log_path: Path, event: str, field_name: str) -> str | None:
    """Extract a field from the last matching structured log event."""
    if not log_path.exists():
        return None

    last_match: str | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if f'"event": "{event}"' not in line:
            continue
        last_match = line

    if not last_match:
        return None

    cleaned = _ANSI_RE.sub("", last_match)
    m = re.search(r"\{.*\}", cleaned)
    if not m:
        return None

    try:
        data = json.loads(m.group())
        val = data.get(field_name)
        return str(val) if val is not None else None
    except json.JSONDecodeError:
        return None


def extract_metrics(log_path: Path) -> TransferMetrics:
    """Extract all transfer metrics from a log file."""
    metrics = TransferMetrics()

    raw = extract_log_field(log_path, "model_loaded", "elapsed_s")
    if raw:
        try:
            metrics.weight_load_s = float(raw)
        except ValueError:
            pass

    raw = extract_log_field(log_path, "nixl_transfer_complete", "gbps")
    if raw:
        try:
            metrics.transfer_gbps = float(raw)
        except ValueError:
            pass

    raw = extract_log_field(log_path, "nixl_transfer_complete", "total_bytes")
    if raw:
        try:
            metrics.transfer_bytes = int(float(raw))
        except ValueError:
            pass

    raw = extract_log_field(log_path, "checksum_verification", "elapsed_s")
    if raw:
        try:
            metrics.checksum_s = float(raw)
        except ValueError:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

# Colors chosen to be distinguishable and look decent on both light/dark backgrounds.
# Ordered: baseline grey, NFS blue, NIXL teal, NIXL+cache green.
PALETTE = ["#8c8c8c", "#4878a8", "#2a9d8f", "#57cc99"]


def generate_charts(run: BenchmarkRun, output_dir: Path) -> list[Path]:
    """Generate comparison charts from benchmark results. Returns paths of created files."""
    if not run.results:
        return []

    charts: list[Path] = []
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # -- Time-to-Ready comparison --
    ttr_results = [(r.scenario, r.ttr_s) for r in run.results if r.ttr_s is not None]
    if len(ttr_results) >= 2:
        path = _bar_chart(
            labels=[s for s, _ in ttr_results],
            values=[v for _, v in ttr_results],
            title=f"Time-to-Ready: {run.model}",
            ylabel="Seconds",
            output=output_dir / "chart_ttr.png",
            palette=PALETTE,
        )
        charts.append(path)

    # -- Weight load time comparison --
    wl_results = [
        (r.scenario, r.metrics.weight_load_s)
        for r in run.results
        if r.metrics.weight_load_s is not None
    ]
    if len(wl_results) >= 2:
        path = _bar_chart(
            labels=[s for s, _ in wl_results],
            values=[v for _, v in wl_results],
            title=f"Weight Load Time: {run.model}",
            ylabel="Seconds",
            output=output_dir / "chart_weight_load.png",
            palette=PALETTE,
        )
        charts.append(path)

    # -- Transfer throughput comparison --
    tp_results = [
        (r.scenario, r.metrics.transfer_gbps)
        for r in run.results
        if r.metrics.transfer_gbps is not None
    ]
    if len(tp_results) >= 2:
        path = _bar_chart(
            labels=[s for s, _ in tp_results],
            values=[v for _, v in tp_results],
            title=f"NIXL Transfer Throughput: {run.model}",
            ylabel="GB/s",
            output=output_dir / "chart_throughput.png",
            palette=PALETTE,
        )
        charts.append(path)

    # -- Stacked TTR breakdown (weight load vs overhead) --
    breakdown = []
    for r in run.results:
        if r.ttr_s is not None and r.metrics.weight_load_s is not None:
            overhead = max(0, r.ttr_s - r.metrics.weight_load_s)
            breakdown.append((r.scenario, r.metrics.weight_load_s, overhead))
    if len(breakdown) >= 2:
        path = _stacked_bar_chart(
            labels=[s for s, _, _ in breakdown],
            bottom_values=[w for _, w, _ in breakdown],
            top_values=[o for _, _, o in breakdown],
            bottom_label="Weight loading",
            top_label="Overhead (init, compile, etc.)",
            title=f"TTR Breakdown: {run.model}",
            ylabel="Seconds",
            output=output_dir / "chart_ttr_breakdown.png",
        )
        charts.append(path)

    return charts


def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output: Path,
    palette: list[str],
) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    info(f"Chart saved: {output}")
    return output


def _stacked_bar_chart(
    labels: list[str],
    bottom_values: list[float],
    top_values: list[float],
    bottom_label: str,
    top_label: str,
    title: str,
    ylabel: str,
    output: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))
    ax.bar(labels, bottom_values, color="#2a9d8f", width=0.6, label=bottom_label,
           edgecolor="white", linewidth=0.5)
    ax.bar(labels, top_values, bottom=bottom_values, color="#e9c46a", width=0.6,
           label=top_label, edgecolor="white", linewidth=0.5)

    for i, (b, t) in enumerate(zip(bottom_values, top_values)):
        ax.text(i, b + t, f"{b + t:.1f}s", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    info(f"Chart saved: {output}")
    return output


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_markdown_table(run: BenchmarkRun) -> str:
    """Generate a markdown results table with hardware context."""
    lines = [
        "",
        "## Benchmark Results",
        "",
        f"**Model**: `{run.model}`  ",
        f"**Cluster**: {run.cluster}  ",
        f"**Date**: {run.timestamp}  ",
        f"**vLLM image**: `{run.vllm_image}`  ",
        f"**Logs**: `{run.log_dir}/`  ",
    ]

    # Hardware context from first result that has it
    hw = next((r.hardware for r in run.results if r.hardware.gpu_product), None)
    if hw:
        lines.append(f"**GPU**: {hw.gpu_product} (x{hw.gpu_count} per node)  ")
        if hw.ib_device:
            lines.append(f"**Machine**: {hw.ib_device}  ")

    lines.append("")

    # Table header
    cols = [
        ("Scenario", 25),
        ("TTR (s)", 10),
        ("Weight Load (s)", 16),
        ("Throughput", 12),
        ("Model Size", 12),
        ("Checksum (s)", 13),
        ("Method", 18),
    ]
    header = "| " + " | ".join(f"{name:<{w}}" for name, w in cols) + " |"
    separator = "| " + " | ".join("-" * w for _, w in cols) + " |"
    lines.extend([header, separator])

    for r in run.results:
        ttr = f"{r.ttr_s:.1f}" if r.ttr_s is not None else "N/A"
        wl = f"{r.metrics.weight_load_s:.1f}" if r.metrics.weight_load_s is not None else "N/A"
        gbps = f"{r.metrics.transfer_gbps:.1f} GB/s" if r.metrics.transfer_gbps is not None else "N/A"
        model_gb = f"{r.metrics.transfer_bytes / 1e9:.1f} GB" if r.metrics.transfer_bytes else "N/A"
        cksum = f"{r.metrics.checksum_s:.1f}" if r.metrics.checksum_s is not None else "N/A"

        cc = ""
        if r.metrics.compile_cache_gets is not None and r.metrics.compile_cache_gets > 0:
            hits = r.metrics.compile_cache_hits or 0
            cc = f" (cc {hits}/{r.metrics.compile_cache_gets})"

        vals = [
            (r.scenario, 25),
            (ttr, 10),
            (wl, 16),
            (gbps, 12),
            (model_gb, 12),
            (cksum, 13),
            (f"{r.method}{cc}", 18),
        ]
        lines.append("| " + " | ".join(f"{v:<{w}}" for v, w in vals) + " |")

    # Speedup summary
    ttr_values = {r.method: r.ttr_s for r in run.results if r.ttr_s is not None}
    wl_values = {r.method: r.metrics.weight_load_s for r in run.results if r.metrics.weight_load_s is not None}

    lines.append("")
    baseline_ttr = ttr_values.get("nfs-read") or ttr_values.get("hf-download")
    if baseline_ttr:
        baseline_label = "NFS" if "nfs-read" in ttr_values else "HF"
        for r in run.results:
            if r.ttr_s and r.method == "nixl-gpudirect" and r.ttr_s < baseline_ttr:
                speedup = baseline_ttr / r.ttr_s
                lines.append(
                    f"**{r.scenario}**: {speedup:.1f}x faster TTR vs {baseline_label} "
                    f"({r.ttr_s:.1f}s vs {baseline_ttr:.1f}s)  "
                )
                if r.metrics.weight_load_s and baseline_ttr:
                    wl_speedup = baseline_ttr / r.metrics.weight_load_s
                    lines.append(
                        f"Weight load: {wl_speedup:.1f}x faster "
                        f"({r.metrics.weight_load_s:.1f}s vs {baseline_ttr:.1f}s)  "
                    )

    lines.append("")
    return "\n".join(lines)


def save_json_results(run: BenchmarkRun, output_dir: Path) -> Path:
    """Save structured results as JSON."""
    path = output_dir / "results.json"
    path.write_text(json.dumps(asdict(run), indent=2, default=str))
    info(f"Results saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run_hf_cold(
    k8s: K8sRunner, cfg: BenchConfig, log_dir: Path,
) -> ScenarioResult:
    """Scenario: cold model download from HuggingFace."""
    info("=== Scenario: HF Cold Download ===")
    result = ScenarioResult(scenario="HF Cold", method="hf-download")

    yaml_path = f"{cfg.deploy_dir}/hf-cold.yaml"
    tmpl_kwargs = dict(model=cfg.model, vllm_image=cfg.vllm_image)

    k8s.delete_template(yaml_path, **tmpl_kwargs)
    time.sleep(2)
    k8s.apply_template(yaml_path, **tmpl_kwargs)

    log_path = log_dir / "hf-cold.log"
    k8s.start_stern("hf-cold", log_path)
    result.log_file = str(log_path)

    k8s.wait_for_ready("hf-cold", timeout=3600)
    pod = k8s.get_pod("hf-cold")
    if pod:
        result.pod_name = pod
        result.ttr_s = k8s.get_k8s_ttr(pod)
        result.hardware = HardwareContext.from_pod(pod, k8s.kubectl)

    k8s.stop_stern()
    k8s.delete_template(yaml_path, **tmpl_kwargs)
    return result


def run_nfs_cached(
    k8s: K8sRunner, cfg: BenchConfig, log_dir: Path,
) -> ScenarioResult:
    """Scenario: model load from pre-populated NFS PVC."""
    info("=== Scenario: NFS Cached (VAST) ===")
    result = ScenarioResult(scenario="NFS Cached", method="nfs-read")

    yaml_path = f"{cfg.deploy_dir}/nfs-cached.yaml"
    tmpl_kwargs = dict(model=cfg.model, vllm_image=cfg.vllm_image)

    k8s.apply_raw(f"{cfg.deploy_dir}/pvc.yaml")
    k8s.delete_raw("deployment", "bench-nfs-cached")
    time.sleep(2)
    k8s.apply_template(yaml_path, **tmpl_kwargs)

    log_path = log_dir / "nfs-cached.log"
    k8s.start_stern("nfs-cached", log_path)
    result.log_file = str(log_path)

    k8s.wait_for_ready("nfs-cached", timeout=900)
    pod = k8s.get_pod("nfs-cached")
    if pod:
        result.pod_name = pod
        result.ttr_s = k8s.get_k8s_ttr(pod)
        result.hardware = HardwareContext.from_pod(pod, k8s.kubectl)

    k8s.stop_stern()
    k8s.delete_raw("deployment", "bench-nfs-cached")
    return result


def run_nixl_p2p(
    k8s: K8sRunner, cfg: BenchConfig, log_dir: Path,
) -> ScenarioResult:
    """Scenario: NIXL GPUDirect P2P weight transfer."""
    info("=== Scenario: NIXL GPUDirect P2P ===")
    result = ScenarioResult(scenario="NIXL P2P", method="nixl-gpudirect")

    yaml_path = f"{cfg.deploy_dir}/nixl-p2p.yaml"
    tmpl_kwargs = dict(model=cfg.model, vllm_image=cfg.vllm_image)

    # Ensure metadata server + PVC
    _run([*k8s.kubectl, "apply", "-k", "deploy/metadata-server/"])
    k8s.apply_raw(f"{cfg.deploy_dir}/pvc.yaml")

    k8s.delete_template(yaml_path, **tmpl_kwargs)
    time.sleep(2)
    k8s.apply_template(yaml_path, **tmpl_kwargs)

    seed_log = log_dir / "nixl-seed.log"
    consumer_log = log_dir / "nixl-consumer.log"
    k8s.start_stern("nixl-seed", seed_log)
    k8s.start_stern("nixl-consumer", consumer_log)
    result.log_file = str(consumer_log)

    info("Waiting for NIXL seed...")
    k8s.wait_for_ready("nixl-seed", timeout=900)

    info("Waiting for NIXL consumer (this is the metric)...")
    k8s.wait_for_ready("nixl-consumer", timeout=600)

    pod = k8s.get_pod("nixl-consumer")
    if pod:
        result.pod_name = pod
        result.ttr_s = k8s.get_k8s_ttr(pod)
        result.hardware = HardwareContext.from_pod(pod, k8s.kubectl)

    # Let stern flush
    time.sleep(3)
    result.metrics = extract_metrics(consumer_log)

    k8s.stop_stern()
    k8s.delete_template(yaml_path, **tmpl_kwargs)
    return result


def run_nixl_scaling(
    k8s: K8sRunner, cfg: BenchConfig, log_dir: Path,
) -> list[ScenarioResult]:
    """Scenario: NIXL scaling with seed + 2 consumers."""
    info("=== Scenario: NIXL Scaling (seed -> c1 -> c2) ===")

    yaml_path = f"{cfg.deploy_dir}/nixl-scaling.yaml"
    tmpl_kwargs = dict(model=cfg.model, vllm_image=cfg.vllm_image)

    # Ensure metadata server + PVC
    _run([*k8s.kubectl, "apply", "-k", "deploy/metadata-server/"])
    k8s.apply_raw(f"{cfg.deploy_dir}/pvc.yaml")

    k8s.delete_template(yaml_path, **tmpl_kwargs)
    time.sleep(2)
    k8s.apply_template(yaml_path, **tmpl_kwargs)

    roles = ["scale-seed", "scale-c1", "scale-c2"]
    log_paths: dict[str, Path] = {}
    for role in roles:
        log_paths[role] = log_dir / f"{role}.log"
        k8s.start_stern(role, log_paths[role])

    # Sequential: init containers enforce ordering
    info("Waiting for scaling seed...")
    k8s.wait_for_ready("scale-seed", timeout=900)
    info("Waiting for scaling consumer-1 (NIXL + compile cache populate)...")
    k8s.wait_for_ready("scale-c1", timeout=600)
    info("Waiting for scaling consumer-2 (NIXL + compile cache hits)...")
    k8s.wait_for_ready("scale-c2", timeout=600)

    # Let stern flush
    time.sleep(3)

    results: list[ScenarioResult] = []
    for role in roles:
        label = f"Scale {role.split('-', 1)[1]}"
        method = "hf-download" if "seed" in role else "nixl-gpudirect"
        r = ScenarioResult(scenario=label, method=method, log_file=str(log_paths[role]))

        pod = k8s.get_pod(role)
        if pod:
            r.pod_name = pod
            r.ttr_s = k8s.get_k8s_ttr(pod)
            r.hardware = HardwareContext.from_pod(pod, k8s.kubectl)

        r.metrics = extract_metrics(log_paths[role])

        # Compile cache stats
        cc_stats = k8s.scrape_compile_cache_stats()
        if cc_stats:
            r.metrics.compile_cache_gets = cc_stats.get("gets", 0)
            r.metrics.compile_cache_hits = sum(
                cc_stats.get(k, 0)
                for k in ("mem_hits", "disk_hits", "peer_hits")
            )

        info(f"{label}: TTR={r.ttr_s}s, cc {r.metrics.compile_cache_hits}/{r.metrics.compile_cache_gets}")
        results.append(r)

    k8s.stop_stern()
    k8s.delete_template(yaml_path, **tmpl_kwargs)
    return results


def reprocess_logs(log_dir: Path) -> list[ScenarioResult]:
    """Re-extract metrics from existing log files (no K8s interaction)."""
    results: list[ScenarioResult] = []

    # Map log file names to scenario metadata
    log_map: dict[str, tuple[str, str]] = {
        "hf-cold.log": ("HF Cold", "hf-download"),
        "nfs-cached.log": ("NFS Cached", "nfs-read"),
        "nixl-consumer.log": ("NIXL P2P", "nixl-gpudirect"),
        "scale-seed.log": ("Scale seed", "hf-download"),
        "scale-c1.log": ("Scale c1", "nixl-gpudirect"),
        "scale-c2.log": ("Scale c2", "nixl-gpudirect"),
    }

    for filename, (scenario, method) in log_map.items():
        log_path = log_dir / filename
        if not log_path.exists():
            continue

        r = ScenarioResult(scenario=scenario, method=method, log_file=str(log_path))
        r.metrics = extract_metrics(log_path)
        results.append(r)
        info(f"Re-processed {filename}: weight_load={r.metrics.weight_load_s}s, gbps={r.metrics.transfer_gbps}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    model: str
    vllm_image: str
    context: str
    namespace: str
    deploy_dir: str
    skip_hf: bool = False
    skip_nfs: bool = False
    skip_nixl: bool = False
    skip_nixl_scaling: bool = False
    skip_populate: bool = False


def info(msg: str) -> None:
    print(f"==> {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layercast model load benchmarks")
    p.add_argument("--model", default=os.environ.get("BENCH_MODEL", "Qwen/Qwen2.5-32B"))
    p.add_argument("--vllm-image", default=os.environ.get(
        "VLLM_IMAGE", "ghcr.io/wseaton/layercast:vllm-plugin-main"))
    p.add_argument("--context", default=os.environ.get("KUBE_CONTEXT", "coreweave-waldorf"))
    p.add_argument("--namespace", default=os.environ.get("BENCH_NS", "layercast"))
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--skip-nfs", action="store_true")
    p.add_argument("--skip-nixl", action="store_true")
    p.add_argument("--skip-nixl-scaling", action="store_true")
    p.add_argument("--skip-populate", action="store_true")
    p.add_argument("--cleanup", action="store_true", help="Tear down all benchmark resources")
    p.add_argument("--from-logs", type=str, help="Re-process existing log directory (no deploy)")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deploy_dir = "deploy/benchmark"

    k8s = K8sRunner(args.context, args.namespace)

    # Cleanup mode
    if args.cleanup:
        info("Cleaning up all benchmark resources...")
        tmpl = dict(model=args.model, vllm_image=args.vllm_image)
        for yaml_name in ["nixl-p2p.yaml", "nixl-scaling.yaml", "nfs-cached.yaml", "hf-cold.yaml"]:
            k8s.delete_template(f"{deploy_dir}/{yaml_name}", **tmpl)
        k8s.delete_raw("job", "populate-model")
        try:
            _run([*k8s.kubectl, "delete", "-f", f"{deploy_dir}/pvc.yaml", "--ignore-not-found"])
        except SubprocessError:
            pass
        info("Cleanup complete.")
        return

    # Re-process existing logs
    if args.from_logs:
        log_dir = Path(args.from_logs)
        if not log_dir.is_dir():
            print(f"Error: {log_dir} is not a directory", file=sys.stderr)
            sys.exit(1)

        results = reprocess_logs(log_dir)
        run = BenchmarkRun(
            model=args.model,
            cluster=args.context,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            vllm_image=args.vllm_image,
            log_dir=str(log_dir),
            results=results,
        )

        print(format_markdown_table(run))
        save_json_results(run, log_dir)
        if not args.no_charts:
            charts = generate_charts(run, log_dir)
            if charts:
                info(f"Generated {len(charts)} chart(s)")
        return

    # Full benchmark run
    cfg = BenchConfig(
        model=args.model,
        vllm_image=args.vllm_image,
        context=args.context,
        namespace=args.namespace,
        deploy_dir=deploy_dir,
        skip_hf=args.skip_hf,
        skip_nfs=args.skip_nfs,
        skip_nixl=args.skip_nixl,
        skip_nixl_scaling=args.skip_nixl_scaling,
        skip_populate=args.skip_populate,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(f"benchmark-logs/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    info("Benchmark configuration:")
    info(f"  Model:      {cfg.model}")
    info(f"  vLLM image: {cfg.vllm_image}")
    info(f"  Context:    {cfg.context}")
    info(f"  Namespace:  {cfg.namespace}")
    info(f"  Log dir:    {log_dir}")

    k8s.ensure_namespace()

    # Populate NFS PVC
    if not cfg.skip_populate and (not cfg.skip_nfs or not cfg.skip_nixl):
        info("Ensuring PVC exists...")
        k8s.apply_raw(f"{deploy_dir}/pvc.yaml")
        info(f"Populating NFS PVC with {cfg.model}...")
        k8s.delete_raw("job", "populate-model")
        k8s.apply_template(
            f"{deploy_dir}/populate-job.yaml",
            model=cfg.model, vllm_image=cfg.vllm_image,
        )
        info("Waiting for populate job...")
        if not k8s.wait_job("populate-model"):
            warn("Populate job may still be running or failed")

    results: list[ScenarioResult] = []

    try:
        if not cfg.skip_hf:
            results.append(run_hf_cold(k8s, cfg, log_dir))

        if not cfg.skip_nfs:
            results.append(run_nfs_cached(k8s, cfg, log_dir))

        if not cfg.skip_nixl:
            results.append(run_nixl_p2p(k8s, cfg, log_dir))

        if not cfg.skip_nixl_scaling:
            results.extend(run_nixl_scaling(k8s, cfg, log_dir))
    finally:
        k8s.stop_stern()

    run = BenchmarkRun(
        model=cfg.model,
        cluster=cfg.context,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vllm_image=cfg.vllm_image,
        log_dir=str(log_dir),
        results=results,
    )

    print(format_markdown_table(run))
    save_json_results(run, log_dir)
    if not args.no_charts:
        charts = generate_charts(run, log_dir)
        if charts:
            info(f"Generated {len(charts)} chart(s)")

    info(f"Logs saved to {log_dir}/")
    info("Done.")


if __name__ == "__main__":
    main()
