"""vLLM plugin registration for layercast.

This module is the entry point referenced by pyproject.toml:

    [project.entry-points."vllm.general_plugins"]
    layercast = "vllm_layercast.plugin:register"

vLLM calls register() in every process (main, workers, etc.) during
startup. It must be re-entrant and fast.
"""

from __future__ import annotations

import os
import traceback

from vllm_layercast.log import get_logger, setup as setup_logging

_LOAD_FORMAT = "layercast"

setup_logging()
log = get_logger("vllm_layercast.plugin")


def register() -> None:
    """Register the layercast model loader with vLLM.

    Called automatically by vLLM's plugin system on startup. Registers
    our custom BaseModelLoader subclass under the "layercast" load format.

    Users activate it by passing --load-format=layercast to vLLM, or by
    setting the VLLM_LOAD_FORMAT=layercast environment variable.
    """
    log.info("register_called", pid=os.getpid())

    try:
        from vllm.model_executor.model_loader import register_model_loader
    except ImportError:
        log.warning("vllm_import_failed", component="register_model_loader")
        return

    try:
        from vllm_layercast.loader import LayercastModelLoader
    except Exception:
        log.error("loader_import_failed", traceback=traceback.format_exc())
        return

    register_model_loader(_LOAD_FORMAT)(LayercastModelLoader)

    server_addr = os.environ.get(
        "LAYERCAST_SERVER_ADDR", "layercast-metadata-server:50051"
    )
    log.info(
        "registered",
        load_format=_LOAD_FORMAT,
        server=server_addr,
    )
