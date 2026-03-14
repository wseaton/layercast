from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import grpc
import grpc.aio

from vllm_layercast.log import get_logger
from vllm_layercast.pb.layercast import (
    ModelLoaded,
    ModelLoadedRequest,
    ModelUnloaded,
    ModelUnloadedRequest,
    Prepared,
    PrepareModel,
    PrepareModelRequest,
    PrepareModelResponse,
    TensorInfo,
)

if TYPE_CHECKING:
    pass

log = get_logger("vllm_layercast.grpc")

DEFAULT_SERVER_ADDR = "layercast-metadata-server:50051"

_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 10.0

# Re-sends ModelLoaded periodically to rebuild server state after failover.
_HEARTBEAT_INTERVAL_S = 60.0


class GrpcClient:
    def __init__(self, server_addr: str = DEFAULT_SERVER_ADDR) -> None:
        self._server_addr = server_addr
        self._pod_name = os.environ.get(
            "POD_NAME", os.environ.get("HOSTNAME", "unknown")
        )
        self._pod_ip = os.environ.get("POD_IP", "")
        self._channel: grpc.aio.Channel | None = None

        # gRPC method descriptors (unary-unary)
        self._prepare_method = "/layercast.LayercastService/PrepareModel"
        self._loaded_method = "/layercast.LayercastService/ModelLoaded"
        self._unloaded_method = "/layercast.LayercastService/ModelUnloaded"

        # Heartbeat state
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._last_loaded_request: ModelLoadedRequest | None = None

    async def connect(self) -> None:
        self._channel = grpc.aio.insecure_channel(
            self._server_addr,
            options=[
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        log.info("grpc_connected", server=self._server_addr, pod=self._pod_name)

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._channel is not None:
            await self._channel.close()
            self._channel = None

    async def prepare_model(
        self,
        model_id: str,
        revision: str,
        tp_rank: int,
    ) -> Prepared:
        req = PrepareModelRequest(
            pod_name=self._pod_name,
            pod_ip=self._pod_ip,
            prepare=PrepareModel(model_id=model_id, revision=revision, tp_rank=tp_rank),
        )
        resp_bytes = await self._call_with_retry(self._prepare_method, bytes(req))
        resp = PrepareModelResponse().parse(resp_bytes)
        return resp.prepared

    async def model_loaded(
        self,
        agent_name: str,
        nixl_md: bytes,
        tensors: list[TensorInfo],
        model_id: str,
        files: list[str],
        tp_rank: int,
    ) -> None:
        req = ModelLoadedRequest(
            pod_name=self._pod_name,
            loaded=ModelLoaded(
                agent_name=agent_name,
                nixl_md=nixl_md,
                tensors=tensors,
                model_id=model_id,
                files=files,
                tp_rank=tp_rank,
            ),
        )
        await self._call_with_retry(self._loaded_method, bytes(req))

        # Stash for heartbeat re-sends
        self._last_loaded_request = req
        self._start_heartbeat()

    async def model_unloaded(self, agent_name: str) -> None:
        req = ModelUnloadedRequest(
            pod_name=self._pod_name,
            unloaded=ModelUnloaded(agent_name=agent_name),
        )
        await self._call_with_retry(self._unloaded_method, bytes(req))

        # Stop heartbeat since we no longer have an active model
        self._last_loaded_request = None
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    def _start_heartbeat(self) -> None:
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            req = self._last_loaded_request
            if req is None:
                return
            try:
                await self._call_with_retry(self._loaded_method, bytes(req))
                log.debug("heartbeat_sent", pod=self._pod_name)
            except Exception as exc:
                log.warning("heartbeat_failed", error=str(exc))

    async def _call_with_retry(self, method: str, request_bytes: bytes) -> bytes:
        channel = self._ensure_channel()
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(_MAX_RETRIES):
            try:
                resp: bytes = await channel.unary_unary(
                    method,
                    request_serializer=lambda x: x,
                    response_deserializer=lambda x: x,
                )(request_bytes)
                return resp
            except grpc.aio.AioRpcError as rpc_err:
                if rpc_err.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                ):
                    if attempt < _MAX_RETRIES - 1:
                        log.warning(
                            "grpc_retry",
                            method=method,
                            attempt=attempt + 1,
                            backoff_s=backoff,
                            code=rpc_err.code().name,
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, _MAX_BACKOFF_S)
                        continue
                raise RuntimeError(
                    f"gRPC {method} failed: {rpc_err.code().name}: {rpc_err.details()}"
                ) from rpc_err

        raise RuntimeError(f"gRPC {method} failed after {_MAX_RETRIES} retries")

    def _ensure_channel(self) -> grpc.aio.Channel:
        if self._channel is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._channel
