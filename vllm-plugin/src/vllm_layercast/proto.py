"""Wire protocol messages for the layercast daemon IPC.

All messages are length-prefixed msgpack: 4-byte big-endian length followed
by the msgpack payload. Pydantic models mirror the Rust IPC server's
state machine message types.

State machine (per connection):
  Connected ──PrepareModel──► Preparing ──ModelLoaded──► Ready
  Ready ─────ModelUnloaded──► Connected
  Ready ─────PrepareModel───► Preparing  (hot-swap)
  Ready ─────EOF────────────► cleanup    (crash protection)
"""

from __future__ import annotations

import struct
from typing import Annotated, Literal, Union

import msgpack
from pydantic import BaseModel, Field, TypeAdapter


# Request messages (client -> daemon)


class PrepareModel(BaseModel):
    """Batch: list model files + discover NIXL peers + fetch metadata.

    Valid in: Connected, Ready (triggers hot-swap unadvertise).
    """

    type: Literal["prepare_model"] = "prepare_model"
    model_id: str
    revision: str
    tp_rank: int


class ModelLoaded(BaseModel):
    """Batch: store NIXL metadata + advertise via CRD.

    Valid in: Preparing only.
    """

    type: Literal["model_loaded"] = "model_loaded"
    agent_name: str
    metadata: bytes
    model_id: str
    files: list[str]
    tp_rank: int


class ModelUnloaded(BaseModel):
    """Explicit teardown: unadvertise + remove metadata.

    Valid in: Ready only.
    """

    type: Literal["model_unloaded"] = "model_unloaded"
    agent_name: str


RequestMessage = Union[PrepareModel, ModelLoaded, ModelUnloaded]


# Response messages (daemon -> client)


class PeerNixlMd(BaseModel):
    agent_name: str
    metadata: bytes


class Prepared(BaseModel):
    """Response to PrepareModel: safetensor file list + peer NIXL metadata."""

    type: Literal["prepared"] = "prepared"
    files: list[str]
    peers: list[PeerNixlMd]


class Ok(BaseModel):
    type: Literal["ok"] = "ok"


class Error(BaseModel):
    type: Literal["error"] = "error"
    message: str


ResponseMessage = Annotated[
    Union[Prepared, Ok, Error],
    Field(discriminator="type"),
]

_response_adapter: TypeAdapter[ResponseMessage] = TypeAdapter(ResponseMessage)


# Serialization helpers


def encode(msg: RequestMessage) -> bytes:
    """Serialize a request message to length-prefixed msgpack bytes."""
    payload = msgpack.packb(msg.model_dump(), use_bin_type=True)
    return struct.pack(">I", len(payload)) + payload


def decode_response(data: bytes) -> Prepared | Ok | Error | dict[str, object]:
    """Deserialize a daemon response from msgpack bytes."""
    raw = msgpack.unpackb(data, raw=False)
    if not isinstance(raw, dict):
        return raw

    try:
        return _response_adapter.validate_python(raw)
    except Exception:
        return raw
