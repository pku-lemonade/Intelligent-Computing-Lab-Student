from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from microllm.model.qwen3_torch import Qwen3Config


class AFStage(str, Enum):
    ATTENTION = "attention"
    TO_FFN = "to_ffn"
    FFN = "ffn"
    TO_ATTENTION = "to_attention"


@dataclass(frozen=True, slots=True)
class AFTask:
    task_id: str
    flow_id: str
    layer_id: int
    stage: AFStage
    duration_s: float
    transfer_bytes: int = 0
    depends_on: tuple[str, ...] = ()


def _make_af_task(
    *,
    prefix: str,
    flow_id: str,
    layer_id: int,
    stage: AFStage,
    duration_s: float,
    transfer_bytes: int = 0,
    previous: AFTask | None = None,
) -> AFTask:
    return AFTask(
        task_id=f"{prefix}:{stage.value}",
        flow_id=flow_id,
        layer_id=layer_id,
        stage=stage,
        duration_s=duration_s,
        transfer_bytes=transfer_bytes,
        depends_on=(previous.task_id,) if previous is not None else (),
    )


def hidden_state_transfer_bytes(*, tokens: int, hidden_size: int, dtype: torch.dtype) -> int:
    if tokens < 0 or hidden_size <= 0:
        raise ValueError("tokens must be non-negative and hidden_size must be positive")
    return tokens * hidden_size * torch.empty((), dtype=dtype).element_size()


def partition_decoder_layer(
    config: Qwen3Config,
    *,
    layer_id: int,
    tokens: int,
    dtype: torch.dtype,
    attention_duration_s: float,
    ffn_duration_s: float,
    bandwidth_gbps: float,
    flow_id: str = "flow0",
) -> list[AFTask]:
    if layer_id < 0 or layer_id >= config.num_hidden_layers:
        raise ValueError("layer_id is outside the model")
    if attention_duration_s < 0 or ffn_duration_s < 0:
        raise ValueError("compute durations must be non-negative")
    if bandwidth_gbps <= 0 or not math.isfinite(bandwidth_gbps):
        raise ValueError("bandwidth_gbps must be finite and positive")
    if not flow_id:
        raise ValueError("flow_id must not be empty")
    transfer_bytes = hidden_state_transfer_bytes(tokens=tokens, hidden_size=config.hidden_size, dtype=dtype)
    transfer_duration = transfer_bytes / (bandwidth_gbps * 1_000_000_000)
    prefix = f"{flow_id}:layer{layer_id}"
    specs = (
        (AFStage.ATTENTION, attention_duration_s, 0),
        (AFStage.TO_FFN, transfer_duration, transfer_bytes),
        (AFStage.FFN, ffn_duration_s, 0),
        (AFStage.TO_ATTENTION, transfer_duration, transfer_bytes),
    )
    tasks: list[AFTask] = []
    for stage, duration_s, stage_transfer_bytes in specs:
        tasks.append(
            _make_af_task(
                prefix=prefix,
                flow_id=flow_id,
                layer_id=layer_id,
                stage=stage,
                duration_s=duration_s,
                transfer_bytes=stage_transfer_bytes,
                previous=tasks[-1] if tasks else None,
            )
        )
    return tasks


@dataclass(frozen=True, slots=True)
class AFEvent:
    task_id: str
    flow_id: str
    layer_id: int
    stage: AFStage
    resource: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class AFTrace:
    events: tuple[AFEvent, ...]
    completion_s: float
