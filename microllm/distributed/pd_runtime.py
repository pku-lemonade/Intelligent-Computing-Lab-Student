from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import torch

from microllm.engine.request import Sequence
from microllm.model.qwen3_torch import Qwen3Config


class PDPhase(str, Enum):
    WAITING_PREFILL = "waiting_prefill"
    PREFILLING = "prefilling"
    TRANSFERRING = "transferring"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass(slots=True)
class PDRequestState:
    sequence: Sequence
    decode_tokens: int
    arrival_s: float = 0.0
    phase: PDPhase = PDPhase.WAITING_PREFILL
    history: list[tuple[PDPhase, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.decode_tokens <= 0:
            raise ValueError("decode_tokens must be positive")
        if not math.isfinite(self.arrival_s) or self.arrival_s < 0:
            raise ValueError("arrival_s must be finite and non-negative")
        self.history.append((self.phase, self.arrival_s))

    @property
    def request_id(self) -> int:
        return self.sequence.request_id

    def advance(self, next_phase: PDPhase, *, at_s: float) -> None:
        # TODO_BEGIN(W15_T01)
        raise NotImplementedError("implement the P/D state transition")
        # TODO_END(W15_T01)


@dataclass(frozen=True, slots=True)
class KVTransferPlan:
    request_id: int
    tokens: int
    bytes: int
    bandwidth_gbps: float
    duration_s: float

    @classmethod
    def for_request(
        cls,
        state: PDRequestState,
        config: Qwen3Config,
        *,
        dtype: torch.dtype,
        bandwidth_gbps: float,
    ) -> "KVTransferPlan":
        # TODO_BEGIN(W15_T02)
        raise NotImplementedError("implement the KV transfer plan")
        # TODO_END(W15_T02)


@dataclass(frozen=True, slots=True)
class PDWorkload:
    state: PDRequestState
    transfer: KVTransferPlan
    prefill_duration_s: float
    decode_duration_s: float

    def __post_init__(self) -> None:
        if self.prefill_duration_s < 0 or self.decode_duration_s < 0:
            raise ValueError("stage durations must be non-negative")
        if self.transfer.request_id != self.state.request_id:
            raise ValueError("transfer plan belongs to another request")


@dataclass(frozen=True, slots=True)
class PDEvent:
    request_id: int
    phase: PDPhase
    resource: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class PDTrace:
    events: tuple[PDEvent, ...]
    completion_s: dict[int, float]
