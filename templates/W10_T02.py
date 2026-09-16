from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TransferTicket:
    tensor: torch.Tensor
    ready_event: torch.cuda.Event | None
    slot: int


class TransferPipeline:
    def __init__(self, device: str | torch.device, *, num_slots: int = 2):
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.device = torch.device(device)
        self.num_slots = num_slots
        self.transfer_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._host_buffers: list[torch.Tensor | None] = [None] * num_slots
        self._ready_events: list[torch.cuda.Event | None] = [None] * num_slots
        self._next_slot = 0

    def submit(self, tensor: torch.Tensor) -> TransferTicket:
        # TODO_BEGIN(W10_T02)
        return TransferTicket(tensor=tensor.to(self.device), ready_event=None, slot=-1)
        # TODO_END(W10_T02)

    def wait(
        self,
        ticket: TransferTicket,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if ticket.ready_event is None:
            return ticket.tensor
        consumer = stream or torch.cuda.current_stream(self.device)
        consumer.wait_event(ticket.ready_event)
        return ticket.tensor
