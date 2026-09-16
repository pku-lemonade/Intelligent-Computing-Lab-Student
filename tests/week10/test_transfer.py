from __future__ import annotations

import torch

from microllm.model.qwen3_continuous_backend import Qwen3TorchBackend
from microllm.runtime import TransferPipeline


class _FakeTensor:
    def __init__(self):
        self.calls = []

    def pin_memory(self):
        self.calls.append(("pin_memory",))
        return self

    def to(self, device, *, non_blocking=False):
        self.calls.append(("to", device, non_blocking))
        return "moved"


def test_w10_t01_to_device_uses_pinned_non_blocking_copy():
    backend = object.__new__(Qwen3TorchBackend)
    backend.device = torch.device("cuda")
    backend.use_pinned_memory = True
    backend.transfer_stream = None
    tensor = _FakeTensor()

    result = backend._to_device(tensor)

    assert result == "moved"
    assert tensor.calls == [
        ("pin_memory",),
        ("to", torch.device("cuda"), True),
    ]


def test_w10_t01_to_device_uses_transfer_stream_safely():
    _require_cuda()
    backend = object.__new__(Qwen3TorchBackend)
    backend.device = torch.device("cuda")
    backend.use_pinned_memory = True
    backend.transfer_stream = torch.cuda.Stream()
    source = torch.randn(4096)

    result = backend._to_device(source)
    torch.cuda.current_stream().synchronize()

    assert result.device.type == "cuda"
    assert torch.equal(result.cpu(), source)


def test_w10_t02_transfer_pipeline_cycles_safe_slots():
    _require_cuda()
    pipeline = TransferPipeline("cuda", num_slots=2)
    inputs = [torch.arange(1024, dtype=torch.float32) + offset for offset in range(3)]
    tickets = [pipeline.submit(value) for value in inputs]

    assert [ticket.slot for ticket in tickets] == [0, 1, 0]
    assert all(ticket.ready_event is not None for ticket in tickets)
    assert all(buffer is not None and buffer.is_pinned() for buffer in pipeline._host_buffers)
    outputs = [pipeline.wait(ticket) for ticket in tickets]
    torch.cuda.current_stream().synchronize()
    for output, expected in zip(outputs, inputs):
        assert torch.equal(output.cpu(), expected)

    with torch.no_grad():
        gpu_input = torch.ones(4, device="cuda")
    try:
        pipeline.submit(gpu_input)
    except ValueError as exc:
        assert "CPU source" in str(exc)
    else:
        raise AssertionError("GPU source tensor must be rejected")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 10 public tests require CUDA")
