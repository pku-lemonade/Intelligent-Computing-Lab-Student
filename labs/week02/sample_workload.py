"""Model-free CUDA workload used to create the Week 02 sample reports.

It deliberately exposes several kinds of timeline evidence: host-to-device
and device-to-host copies, two streams, matrix/elementwise kernels, NVTX
phase ranges, and an explicit synchronization gap.
"""
from __future__ import annotations

import torch


def run_request(index: int, device: torch.device) -> None:
    host = torch.randn((1024, 1024), dtype=torch.float32, pin_memory=True)
    weights = torch.randn((1024, 1024), device=device)
    transfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    with torch.cuda.nvtx.range(f"course::request::{index}"):
        with torch.cuda.stream(transfer_stream), torch.cuda.nvtx.range("course::h2d"):
            device_input = host.to(device, non_blocking=True)
        transfer_done = torch.cuda.Event()
        transfer_done.record(transfer_stream)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(transfer_done)
            with torch.cuda.nvtx.range("course::prefill"):
                hidden = torch.mm(device_input, weights)
                hidden = torch.relu(hidden)
            with torch.cuda.nvtx.range("course::decode"):
                logits = torch.mm(hidden, weights.t())
                probs = torch.softmax(logits[0], dim=-1)
            with torch.cuda.nvtx.range("course::sampling"):
                _ = torch.argmax(probs)
        # Keep one visible synchronization boundary so students can discuss
        # API-to-GPU delay and the cost of waiting for a stream.
        with torch.cuda.nvtx.range("course::sync"):
            compute_stream.synchronize()
        with torch.cuda.stream(transfer_stream), torch.cuda.nvtx.range("course::d2h"):
            _ = hidden[:1].to("cpu", non_blocking=False)


def main() -> None:
    device = torch.device("cuda")
    for index in range(3):
        run_request(index, device)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
