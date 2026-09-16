from __future__ import annotations

import torch

from microllm.distributed import (
    AFPipeline,
    AFStage,
    hidden_state_transfer_bytes,
    partition_decoder_layer,
)
from microllm.model.qwen3_torch import Qwen3Config


def test_w15_t01_partition_preserves_decoder_dependencies():
    tasks = partition_decoder_layer(
        _config(),
        layer_id=0,
        tokens=4,
        dtype=torch.bfloat16,
        attention_duration_s=1.0,
        ffn_duration_s=2.0,
        bandwidth_gbps=1.0,
        flow_id="request0",
    )
    assert [task.stage for task in tasks] == [
        AFStage.ATTENTION,
        AFStage.TO_FFN,
        AFStage.FFN,
        AFStage.TO_ATTENTION,
    ]
    assert tasks[0].depends_on == ()
    assert tasks[1].depends_on == (tasks[0].task_id,)
    assert tasks[2].depends_on == (tasks[1].task_id,)
    assert tasks[3].depends_on == (tasks[2].task_id,)
    assert tasks[1].transfer_bytes == tasks[3].transfer_bytes == 4 * 8 * 2


def test_w15_t02_hidden_state_bytes_follow_tensor_shape():
    assert hidden_state_transfer_bytes(tokens=3, hidden_size=8, dtype=torch.bfloat16) == 48
    assert hidden_state_transfer_bytes(tokens=3, hidden_size=8, dtype=torch.float32) == 96


def test_w15_t03_pipeline_respects_resources_and_layer_order():
    tasks = []
    for flow_id in ("a", "b"):
        for layer_id in (0, 1):
            tasks.extend(
                partition_decoder_layer(
                    _config(),
                    layer_id=layer_id,
                    tokens=16,
                    dtype=torch.float32,
                    attention_duration_s=1.0,
                    ffn_duration_s=2.0,
                    bandwidth_gbps=0.001,
                    flow_id=flow_id,
                )
            )
    trace = AFPipeline().schedule(tasks)
    assert len(trace.events) == 16
    assert trace.completion_s > 0
    events = {event.task_id: event for event in trace.events}
    for task in tasks:
        for dependency in task.depends_on:
            assert events[task.task_id].start_s >= events[dependency].end_s
    for flow_id in ("a", "b"):
        assert events[f"{flow_id}:layer1:attention"].start_s >= events[f"{flow_id}:layer0:to_attention"].end_s
    for resource in ("attention", "ffn", "link"):
        selected = sorted((event for event in trace.events if event.resource == resource), key=lambda event: event.start_s)
        assert all(right.start_s >= left.end_s for left, right in zip(selected, selected[1:]))


def _config() -> Qwen3Config:
    return Qwen3Config(32, 8, 16, 2, 4, 2, 4, 128, 1e-6, 10000.0)
