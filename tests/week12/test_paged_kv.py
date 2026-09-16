from __future__ import annotations

import torch

from microllm.engine.block_manager import BlockManager
from microllm.engine.paged_kv_cache import PagedKVCache, PagedKVConfig
from microllm.kernels import assert_close
from microllm.kernels.cuda_ops import cuda_paged_attention_decode
from microllm.model.attention import paged_attention_decode_reference


def test_w12_t01_ensure_capacity_is_atomic():
    manager = BlockManager(num_blocks=2, block_size=4)
    manager.allocate(1, 0)
    table = manager.ensure_capacity(1, 8)
    assert len(table.block_ids) == 2
    manager.allocate(2, 0)
    try:
        manager.ensure_capacity(2, 1)
    except RuntimeError as exc:
        assert "not enough KV blocks" in str(exc)
    else:
        raise AssertionError("capacity exhaustion must fail")
    assert manager.tables[2].block_ids == []


def test_w12_t02_slot_mapping_uses_physical_block_ids():
    manager = BlockManager(num_blocks=6, block_size=4)
    table = manager.allocate(3, 8)
    positions = [0, 3, 4, 7]
    expected = [
        table.block_ids[position // 4] * 4 + position % 4
        for position in positions
    ]
    assert manager.slot_mapping(3, positions) == expected


def test_w12_t03_reserve_returns_new_logical_positions():
    cache = _cache(device="cpu")
    cache.allocate(4, 2)
    assert cache.reserve(4, 3) == [2, 3, 4]
    assert cache.block_manager.tables[4].length == 5
    assert len(cache.block_table(4)) == 2


def test_w12_t04_write_keeps_layers_and_sequences_isolated():
    cache = _cache(device="cpu")
    for sequence_id, base in ((1, 10.0), (2, 20.0)):
        cache.allocate(sequence_id, 0)
        positions = cache.reserve(sequence_id, 3)
        for layer_id in (0, 1):
            value = torch.full((1, 2, 3, 4), base + layer_id)
            cache.write(sequence_id, layer_id, positions, value, value + 100)
    for sequence_id, base in ((1, 10.0), (2, 20.0)):
        for layer_id in (0, 1):
            key, value = cache.get_dense(sequence_id, layer_id)
            assert key.shape == value.shape == (1, 2, 3, 4)
            assert torch.equal(key, torch.full_like(key, base + layer_id))
            assert torch.equal(value, torch.full_like(value, base + layer_id + 100))


def test_w12_t05_paged_attention_handles_variable_context():
    if not torch.cuda.is_available():
        raise RuntimeError("Week 12 paged attention test requires CUDA")
    block_size = 4
    query = torch.randn((2, 4, 32), device="cuda")
    key_cache = torch.randn((24, 2, 32), device="cuda")
    value_cache = torch.randn_like(key_cache)
    block_tables = [[5], [3, 1]]
    context_lens = [3, 7]
    actual = cuda_paged_attention_decode(
        query, key_cache, value_cache, block_tables, context_lens, block_size
    )
    expected = paged_attention_decode_reference(
        query, key_cache, value_cache, block_tables, context_lens, block_size
    )
    assert_close("W12_T05", actual, expected, atol=1e-4, rtol=1e-4)


def _cache(*, device: str) -> PagedKVCache:
    return PagedKVCache(
        PagedKVConfig(
            num_layers=2,
            num_blocks=8,
            block_size=4,
            num_kv_heads=2,
            head_dim=4,
            device=device,
        )
    )
