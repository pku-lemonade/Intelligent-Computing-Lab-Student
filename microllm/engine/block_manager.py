from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from math import ceil


@dataclass(slots=True)
class Block:
    block_id: int
    ref_count: int = 0
    token_hash: int | None = None
    token_ids: tuple[int, ...] | None = None

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


@dataclass(slots=True)
class BlockTable:
    block_size: int
    block_ids: list[int] = field(default_factory=list)
    owned_block_ids: set[int] = field(default_factory=set)
    length: int = 0

    def required_blocks(self, new_length: int) -> int:
        return ceil(new_length / self.block_size) if new_length > 0 else 0


class BlockManager:
    """Small vLLM-style block allocator for paged KV experiments."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.block_size = block_size
        self.blocks = [Block(block_id=i) for i in range(num_blocks)]
        self.free_block_ids = list(range(num_blocks))
        self.tables: dict[int, BlockTable] = {}
        self.hash_to_block_id: dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return len(self.blocks) - self.num_free_blocks

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)

    def allocate(self, seq_id: int, num_tokens: int, *, token_ids: list[int] | None = None) -> BlockTable:
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already has a block table")
        table = BlockTable(block_size=self.block_size)
        self.tables[seq_id] = table
        try:
            if token_ids is None:
                self.ensure_capacity(seq_id, num_tokens)
            else:
                if len(token_ids) != num_tokens:
                    raise ValueError("token_ids length must match num_tokens")
                self._allocate_with_prefix_cache(table, token_ids)
        except Exception:
            self.release(seq_id)
            raise
        table.length = num_tokens
        return table

    def ensure_capacity(self, seq_id: int, new_length: int) -> BlockTable:
        # TODO_BEGIN(W12_T01)
        return self.tables[seq_id]
        # TODO_END(W12_T01)

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> BlockTable:
        table = self.tables[seq_id]
        return self.ensure_capacity(seq_id, table.length + num_new_tokens)

    def block_table(self, seq_id: int) -> list[int]:
        return list(self.tables[seq_id].block_ids)

    def slot_mapping(self, seq_id: int, positions: list[int]) -> list[int]:
        # TODO_BEGIN(W12_T02)
        return list(positions)
        # TODO_END(W12_T02)

    def release(self, seq_id: int) -> None:
        table = self.tables.pop(seq_id)
        for block_id in table.block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError(f"negative ref_count for block {block_id}")
            if block.ref_count == 0:
                self.free_block_ids.append(block_id)

    def usage_stats(self) -> dict:
        allocated_slots = sum(len(table.block_ids) * self.block_size for table in self.tables.values())
        live_tokens = sum(table.length for table in self.tables.values())
        wasted_slots = allocated_slots - live_tokens
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.num_used_blocks,
            "block_size": self.block_size,
            "active_sequences": len(self.tables),
            "live_tokens": live_tokens,
            "allocated_slots": allocated_slots,
            "wasted_slots": wasted_slots,
            "fragmentation_ratio": wasted_slots / allocated_slots if allocated_slots else 0.0,
            "prefix_cached_blocks": sum(1 for block in self.blocks if block.ref_count > 1),
            "cached_free_blocks": sum(1 for block in self.blocks if block.ref_count == 0 and block.token_hash is not None),
        }

    def _allocate_with_prefix_cache(self, table: BlockTable, token_ids: list[int]) -> None:
        prefix_hash = -1
        for start in range(0, len(token_ids), self.block_size):
            block_tokens = tuple(token_ids[start : start + self.block_size])
            if len(block_tokens) == self.block_size:
                token_hash = self._block_hash(block_tokens, prefix_hash)
                cached_block_id = self.hash_to_block_id.get(token_hash)
                if cached_block_id is not None and self.blocks[cached_block_id].token_ids == block_tokens:
                    self._claim_cached_block(cached_block_id)
                    table.block_ids.append(cached_block_id)
                    prefix_hash = token_hash
                    continue
                block_id = self._allocate_fresh_block()
                block = self.blocks[block_id]
                block.token_hash = token_hash
                block.token_ids = block_tokens
                self.hash_to_block_id[token_hash] = block_id
                table.block_ids.append(block_id)
                table.owned_block_ids.add(block_id)
                prefix_hash = token_hash
            else:
                block_id = self._allocate_fresh_block()
                table.block_ids.append(block_id)
                table.owned_block_ids.add(block_id)

    def _allocate_fresh_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("not enough KV blocks: need 1, free 0")
        free_index = None
        for index in range(len(self.free_block_ids) - 1, -1, -1):
            if self.blocks[self.free_block_ids[index]].token_hash is None:
                free_index = index
                break
        if free_index is None:
            block_id = self.free_block_ids.pop()
        else:
            block_id = self.free_block_ids.pop(free_index)
        block = self.blocks[block_id]
        if block.token_hash is not None and self.hash_to_block_id.get(block.token_hash) == block_id:
            del self.hash_to_block_id[block.token_hash]
        block.ref_count = 1
        block.token_hash = None
        block.token_ids = None
        return block_id

    def _claim_cached_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.free_block_ids.remove(block_id)
            block.ref_count = 1
        else:
            block.ref_count += 1

    def _block_hash(self, token_ids: tuple[int, ...], prefix_hash: int) -> int:
        hasher = hashlib.blake2b(digest_size=8)
        # Prefix hashes are unsigned 64-bit values; signed serialization
        # overflows once a long prefix produces a high bit.
        hasher.update((prefix_hash & ((1 << 64) - 1)).to_bytes(8, "little", signed=False))
        for token_id in token_ids:
            hasher.update(int(token_id).to_bytes(8, "little", signed=True))
        return int.from_bytes(hasher.digest(), "little")
