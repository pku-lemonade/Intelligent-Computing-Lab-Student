from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    past_key_values: object | None


@dataclass(slots=True)
class BatchModelOutput:
    logits: list[torch.Tensor]
    past_key_values: list[object | None]


def bucket_by_length(batch_token_ids: list[list[int]]) -> dict[int, list[int]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, token_ids in enumerate(batch_token_ids):
        buckets[len(token_ids)].append(index)
    return dict(buckets)


def parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[dtype]
