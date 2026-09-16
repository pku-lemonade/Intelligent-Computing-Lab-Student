from __future__ import annotations

from dataclasses import dataclass

import torch

from microllm.kernels.errors import KernelUnavailable


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.8
    max_tokens: int | None = None
    top_k: int | None = None
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0 when provided")


class Sampler:
    def __init__(self, params: SamplingParams):
        self.params = params
        self.generator = None
        if params.seed is not None:
            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(params.seed)

    def sample(self, logits: torch.Tensor) -> int:
        logits = logits.float().detach()
        if logits.ndim == 2:
            logits = logits[-1]
        if logits.ndim != 1:
            raise ValueError(f"expected 1D logits, got shape {tuple(logits.shape)}")

        if self.params.temperature == 0:
            return int(torch.argmax(logits).item())

        logits = logits / self.params.temperature
        if self.params.top_k is not None and self.params.top_k < logits.numel():
            values, indices = torch.topk(logits, self.params.top_k)
            probs = self._softmax(values)
            selected = torch.multinomial(probs.cpu(), num_samples=1, generator=self.generator)
            return int(indices[selected].item())

        probs = self._softmax(logits)
        selected = torch.multinomial(probs.cpu(), num_samples=1, generator=self.generator)
        return int(selected.item())

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.is_cuda:
            try:
                from microllm.kernels.cuda_ops import cuda_softmax

                return cuda_softmax(logits.reshape(1, -1)).reshape(-1)
            except KernelUnavailable:
                pass
        return torch.softmax(logits.cpu(), dim=-1)
