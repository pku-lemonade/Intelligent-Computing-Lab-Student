from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SamplingParamsRequest(BaseModel):
    temperature: float = 0.8
    max_tokens: int | None = None
    top_k: int | None = None
    seed: int | None = None


class GenerateRequest(BaseModel):
    prompt: str
    sampling_params: SamplingParamsRequest = Field(default_factory=SamplingParamsRequest)
    stream: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    sampling_params: SamplingParamsRequest = Field(default_factory=SamplingParamsRequest)
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    token_ids: list[int]
    finish_reason: str
