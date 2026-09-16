from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from microllm.engine.engine import Engine
from microllm.engine.sampler import SamplingParams
from microllm.server.batching import BatchingEngine
from microllm.server.protocol import (
    ChatCompletionRequest,
    GenerateRequest,
    GenerateResponse,
)


def create_app(
    model: str,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    backend: str = "microllm",
    kv_mode: str = "paged",
    stage: str | int | None = None,
    kernel_impl: str = "torch",
    use_pinned_memory: bool = False,
    use_transfer_stream: bool = False,
    max_batch_size: int = 8,
    batch_wait_ms: float = 2.0,
    max_batched_tokens: int = 2048,
    max_queue_size: int | None = None,
    max_prompt_chars: int | None = None,
    enable_chunked_prefill: bool = False,
    cache_aware_scheduling: bool = False,
) -> FastAPI:
    engine = Engine(
        model=model,
        dtype=dtype,
        device=device,
        backend=backend,
        kv_mode=kv_mode,
        stage=stage,
        kernel_impl=kernel_impl,
        use_pinned_memory=use_pinned_memory,
        use_transfer_stream=use_transfer_stream,
    )
    batching_engine = BatchingEngine(
        engine,
        max_batch_size=max_batch_size,
        batch_wait_ms=batch_wait_ms,
        max_num_batched_tokens=max_batched_tokens,
        max_queue_size=max_queue_size,
        max_prompt_chars=max_prompt_chars,
        enable_chunked_prefill=enable_chunked_prefill,
        cache_aware_scheduling=cache_aware_scheduling,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await batching_engine.start()
        try:
            yield
        finally:
            await batching_engine.stop()

    app = FastAPI(title="microLLM", lifespan=lifespan)
    app.state.engine = engine
    app.state.batching_engine = batching_engine

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": model,
            "backend": backend,
            "kv_mode": kv_mode,
            "engine": engine.info(),
            "max_batch_size": max_batch_size,
            "max_batched_tokens": max_batched_tokens,
            "batching": batching_engine.stats_dict(),
        }

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        params = _sampling_params(request.sampling_params)
        if request.stream:
            return StreamingResponse(
                _sse(batching_engine.stream(request.prompt, params)),
                media_type="text/event-stream",
            )
        result = await batching_engine.generate(request.prompt, params)
        return GenerateResponse(**result)

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest):
        params = _sampling_params(request.sampling_params)
        messages = [message.model_dump() for message in request.messages]
        if request.stream:
            prompt = engine.backend.apply_chat_template(messages)
            return StreamingResponse(
                _sse(batching_engine.stream(prompt, params)),
                media_type="text/event-stream",
            )
        prompt = engine.backend.apply_chat_template(messages)
        result = await batching_engine.generate(prompt, params)
        return GenerateResponse(**result)

    return app


def _sampling_params(params) -> SamplingParams:
    return SamplingParams(
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        top_k=params.top_k,
        seed=params.seed,
    )


async def _sse(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as exc:
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else f"inference failed: {type(exc).__name__}"
        yield f"data: {json.dumps({'event': 'error', 'message': message}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", default="microllm", choices=["reference", "microllm", "hf", "paged"])
    parser.add_argument("--kv-mode", default="paged", choices=["dense", "paged"])
    parser.add_argument("--stage", default=None, help="course stage such as week04 or 4")
    parser.add_argument("--kernel-impl", default="torch", choices=["torch", "auto", "cuda"])
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--transfer-stream", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-queue-size", type=int, default=None)
    parser.add_argument("--max-prompt-chars", type=int, default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--cache-aware-scheduling", action="store_true")
    args = parser.parse_args()
    app = create_app(
        args.model,
        dtype=args.dtype,
        device=args.device,
        backend=args.backend,
        kv_mode=args.kv_mode,
        stage=args.stage,
        kernel_impl=args.kernel_impl,
        use_pinned_memory=args.pinned_memory,
        use_transfer_stream=args.transfer_stream,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        max_batched_tokens=args.max_batched_tokens,
        max_queue_size=args.max_queue_size,
        max_prompt_chars=args.max_prompt_chars,
        enable_chunked_prefill=args.enable_chunked_prefill,
        cache_aware_scheduling=args.cache_aware_scheduling,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
