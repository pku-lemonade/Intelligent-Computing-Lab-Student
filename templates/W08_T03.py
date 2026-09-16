from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from microllm.kernels.errors import KernelUnavailable
from microllm.model.model_path import resolve_local_model_path
from microllm.model.ops import CourseLinear, dense_attention_decode, dense_attention_prefill


@dataclass(slots=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool = False
    hidden_act: str = "silu"

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        data = json.loads(Path(path).read_text())
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
            max_position_embeddings=data["max_position_embeddings"],
            rms_norm_eps=data["rms_norm_eps"],
            rope_theta=data["rope_theta"],
            attention_bias=data.get("attention_bias", False),
            hidden_act=data.get("hidden_act", "silu"),
        )


@dataclass(slots=True)
class Qwen3KVCache:
    key_values: list[tuple[torch.Tensor, torch.Tensor]]
    seq_len: int


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, *, kernel_impl: str = "torch"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.kernel_impl = kernel_impl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO_BEGIN(W08_T03)
        return x
        # TODO_END(W08_T03)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    kernel_impl: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    if kernel_impl in {"auto", "cuda"} and q.is_cuda:
        try:
            q_out = _cuda_rope_nd(q, cos, sin)
            k_out = _cuda_rope_nd(k, cos, sin)
            return q_out, k_out
        except KernelUnavailable:
            if kernel_impl == "cuda":
                raise
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _cuda_rope_nd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor | None = None) -> torch.Tensor:
    if sin is None:
        raise RuntimeError("internal error: sin must be provided")
    from microllm.kernels.cuda_ops import cuda_rope

    expanded_cos = cos.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    expanded_sin = sin.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_cos = expanded_cos.reshape(-1, x.shape[-1])
    flat_sin = expanded_sin.reshape(-1, x.shape[-1])
    return cuda_rope(flat_x, flat_cos, flat_sin).reshape_as(x)


def repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    if num_groups == 1:
        return x
    x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, num_groups, seq_len, head_dim)
    return x.reshape(batch_size, num_kv_heads * num_groups, seq_len, head_dim)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].to(x.device)
        position_ids = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ position_ids.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, *, kernel_impl: str = "torch"):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported activation: {config.hidden_act}")
        self.gate_proj = CourseLinear(config.hidden_size, config.intermediate_size, bias=False, kernel_impl=kernel_impl)
        self.up_proj = CourseLinear(config.hidden_size, config.intermediate_size, bias=False, kernel_impl=kernel_impl)
        self.down_proj = CourseLinear(config.intermediate_size, config.hidden_size, bias=False, kernel_impl=kernel_impl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, *, kernel_impl: str = "torch"):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.kernel_impl = kernel_impl

        self.q_proj = CourseLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
            kernel_impl=kernel_impl,
        )
        self.k_proj = CourseLinear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
            kernel_impl=kernel_impl,
        )
        self.v_proj = CourseLinear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
            kernel_impl=kernel_impl,
        )
        self.o_proj = CourseLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            kernel_impl=kernel_impl,
        )
        self.q_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps, kernel_impl=kernel_impl)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps, kernel_impl=kernel_impl)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin, kernel_impl=self.kernel_impl)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        present_key_value = (k, v) if use_cache else None

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        if past_key_value is None:
            attn_output = dense_attention_prefill(
                q,
                k,
                v,
                scale=self.scaling,
                kernel_impl=self.kernel_impl,
            )
        else:
            key_len = k.shape[-2]
            key_positions = torch.arange(key_len, dtype=position_ids.dtype, device=x.device)
            causal_mask = key_positions.view(1, 1, key_len) > position_ids.view(batch_size, seq_len, 1)
            if seq_len == 1 and not causal_mask.any():
                attn_output = dense_attention_decode(
                    q.squeeze(-2),
                    k,
                    v,
                    scale=self.scaling,
                    kernel_impl=self.kernel_impl,
                ).unsqueeze(-2)
            else:
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
                attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(1), torch.finfo(attn_scores.dtype).min)
                attn_weights = F.softmax(attn_scores.float(), dim=-1).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, *, kernel_impl: str = "torch"):
        super().__init__()
        self.self_attn = Qwen3Attention(config, kernel_impl=kernel_impl)
        self.mlp = Qwen3MLP(config, kernel_impl=kernel_impl)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, kernel_impl=kernel_impl)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, kernel_impl=kernel_impl)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            x,
            cos,
            sin,
            position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x, present_key_value


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config, *, kernel_impl: str = "torch"):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, kernel_impl=kernel_impl) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, kernel_impl=kernel_impl)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Qwen3KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Qwen3KVCache | None]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        past_seq_len = past_key_values.seq_len if past_key_values is not None else 0
        position_ids = torch.arange(
            past_seq_len,
            past_seq_len + seq_len,
            dtype=torch.long,
            device=input_ids.device,
        ).expand(batch_size, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        new_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_id, layer in enumerate(self.layers):
            past_key_value = None
            if past_key_values is not None:
                past_key_value = past_key_values.key_values[layer_id]
            hidden_states, present_key_value = layer(
                hidden_states,
                cos,
                sin,
                position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if present_key_value is not None:
                new_key_values.append(present_key_value)

        cache = Qwen3KVCache(key_values=new_key_values, seq_len=past_seq_len + seq_len) if use_cache else None
        return self.norm(hidden_states), cache


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config, *, kernel_impl: str = "torch"):
        super().__init__()
        if kernel_impl not in {"torch", "auto", "cuda"}:
            raise ValueError("kernel_impl must be one of: torch, auto, cuda")
        self.config = config
        self.kernel_impl = kernel_impl
        self.model = Qwen3Model(config, kernel_impl=kernel_impl)
        self.lm_head = CourseLinear(config.hidden_size, config.vocab_size, bias=False, kernel_impl=kernel_impl)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        kernel_impl: str = "torch",
    ) -> "Qwen3ForCausalLM":
        model_path = resolve_local_model_path(model_path)
        config = Qwen3Config.from_json(model_path / "config.json")
        model = cls(config, kernel_impl=kernel_impl)
        model.to(dtype=dtype)
        state_dict = load_file(model_path / "model.safetensors", device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if key != "lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(f"failed to load Qwen3 weights: missing={missing}, unexpected={unexpected}")
        if "lm_head.weight" not in state_dict:
            model.lm_head.weight = model.model.embed_tokens.weight
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device=device)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.model(input_ids)
        return self.lm_head(hidden_states)

    @torch.inference_mode()
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Qwen3KVCache | None = None,
    ) -> tuple[torch.Tensor, Qwen3KVCache]:
        hidden_states, cache = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        if cache is None:
            raise RuntimeError("cache was not returned")
        return self.lm_head(hidden_states), cache
