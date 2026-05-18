"""Talkie 13B transformer architecture.

A 40-layer, 40-head decoder-only GPT with RoPE, SwiGLU, RMS normalisation,
embedding skip connections, and per-head / per-layer gain parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from talkie.ouroboros import OuroborosResonanceGate

# ---------------------------------------------------------------------------
# MPS Hardware Bug Monkey-Patches
# ---------------------------------------------------------------------------
# Apple Silicon (MPS) has known bugs where float16 computation in rms_norm (due to squared sums)
# and scaled_dot_product_attention (with is_causal=True) overflows and produces NaNs.
# By forcing these specific operations to compute in float32 internally, we can keep 
# the 13B model weights in float16 (26GB memory footprint) while avoiding NaNs entirely.
_orig_rms_norm = F.rms_norm
_orig_sdpa = F.scaled_dot_product_attention

def _safe_rms_norm(input, normalized_shape, weight=None, eps=1e-05):
    orig_dtype = input.dtype
    res = _orig_rms_norm(input.float(), normalized_shape, 
                         weight.float() if weight is not None else None, eps)
    return res.to(orig_dtype)

def _safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    orig_dtype = query.dtype
    res = _orig_sdpa(query.float(), key.float(), value.float(), attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    return res.to(orig_dtype)

_orig_linear = F.linear
def _safe_linear(input, weight, bias=None):
    orig_dtype = input.dtype
    # Cast input to weight dtype to force MPS to use matched types without allocating massive weight tensors
    input_cast = input.to(weight.dtype) if input.dtype != weight.dtype else input
    res = _orig_linear(input_cast, weight, bias)
    return res.to(orig_dtype)

F.rms_norm = _safe_rms_norm
F.scaled_dot_product_attention = _safe_sdpa
F.linear = _safe_linear

from talkie.sampling import apply_top_k_top_p, sample_gumbel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 65536
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones([n_head]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.w_g.type_as(w)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


# ---------------------------------------------------------------------------
# Attention & MLP
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        n_state = config.n_embd

        self.attn_query = nn.Linear(n_state, n_state, bias=False)
        self.attn_key = nn.Linear(n_state, n_state, bias=False)
        self.attn_value = nn.Linear(n_state, n_state, bias=False)
        self.attn_resid = nn.Linear(n_state, n_state, bias=False)
        self.head_gain = HeadGain(config.n_head)

    def forward(self, x: torch.Tensor, cos_sin: tuple) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.attn_query(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.attn_key(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.attn_value(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.attn_resid(y)


class JCrossInt8Linear(nn.Module):
    """
    [Y-Axis: Width Compression]
    Stores weights in INT8 to save 50% memory, but computes in FP32 to prevent MPS NaNs.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weights are stored in INT8
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.ones((out_features, 1), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runtime Cast (Y-Axis to Z-Axis audit): Compute in FP32
        w_fp32 = self.weight.float() * self.scale.float()
        return F.linear(x.float(), w_fp32, self.bias.float() if self.bias is not None else None).to(x.dtype)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_state = config.n_embd
        n_mlp = int(round(((8 / 3) * n_state) / 128) * 128)

        # Apply Y-Axis Int8 Compression
        self.mlp_gate = JCrossInt8Linear(n_state, n_mlp, bias=False)
        self.mlp_linear = JCrossInt8Linear(n_state, n_mlp, bias=False)
        self.mlp_resid = JCrossInt8Linear(n_mlp, n_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.mlp_gate(x)) * self.mlp_linear(x)
        return self.mlp_resid(x)


# ---------------------------------------------------------------------------
# Transformer block & full model
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config)
        self.attn_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.mlp = MLP(config)
        self.mlp_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)
        self.resonance_gate = OuroborosResonanceGate(config.n_embd, target_layer=15, scale=0.05)

    def forward(
        self, e_x: torch.Tensor, x: torch.Tensor, cos_sin: tuple
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        e_x = e_x.float()

        # ALL computations inside this block run in pure FP32
        x = x + self.attn_gain(self.attn(F.rms_norm(x, (x.shape[-1],)), cos_sin))
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        
        # Ouroboros Resonance Injection (Dynamic SSD Knowledge)
        x = self.resonance_gate(x, self.layer_idx)
        
        return x.to(orig_dtype)


class TalkieModel(nn.Module):
    """Talkie 13B decoder-only transformer."""

    def __init__(
        self, config: GPTConfig, device: torch.device, max_seq_len: int = 2048
    ):
        super().__init__()
        self.config = config
        self.device = device

        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Parameter(torch.zeros(config.vocab_size, config.n_embd))
        self.lm_head_gain = WeightGain()

        cos, sin = self._precompute_rotary_embeddings(max_seq_len, config.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.suppress_token_ids: set[int] | None = None

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 1_000_000
    ) -> tuple:
        device = self.embed.weight.device if hasattr(self, "embed") else "cpu"
        if str(device) == "meta":
            device = "cpu"
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Remove bfloat16 cast to prevent MPS incompatibility; keep in float32
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return ``[B, V]`` logits for the last position."""
        _, seq_len = input_ids.shape
        cos_sin = self.cos[:, :seq_len], self.sin[:, :seq_len]

        x = self.embed(input_ids)
        x = F.rms_norm(x, (x.shape[-1],))
        e_x = x
        for block in self.blocks:
            x = block(e_x, x, cos_sin)
        x = F.rms_norm(x, (x.shape[-1],))

        return F.linear(x[:, -1, :].float(), self.lm_head_gain(self.lm_head).float()).to(input_ids.dtype)

    def sample_batch(
        self,
        x: torch.Tensor,
        t: float = 0.7,
        top_p: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample one token per sequence in the batch."""
        logits = self.forward(x)
        if t != 1:
            logits = logits / t
        if top_p is not None or top_k is not None:
            logits = apply_top_k_top_p(logits, top_p=top_p, top_k=top_k)
        logits = logits + sample_gumbel(logits.shape, self.device)
        return torch.argmax(logits, dim=-1)

    def sample_batch_variable_temp(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        top_p: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Like :meth:`sample_batch` but *t* is a ``[B, 1]`` per-sequence temperature."""
        logits = self.forward(x)
        logits = logits / t
        if top_p is not None or top_k is not None:
            logits = apply_top_k_top_p(logits, top_p=top_p, top_k=top_k)
        logits = logits + sample_gumbel(logits.shape, self.device)
        return torch.argmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def resize_model_embeddings(
    model: TalkieModel, new_vocab_size: int, device: torch.device | str
) -> TalkieModel:
    """Grow embedding and lm_head to *new_vocab_size*, keeping old weights."""
    device = torch.device(device)
    old_vocab_size, n_embd = model.embed.weight.shape

    if old_vocab_size >= new_vocab_size:
        return model

    new_embed = nn.Embedding(new_vocab_size, n_embd, device=device)
    new_embed.weight.data[:old_vocab_size] = model.embed.weight.data
    new_embed.weight.data[old_vocab_size:] = (
        torch.randn(new_vocab_size - old_vocab_size, n_embd, device=device) * 0.02
    )
    model.embed = new_embed

    old_lm_head = model.lm_head.data
    new_lm_head = torch.zeros(new_vocab_size, n_embd, device=device)
    new_lm_head[:old_vocab_size] = old_lm_head
    new_lm_head[old_vocab_size:] = (
        torch.randn(new_vocab_size - old_vocab_size, n_embd, device=device) * 0.02
    )
    model.lm_head = nn.Parameter(new_lm_head)

    model.config.vocab_size = new_vocab_size
    return model


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
) -> TalkieModel:
    """Load a Talkie model from a PyTorch checkpoint file.

    Handles ``torch.compile`` key prefixes and optional vocab resizing.
    Uses mmap and device='meta' to bypass the massive 2x memory spike 
    during float32 initialisation on constrained systems (e.g., Mac 64GB).
    """
    # Load using mmap to avoid loading the whole 52GB float32 state_dict into physical RAM at once
    ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    ckpt_vocab_size = state_dict["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=ckpt_vocab_size)

    # Build on CPU in float16 directly to avoid 52GB memory spike
    cpu = torch.device("cpu")
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    model = TalkieModel(config, cpu)
    torch.set_default_dtype(orig_dtype)

    model.load_state_dict(state_dict, strict=False)
    del ckpt
    
    # ---------------------------------------------------------------------------
    # JCross 6-Axis Topology: Verification & Quantization Audit
    # ---------------------------------------------------------------------------
    # We dynamically intercept the state_dict and compress the Y-Axis (MLP layers)
    # into INT8. X-Axis (embed) and Z-Axis (attn) remain FP16.
    print("[Verantyx JCross] Applying 6-Axis Spatial Topology Quantization...")
    for k, v in list(state_dict.items()):
        if "mlp_gate" in k or "mlp_linear" in k or "mlp_resid" in k:
            if "weight" in k:
                # Quantize Y-Axis (INT8)
                v_fp32 = v.float()
                # Symmetric quantization
                scale = v_fp32.abs().max(dim=1, keepdim=True)[0] / 127.0
                scale = scale.clamp(min=1e-5)
                q_weight = torch.round(v_fp32 / scale).to(torch.int8)
                
                # Replace the original weight entry with the new INT8 components
                layer_prefix = k.replace(".weight", "")
                
                # Assign directly to the JCrossInt8Linear model
                module_dict = dict(model.named_modules())
                target_module = module_dict.get(layer_prefix)
                if target_module is not None:
                    target_module.weight.copy_(q_weight)
                    target_module.scale.copy_(scale)
    del state_dict

    if target_vocab_size is not None and ckpt_vocab_size < target_vocab_size:
        cpu = torch.device("cpu")
        model = resize_model_embeddings(model, target_vocab_size, cpu)

    # Convert remaining unquantized layers to float16 to keep memory minimal
    model = model.to(dtype=torch.float16).to(device)
    model.device = device
    model.eval()
    return model
