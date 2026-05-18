import mlx.core as mx
from generate_mlx import TalkieMLX
from talkie.model_mlx import rms_norm

model = TalkieMLX("talkie-1930-13b-base")
prompt_ids = model.tokenizer.encode("Hello")
x = mx.array([prompt_ids])
x = model.model.embed(x)
x = rms_norm(x, 1e-5)

for i, block in enumerate(model.model.blocks):
    if mx.isinf(x).any().item() or mx.isnan(x).any().item():
        print(f"x is inf/nan before attn at block {i}")
        break
        
    attn_out, _ = block.attn(x, None, None)
    if mx.isinf(attn_out).any().item() or mx.isnan(attn_out).any().item():
        print(f"attn_out is inf/nan at block {i}")
        break
        
    x = x + block.attn_gain(attn_out)
    if mx.isinf(x).any().item() or mx.isnan(x).any().item():
        print(f"x is inf/nan after attn_gain at block {i}")
        break
        
    norm_x = rms_norm(x, 1e-5)
    mlp_out = block.mlp(norm_x)
    if mx.isinf(mlp_out).any().item() or mx.isnan(mlp_out).any().item():
        print(f"mlp_out is inf/nan at block {i}")
        break
        
    x = x + block.mlp_gain(mlp_out)
    if mx.isinf(x).any().item() or mx.isnan(x).any().item():
        print(f"x is inf/nan after mlp_gain at block {i}")
        break
        
    x = block.resonance_gate(x, i)
