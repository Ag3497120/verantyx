import mlx.core as mx
from generate_mlx import TalkieMLX
from talkie.model_mlx import rms_norm

model = TalkieMLX("talkie-1930-13b-base")
prompt_ids = model.tokenizer.encode("Hello")
x = mx.array([prompt_ids])
x = model.model.embed(x)
x = rms_norm(x, 1e-5)

for i, block in enumerate(model.model.blocks):
    if i == 38:
        print(f"Block {i}")
        attn_out, _ = block.attn(x, None, None)
        print("  After attn NaN?", mx.isnan(attn_out).any().item())
        x = x + block.attn_gain(attn_out)
        print("  After attn_gain NaN?", mx.isnan(x).any().item())
        mlp_out = block.mlp(rms_norm(x, 1e-5))
        print("  After mlp NaN?", mx.isnan(mlp_out).any().item())
        x = x + block.mlp_gain(mlp_out)
        print("  After mlp_gain NaN?", mx.isnan(x).any().item())
        x = block.resonance_gate(x, i)
        print("  After resonance NaN?", mx.isnan(x).any().item())
        break
    else:
        attn_out, _ = block.attn(x, None, None)
        x = x + block.attn_gain(attn_out)
        mlp_out = block.mlp(rms_norm(x, 1e-5))
        x = x + block.mlp_gain(mlp_out)
        x = block.resonance_gate(x, i)

