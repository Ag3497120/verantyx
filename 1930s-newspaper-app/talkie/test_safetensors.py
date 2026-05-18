import mlx.core as mx

checkpoint_path = "/Users/motonishikoudai/.cache/huggingface/hub/models--talkie-lm--talkie-1930-13b-base/snapshots/b7c97680791f7fca4262c3c80b36ff7d666faab0/final.ckpt.mlx.safetensors"

w = mx.load(checkpoint_path)
print("Keys count:", len(w))
if "embed.weight" in w:
    print("embed.weight shape:", w["embed.weight"].shape, "dtype:", w["embed.weight"].dtype)
else:
    print("embed.weight MISSING")

if "blocks.39.mlp.mlp_gate.weight" in w:
    print("blocks.39.mlp.mlp_gate.weight shape:", w["blocks.39.mlp.mlp_gate.weight"].shape, "dtype:", w["blocks.39.mlp.mlp_gate.weight"].dtype)
    print("blocks.39.mlp.mlp_gate.scales shape:", w["blocks.39.mlp.mlp_gate.scales"].shape, "dtype:", w["blocks.39.mlp.mlp_gate.scales"].dtype)
else:
    print("blocks.39.mlp.mlp_gate.weight MISSING")

if "blocks.0.attn_gain.a_g" in w:
    print("blocks.0.attn_gain.a_g value:", w["blocks.0.attn_gain.a_g"])
else:
    print("blocks.0.attn_gain.a_g MISSING")

