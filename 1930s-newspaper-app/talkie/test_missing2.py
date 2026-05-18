import mlx.core as mx
from generate_mlx import TalkieMLX

model = TalkieMLX("talkie-1930-13b-base")
checkpoint_path = "/Users/motonishikoudai/.cache/huggingface/hub/models--talkie-lm--talkie-1930-13b-base/snapshots/b7c97680791f7fca4262c3c80b36ff7d666faab0/final.ckpt.mlx.safetensors"

w = mx.load(checkpoint_path)
print("lm_head in w:", "lm_head" in w)
w.pop("lm_head", None)

try:
    model.model.load_weights(list(w.items()), strict=True)
except Exception as e:
    print(str(e)[:500])
