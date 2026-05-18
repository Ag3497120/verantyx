import mlx.core as mx
import mlx.nn as nn
from generate_mlx import TalkieMLX

model = TalkieMLX("talkie-1930-13b-base")
checkpoint_path = "/Users/motonishikoudai/.cache/huggingface/hub/models--talkie-lm--talkie-1930-13b-base/snapshots/b7c97680791f7fca4262c3c80b36ff7d666faab0/final.ckpt.mlx.safetensors"

# Test missing
try:
    model.model.load_weights(checkpoint_path, strict=True)
except Exception as e:
    print(str(e)[:500])
