import mlx.core as mx
import mlx.nn as nn
from talkie.model_mlx import TalkieModelMLX, GPTConfig
import psutil
import os
import gc

process = psutil.Process(os.getpid())
print("Before init:", process.memory_info().rss / 1024**3, "GB")

config = GPTConfig(vocab_size=65536)
model = TalkieModelMLX(config)
nn.quantize(model, class_predicate=lambda p, m: isinstance(m, nn.Linear) and "embed" not in p and "lm_head" not in p, group_size=64, bits=8)
print("After quantize structure:", process.memory_info().rss / 1024**3, "GB")

checkpoint_path = "/Users/motonishikoudai/.cache/huggingface/hub/models--talkie-lm--talkie-1930-13b-base/snapshots/b7c97680791f7fca4262c3c80b36ff7d666faab0/final.ckpt.mlx.safetensors"
model.load_weights(checkpoint_path, strict=False)
mx.eval(model.parameters())
gc.collect()

print("After load_weights:", process.memory_info().rss / 1024**3, "GB")
