import mlx.core as mx
import mlx.nn as nn
from talkie.model_mlx import TalkieModelMLX, GPTConfig
import psutil
import os
import gc

process = psutil.Process(os.getpid())

config = GPTConfig(vocab_size=65536)
model = TalkieModelMLX(config)
nn.quantize(model, class_predicate=lambda p, m: isinstance(m, nn.Linear) and "embed" not in p and "lm_head" not in p, group_size=64, bits=8)

checkpoint_path = "/Users/motonishikoudai/.cache/huggingface/hub/models--talkie-lm--talkie-1930-13b-base/snapshots/b7c97680791f7fca4262c3c80b36ff7d666faab0/final.ckpt.mlx.safetensors"
model.load_weights(checkpoint_path, strict=False)

print("Before eval:", process.memory_info().rss / 1024**3, "GB")

inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
logits = model(inputs)
token = mx.argmax(logits, axis=-1)
mx.eval(token)

print("After forward:", process.memory_info().rss / 1024**3, "GB")
