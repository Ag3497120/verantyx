import mlx.core as mx
import mlx.nn as nn
from talkie.model_mlx import TalkieModelMLX, GPTConfig
import time
import os

config = GPTConfig(vocab_size=65536)
model = TalkieModelMLX(config)
nn.quantize(model, class_predicate=lambda p, m: isinstance(m, nn.Linear) and "embed" not in p and "lm_head" not in p, group_size=64, bits=8)
checkpoint_path = "/Users/motonishikoudai/.cache/huggingface/hub/models--talkie-lm--talkie-1930-13b-base/snapshots/b7c97680791f7fca4262c3c80b36ff7d666faab0/final.ckpt.mlx.safetensors"
model.load_weights(checkpoint_path, strict=False)

x = mx.array([[1] * 60])

def step(inputs):
    logits = model(inputs)
    return mx.argmax(logits, axis=-1, keepdims=True)

print("Starting generation...")
start = time.time()
for i in range(5):
    t0 = time.time()
    t = step(x)
    mx.eval(t)
    x = mx.concatenate([x, t], axis=1)
    print(f"Step {i} took {time.time() - t0:.2f}s")
print(f"Total: {time.time() - start:.2f}s")
