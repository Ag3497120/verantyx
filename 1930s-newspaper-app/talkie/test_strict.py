import mlx.nn as nn
import mlx.core as mx
m = nn.Linear(10, 10)
mx.save_safetensors("empty.safetensors", {})
m.load_weights("empty.safetensors", strict=False)
print("Loaded without strict")
