import mlx.nn as nn
import mlx.core as mx
m = nn.Linear(128, 128, bias=False)
nn.quantize(m, group_size=64, bits=8)
arr = mx.ones((128, 128))
m.update({"weight": arr})
print(type(m))
print(m.__dict__)
