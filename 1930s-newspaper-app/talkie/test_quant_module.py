import mlx.nn as nn
import mlx.core as mx
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
m = Model()
for name, child in m.named_modules():
    if isinstance(child, nn.Linear):
        nn.quantize(child)
print(type(m.lin))
nn.quantize(m)
print(type(m.lin))
