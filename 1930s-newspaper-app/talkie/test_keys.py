import mlx.nn as nn
import mlx.core as mx
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(128, 128, bias=True)
m = Model()
nn.quantize(m, group_size=64, bits=8)
m.save_weights("test_q.safetensors")
print(list(mx.load("test_q.safetensors").keys()))
