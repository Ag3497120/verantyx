import mlx.nn as nn
import mlx.core as mx
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
m = Model()
for x in m.children():
    print(type(x))
