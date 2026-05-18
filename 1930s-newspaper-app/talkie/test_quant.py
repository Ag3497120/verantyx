import mlx.nn as nn
m = nn.Linear(1, 1)
print(hasattr(m, 'named_modules'))
