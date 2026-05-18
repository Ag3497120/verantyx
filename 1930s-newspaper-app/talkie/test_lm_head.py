import mlx.core as mx
from generate_mlx import TalkieMLX

model = TalkieMLX("talkie-1930-13b-base")
print("lm_head sum:", mx.sum(model.model.lm_head).item())
