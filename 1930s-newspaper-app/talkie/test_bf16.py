import mlx.core as mx
from generate_mlx import TalkieMLX

model = TalkieMLX("talkie-1930-13b-base")
prompt_ids = model.tokenizer.encode("Hello")
x = mx.array([prompt_ids])

x_bf16 = x
# Just run through model and see if logits are NaN
logits, _ = model.model(x_bf16.astype(mx.bfloat16)) # Embed layer requires int, so wait, embed should be casted after?
# Actually input_ids is int, so pass input_ids as int. model.__call__ does x = embed(input_ids)
# Let's modify model.__call__ to cast x to bfloat16.
