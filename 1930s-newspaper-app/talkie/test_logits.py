import mlx.core as mx
from generate_mlx import TalkieMLX

model = TalkieMLX("talkie-1930-13b-base")
print("Starting generation...")

prompt_ids = model.tokenizer.encode("Hello world, today is a good day to")
x = mx.array([prompt_ids])
logits, _ = model.model(x)

print("Logits shape:", logits.shape)
print("Logits mean:", mx.mean(logits))
print("Logits max:", mx.max(logits))
print("Logits min:", mx.min(logits))
print("Argmax:", mx.argmax(logits, axis=-1))

