import mlx.core as mx
from generate_mlx import TalkieMLX

model = TalkieMLX("talkie-1930-13b-base")
print("Starting generation...")
gen = model.generate("Hello world, today is a good day to", max_tokens=20)
for chunk in gen:
    if isinstance(chunk, str):
        print(chunk, end="", flush=True)
    else:
        print("\n\nFinished:", chunk)
