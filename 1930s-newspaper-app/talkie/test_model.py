import torch
from talkie.model import TalkieModel, GPTConfig

config = GPTConfig(n_layer=16) # Smaller test model
device = "cpu"
model = TalkieModel(config, device=device)

print("TalkieModel (with Ouroboros Integration) initialized successfully!")

x = torch.randint(0, config.vocab_size, (1, 10))
out = model(x)
print(f"Forward pass successful! Output shape: {out.shape}. NaNs: {torch.isnan(out).any().item()}")
