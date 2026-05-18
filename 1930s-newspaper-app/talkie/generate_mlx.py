import sys
import os
import torch
import mlx.core as mx
import mlx.nn as nn
from talkie.config import MODELS
from talkie.download import get_model_files
from talkie.tokenizer import build_tokenizer, IT_VOCAB_SIZE, BASE_VOCAB_SIZE
from talkie.model_mlx import TalkieModelMLX, GPTConfig

def load_checkpoint_mlx(checkpoint_path: str, config: GPTConfig) -> TalkieModelMLX:
    import gc
    mlx_safetensors_path = checkpoint_path + ".mlx.safetensors"
    
    if os.path.exists(mlx_safetensors_path):
        print(f"[MLX] Loading pre-converted MLX safetensors from {mlx_safetensors_path}...", flush=True)
        model = TalkieModelMLX(config)
        # Quantize structure FIRST so the shapes match the saved quantized weights
        nn.quantize(model, class_predicate=lambda p, m: isinstance(m, nn.Linear) and "embed" not in p and "lm_head" not in p, group_size=64, bits=8)
        model.load_weights(mlx_safetensors_path, strict=False)
        return model

    print("[MLX] Directly Synthesizing MLX Safetensors (Zero Model Allocation)...", flush=True)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt.get("model", ckpt)
    
    flat_mlx_dict = {}
    keys = list(state_dict.keys())
    for k in keys:
        v = state_dict.pop(k) # Free PyTorch tensor immediately
        new_k = k.replace("_orig_mod.", "")
        arr = mx.array(v.numpy()).astype(mx.float16)
        
        # We quantize all linear layer weights except embed and lm_head
        if "embed" not in new_k and "lm_head" not in new_k and "weight" in new_k and len(arr.shape) == 2:
            q_w, q_s, q_b = mx.quantize(arr, group_size=64, bits=8)
            mx.eval(q_w, q_s, q_b)
            base_k = new_k.replace(".weight", "")
            flat_mlx_dict[f"{base_k}.weight"] = q_w
            flat_mlx_dict[f"{base_k}.scales"] = q_s
            flat_mlx_dict[f"{base_k}.biases"] = q_b
        else:
            mx.eval(arr)
            flat_mlx_dict[new_k] = arr
            
        del v
        del arr

    del ckpt
    import gc
    gc.collect() # Force garbage collection
    
    print(f"[MLX] Saving ultra-compact MLX weights to {mlx_safetensors_path}...", flush=True)
    mx.save_safetensors(mlx_safetensors_path, flat_mlx_dict)
    del flat_mlx_dict
    
    print(f"[MLX] Conversion complete. Loading natively...", flush=True)
    model = TalkieModelMLX(config)
    nn.quantize(model, class_predicate=lambda p, m: isinstance(m, nn.Linear) and "embed" not in p and "lm_head" not in p, group_size=64, bits=8)
    model.load_weights(mlx_safetensors_path, strict=False)
    
    return model


class GenerationResult:
    def __init__(self, text, token_count, finish_reason):
        self.text = text
        self.token_count = token_count
        self.finish_reason = finish_reason

class TalkieMLX:
    def __init__(self, model_name: str):
        self.spec = MODELS[model_name]
        ckpt_path, vocab_path = get_model_files(model_name)
        
        self.tokenizer = build_tokenizer(vocab_path, style=self.spec.style)
        
        vocab_size = IT_VOCAB_SIZE if self.spec.style == "it" else BASE_VOCAB_SIZE
        config = GPTConfig(vocab_size=vocab_size)
        self.model = load_checkpoint_mlx(str(ckpt_path), config)
        
        # Prepare cache for generation
        self.kv_cache = None # Not implemented in naive port, using full sequence for demo
        
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256, top_p: float = None, top_k: int = None):
        prompt_ids = self.tokenizer.encode(prompt)
        x = mx.array([prompt_ids])
        
        def step(inputs, cache):
            logits, new_cache = self.model(inputs, cache)
            if temperature == 0.0:
                next_token = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                logits = logits / temperature
                next_token = mx.random.categorical(logits)
                if next_token.ndim == 1:
                    next_token = next_token[:, None]
            return next_token, new_cache
            
        tokens = []
        cache = None
        for _ in range(max_tokens):
            next_token, cache = step(x, cache)
            mx.eval(next_token)
            
            token_id = next_token.item()
            if token_id == self.tokenizer.encode_single_token("<|endoftext|>"):
                break
                
            token_str = self.tokenizer.decode([token_id])
            tokens.append(token_str)
            
            x = next_token
            yield token_str
            
        return GenerationResult(text="".join(tokens), token_count=len(tokens), finish_reason="stop")


if __name__ == "__main__":
    print("Initializing MLX Inference Engine...")
    
    # Run the prompt
    prompt = """NEW YORK — A strange, but not altogether
unheard-of, phenomenon occurred yesterday
afternoon. A great ball of fire, about the size
of a barrel, rose from the tracks of the New
York Central Railroad, and flew, at a speed of
500 miles an hour, over the city, passing over
the tallest buildings without touching them. The
fireball, after remaining in sight for"""

    try:
        engine = TalkieMLX("talkie-1930-13b-base")
        engine.generate(prompt)
    except Exception as e:
        print(f"Error during MLX initialization or generation: {e}")
