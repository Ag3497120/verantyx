import mlx.core as mx
from generate_mlx import TalkieMLX
from talkie.model_mlx import rms_norm

model = TalkieMLX("talkie-1930-13b-base")
prompt_ids = model.tokenizer.encode("Hello")
x = mx.array([prompt_ids])
x = model.model.embed(x)
x = rms_norm(x, 1e-5)

for i, block in enumerate(model.model.blocks):
    if i == 38:
        attn = block.attn
        q = attn.attn_query(x)
        print("q inf/nan?", mx.isinf(q).any().item(), mx.isnan(q).any().item())
        k = attn.attn_key(x)
        print("k inf/nan?", mx.isinf(k).any().item(), mx.isnan(k).any().item())
        v = attn.attn_value(x)
        print("v inf/nan?", mx.isinf(v).any().item(), mx.isnan(v).any().item())
        
        bsz, seq_len, _ = x.shape
        q = q.reshape(bsz, seq_len, attn.n_head, attn.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, attn.n_head, attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, attn.n_head, attn.head_dim).transpose(0, 2, 1, 3)
        
        q = attn.rope(q)
        k = attn.rope(k)
        
        q = rms_norm(q, eps=1e-5)
        k = rms_norm(k, eps=1e-5)
        q = attn.head_gain(q)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) * (attn.head_dim ** -0.5)
        print("scores inf/nan?", mx.isinf(scores).any().item(), mx.isnan(scores).any().item())
        
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        y = (scores @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        print("y inf/nan?", mx.isinf(y).any().item(), mx.isnan(y).any().item())
        
        resid = attn.attn_resid(y)
        print("resid inf/nan?", mx.isinf(resid).any().item(), mx.isnan(resid).any().item())
        break
    else:
        attn_out, _ = block.attn(x, None, None)
        x = x + block.attn_gain(attn_out)
        norm_x = rms_norm(x, 1e-5)
        mlp_out = block.mlp(norm_x)
        x = x + block.mlp_gain(mlp_out)
        x = block.resonance_gate(x, i)
