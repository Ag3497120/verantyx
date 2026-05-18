import mlx.core as mx
import mlx.nn as nn

class JCrossMoEMLX(nn.Module):
    """
    MLX Ouroboros MoE (Mixture of Experts) Layer
    """
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.active_experts = {}

    def _load_expert(self, expert_idx: int):
        if expert_idx not in self.active_experts:
            # Simulate MLX SSD loaded expert
            self.active_experts[expert_idx] = {
                "w1": mx.random.normal((self.hidden_dim, self.dim), dtype=mx.float16),
                "w2": mx.random.normal((self.dim, self.hidden_dim), dtype=mx.float16)
            }
        return self.active_experts[expert_idx]

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)
        
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=-1)
        
        # MLX argpartition for top_k
        inds = mx.argpartition(-routing_weights, self.top_k, axis=-1)[:, :self.top_k]
        
        # Extract weights and selected experts
        selected_weights = mx.take_along_axis(routing_weights, inds, axis=-1)
        selected_weights = selected_weights / mx.sum(selected_weights, axis=-1, keepdims=True)
        
        final_output = mx.zeros_like(x_flat)
        
        for i in range(x_flat.shape[0]):
            token_x = x_flat[i]
            token_out = mx.zeros_like(token_x)
            
            for j in range(self.top_k):
                expert_idx = inds[i, j].item()
                weight = selected_weights[i, j]
                
                expert = self._load_expert(expert_idx)
                
                # FFN Execution
                e_w1 = expert["w1"].astype(token_x.dtype)
                e_w2 = expert["w2"].astype(token_x.dtype)
                
                h = nn.silu(token_x @ e_w1.T)
                o = h @ e_w2.T
                
                token_out = token_out + weight * o
                
            final_output[i] = token_out
            
        return final_output.reshape(batch_size, seq_len, dim)


class OuroborosResonanceGateMLX(nn.Module):
    def __init__(self, dim: int, target_layer: int = 15, scale: float = 0.05):
        super().__init__()
        self.dim = dim
        self.target_layer = target_layer
        self.scale = scale
        self.is_active = True
        
        self.ssd_injection = nn.Linear(dim, dim, bias=False)
        # Cast to float16 to match Ouroboros protocol
        self.ssd_injection.weight = self.ssd_injection.weight.astype(mx.float16)
        
    def __call__(self, x: mx.array, current_layer: int) -> mx.array:
        if not self.is_active or current_layer != self.target_layer:
            return x
            
        x_fp16 = x.astype(mx.float16)
        external_signal = self.ssd_injection(x_fp16)
        
        # MLX handles NaNs naturally in many ops, but we can clamp or zero if needed
        external_signal = mx.where(mx.isnan(external_signal), mx.zeros_like(external_signal), external_signal)
        
        fused_x = x + (external_signal * self.scale).astype(x.dtype)
        return fused_x

if __name__ == "__main__":
    print("Ouroboros MLX Architecture Initialized.")
    gate = OuroborosResonanceGateMLX(512)
    x = mx.random.normal((1, 10, 512))
    out = gate(x, 15)
    print(f"Output shape: {out.shape}")
