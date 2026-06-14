import torch
from cs336_basics.nn import (
    Embedding,
    Linear,
    RMSNorm,
    Swiglu,
    RoPE,
    Scaled_dot_product_attention,
)


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, in_features: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = in_features.shape
        Q = self.q_proj.forward(in_features)
        K = self.k_proj.forward(in_features)
        V = self.v_proj.forward(in_features)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool() #左下角和对角线为True（允许看过去和现在），右上角为False（不许看未来）
        attn_output = Scaled_dot_product_attention().forward(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) # 也可以直接用.reshape
        output = self.output_proj.forward(attn_output)
        return output      


class MultiheadSelfAttentionWithRoPE(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = RoPE(theta, self.d_k, max_seq_len)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = in_features.shape
        Q = self.q_proj.forward(in_features)
        K = self.k_proj.forward(in_features)
        V = self.v_proj.forward(in_features)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=in_features.device)
        Q = self.rope.forward(Q, token_positions)
        K = self.rope.forward(K, token_positions)
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool() #左下角和对角线为True（允许看过去和现在），右上角为False（不许看未来）
        attn_output = Scaled_dot_product_attention().forward(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) # 也可以直接用.reshape
        output = self.output_proj.forward(attn_output)
        return output   


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = Swiglu(d_model, d_ff)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        rms_output = self.ln1.forward(in_features)
        attn_output = self.attn.forward(rms_output)
        attn_residual = in_features + attn_output
        ffn_input = self.ln2.forward(attn_residual)
        ffn_output = self.ffn.forward(ffn_input)
        output = attn_residual + ffn_output
        return output


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings.forward(in_indices)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        logits = self.lm_head.forward(x)
        return logits