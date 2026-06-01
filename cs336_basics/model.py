import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings  # 词表大小 vocab_size
        self.embedding_dim = embedding_dim  # embedding 维度 d_model(e.g. 768)
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype)
        )
        std = 1  
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3, b=3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(
            torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype)
        )
        std = 2 / (self.in_features + self.out_features) ** 0.5  # Xavier 初始化标准差
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return in_features @ self.W.T

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(
            torch.ones(self.d_model, device=self.device, dtype=self.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 输入 x 的形状是 (batch_size, in_features, d_model)，在最后一维上计算 RMS, keepdim=True 保持维度不变(batch,in,1)以便后续广播
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        x_normed = x / rms 
        result = x_normed * self.W
        return result.to(in_dtype)

class Swiglu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 如果按作业要求计算 d_ff（此时应该没用d_ff输入）
        # raw_d_ff = int((8 / 3) * d_model)
        # self.d_ff = 64 * math.ceil(raw_d_ff / 64)
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)  # 门控
        self.w3 = torch.nn.Linear(d_model, d_ff, bias=False)  # 值
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)  # 输出投影
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x) # 门控线性变换 weight(x)的底层是 in_features @ weight.T + bias，但这里 bias=False，所以没有偏置项
        value = self.w3(x)
        hidden = gate * torch.sigmoid(gate) * value  # SwiGLU 激活
        return self.w2(hidden)


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # todo
        # 预计算旋转位置编码矩阵，形状为 (max_seq_len, d_k)
        position_ids = torch.arange(max_seq_len, device=self.device).unsqueeze(1)  # (max_seq_len, 1)
        dim_ids = torch.arange(d_k // 2, device=self.device).unsqueeze(0)  # (1, d_k//2)
        freqs = theta ** (-2 * dim_ids / d_k)  # (1, d_k//2)
        self.register_buffer("sin_cache", torch.sin(position_ids * freqs))  # (max_seq_len, d_k//2)
        self.register_buffer("cos_cache", torch.cos(position_ids * freqs))  # (max_seq_len, d_k//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # todo
        seq_len = x.size(1)
        sin_pos = self.sin_cache[:seq_len]  # (seq_len, d_model//2)
        cos_pos = self.cos_cache[:seq_len]  # (seq_len, d_model//2)
        x1, x2 = x.chunk(2, dim=-1)  # 将最后一维分成两半
        x_rotated_1 = x1 * cos_pos - x2 * sin_pos
        x_rotated_2 = x1 * sin_pos + x2 * cos_pos
        return torch.cat([x_rotated_1, x_rotated_2], dim=-1)

# def run_embedding(vocab_size: int, d_model: int, weights: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
#     return weights[token_ids]

# def run_linear(d_in: int, d_out: int, weights: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
#     """
#     用给定的线性层权重和输入特征，计算线性变换的输出。
    
#     输入参数：
#     - d_in: 输入特征的维度，即权重矩阵的第二维度。
#     - d_out: 输出特征的维度，即权重矩阵的第一维度。
#     - weights: 形状为 (d_out, d_in) 的线性层权重矩阵。
#     - in_features: 形状为 (..., d_in) 的输入特征张量，最后一维是 d_in。
    
#     输出：
#     - 形状为 (..., d_out) 的输出张量，最后一维是 d_out。
    
#     注意事项：
#     - 线性变换的计算方式是 out = in_features @ weights.T，其中 @ 表示矩阵乘法，weights.T 是权重矩阵的转置。
#     - 输出张量的数据类型应该与权重矩阵相同，通常是 torch.float32。
#     """
#     return in_features @ weights.T
    