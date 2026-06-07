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
    
class Softmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 为了数值稳定性，先减去 max(x) 再计算 exp : 等价于 softmax(x)公式中分子分母同时除以exp(max(x))，能避免 x 中的值过大导致 exp(x) 溢出。
        x_max = x.max(dim=self.dim, keepdim=True).values #沿指定 dim 找最大值, 取max会变成张量，keepdim=True 保留该维度，方便广播
        x_exp = torch.exp(x - x_max)
        x_exp_sum = x_exp.sum(dim=self.dim, keepdim=True)
        return x_exp / x_exp_sum


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # 预计算旋转位置编码矩阵，形状为 (max_seq_len, d_k)
        position_ids = torch.arange(max_seq_len, device=self.device).float().unsqueeze(1)  # (max_seq_len, 1),转成 float，避免整数乘法截断
        # (1, d_k//2)，索引是从0到d_k//2-1的维度索引，论文里是1到d_k//2，但这里从0开始，所以是 dim_ids / (d_k//2) * 2
        dim_ids = torch.arange(0, d_k, 2, device=self.device).float().unsqueeze(0)  # (1, d_k//2)  [0, 2, 4, ...]
        # theta_{i,k} = i / theta^(2k/d) = i * theta^(-2k/d)
        freqs = position_ids * (theta ** (-dim_ids / d_k))  # (max_seq_len, d_k//2)

        self.register_buffer("sin_cache", torch.sin(freqs))  # (max_seq_len, d_k//2)
        self.register_buffer("cos_cache", torch.cos(freqs))  # (max_seq_len, d_k//2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions 形状 (..., seq_len)，值域 [0, max_seq_len)
        sin_pos = self.sin_cache[token_positions]  # (..., seq_len, d_k//2)
        cos_pos = self.cos_cache[token_positions]  # (..., seq_len, d_k//2)
        x1 = x[..., 0::2]  # 偶数索引: x0, x2, x4, x6...  (..., d//2)
        x2 = x[..., 1::2]  # 奇数索引: x1, x3, x5, x7...  (..., d//2)
        x_rotated_1 = x1 * cos_pos - x2 * sin_pos
        x_rotated_2 = x1 * sin_pos + x2 * cos_pos
        x_out = torch.stack([x_rotated_1, x_rotated_2], dim=-1).flatten(-2)
        # 或：
        # x_out = torch.zeros_like(x)
        # x_out[..., 0::2] = x_rotated_1
        # x_out[..., 1::2] = x_rotated_2
        
        return x_out  # 根据 token_positions 选择对应位置的 RoPE 编码

class Scaled_dot_product_attention(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d_k = Q.size(-1)
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)  # 计算缩放点积注意力的分数
        scores = scores.masked_fill(mask == 0, float("-inf"))  # 将 mask 中为 0 的位置的分数设为 -inf，使其在 softmax 后权重为 0
        attn_weights = torch.softmax(scores, dim=self.dim)  # 对分数进行 softmax，得到注意力权重
        output = attn_weights @ V  # 用注意力权重加权 V，得到输出
        return output