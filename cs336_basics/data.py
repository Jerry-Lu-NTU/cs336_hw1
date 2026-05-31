import torch
import numpy as np
import numpy.typing as npt

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1. 计算允许的最大起始索引，防止切片时越界
    # 需要 context_length 个 token 作为输入，再加 1 个 token 用于最后的标签，由于randint生成的索引是左闭右开区间
    # 所以最大起始索引应该是 len(dataset) - context_length - 1 + 1 = len(dataset) - context_length
    max_idx = len(dataset) - context_length 
    
    # 2. 随机生成 batch_size 个起始索引
    # 使用 torch.randint 比纯 Python 的 random 更快
    # 生成的索引范围是 [0, max_idx)，确保切片时不会越界
    ix = torch.randint(0, max_idx, (batch_size,))
    
    # 3. 根据索引进行切片，获取输入 x 和标签 y
    # x 的切片范围是 [i, i + context_length]
    # y 的切片范围是 [i + 1, i + context_length + 1] 
    x = torch.stack([torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64)) for i in ix])
    
    # 4. 将张量移动到目标设备 (CPU 或 GPU)
    # PyTorch 中 embedding 层和交叉熵损失函数默认需要 torch.long (即 int64) 类型
    x = x.to(device)
    y = y.to(device)
    
    return x, y