from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Args:
            params: 可训练参数
            lr: 学习率 α
            betas: (β1, β2)
            eps: ε，防止除零
            weight_decay: λ，weight decay 系数
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                # 第6行：获取梯度
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                m = state['m']
                v = state['v']
                state['step'] += 1
                t = state['step']

                # 第7行：计算偏差修正后的学习率 α_t
                # α_t = α * sqrt(1 - β2^t) / (1 - β1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * (bias_correction2 ** 0.5) / bias_correction1

                # 第8行：应用 weight decay（AdamW 的核心：decoupled weight decay）
                # 直接在参数上减，不经过动量/自适应调整
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # 第9行：更新一阶矩 m
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 第10行：更新二阶矩 v
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 第11行：应用 moment-adjusted weight updates
                # θ ← θ - α_t * m / sqrt(v + ε)
                denom = (v.sqrt() + eps)
                step_size = alpha_t / denom
                p.data.add_(m * step_size, alpha=-1)

        return loss


def gradient_clipping(parameters: Iterable[Parameter], max_l2_norm: float) -> None:
    pass


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    pass
