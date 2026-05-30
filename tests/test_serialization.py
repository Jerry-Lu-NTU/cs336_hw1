import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import get_adamw_cls, run_load_checkpoint, run_save_checkpoint


# 中文导读：
# 这个文件验证 checkpoint 保存和恢复。
# 你需要实现 run_save_checkpoint、run_load_checkpoint，并且 get_adamw_cls()
# 返回的优化器要能正确提供/恢复 state_dict。


# 测试用的小模型。你的 checkpoint 代码不需要依赖这个类，
# 只要能处理任意 torch.nn.Module 的 state_dict 即可。
class _TestNet(nn.Module):
    def __init__(self, d_input: int = 100, d_output: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, d_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 测试辅助函数：逐项比较两个 optimizer.state_dict()。
# AdamW 的动量、方差、step 等状态都应在加载后保持一致。
def are_optimizers_equal(optimizer1_state_dict, optimizer2_state_dict, atol=1e-8, rtol=1e-5):
    # Check if the keys of the main dictionaries are equal (e.g., 'state', 'param_groups')
    if set(optimizer1_state_dict.keys()) != set(optimizer2_state_dict.keys()):
        return False

    # Check parameter groups are identical
    if optimizer1_state_dict["param_groups"] != optimizer2_state_dict["param_groups"]:
        return False

    # Check states
    state1 = optimizer1_state_dict["state"]
    state2 = optimizer2_state_dict["state"]
    if set(state1.keys()) != set(state2.keys()):
        return False

    for key in state1:
        # Assuming state contents are also dictionaries
        if set(state1[key].keys()) != set(state2[key].keys()):
            return False

        for sub_key in state1[key]:
            item1 = state1[key][sub_key]
            item2 = state2[key][sub_key]

            # If both items are tensors, use torch.allclose
            if torch.is_tensor(item1) and torch.is_tensor(item2):
                if not torch.allclose(item1, item2, atol=atol, rtol=rtol):
                    return False
            # For non-tensor items, check for direct equality
            elif item1 != item2:
                return False
    return True


# 需要实现接口：
# - run_save_checkpoint(model, optimizer, iteration, out)
# - run_load_checkpoint(src, model, optimizer) -> iteration
# 测试目标：训练若干步后保存，再加载到新模型/新优化器，模型参数、优化器状态和迭代数都完全恢复。
def test_checkpointing(tmp_path):
    torch.manual_seed(42)
    d_input = 100
    d_output = 10
    num_iters = 10

    model = _TestNet(d_input=d_input, d_output=d_output)
    optimizer = get_adamw_cls()(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    it = 0
    for _ in range(num_iters):
        optimizer.zero_grad()
        x = torch.rand(d_input)
        y = torch.rand(d_output)
        y_hat = model(x)
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        optimizer.step()
        it += 1

    serialization_path = tmp_path / "checkpoint.pt"
    # Save the model
    run_save_checkpoint(
        model,
        optimizer,
        iteration=it,
        out=serialization_path,
    )

    # Load the model back again
    new_model = _TestNet(d_input=d_input, d_output=d_output)
    new_optimizer = get_adamw_cls()(
        new_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loaded_iterations = run_load_checkpoint(src=serialization_path, model=new_model, optimizer=new_optimizer)
    assert it == loaded_iterations

    # Compare the loaded model state with the original model state
    original_model_state = model.state_dict()
    original_optimizer_state = optimizer.state_dict()
    new_model_state = new_model.state_dict()
    new_optimizer_state = new_optimizer.state_dict()

    # Check that state dict keys match
    assert set(original_model_state.keys()) == set(new_model_state.keys())
    assert set(original_optimizer_state.keys()) == set(new_optimizer_state.keys())

    # compare the model state dicts
    for key in original_model_state.keys():
        numpy.testing.assert_allclose(
            original_model_state[key].detach().numpy(),
            new_model_state[key].detach().numpy(),
        )
    # compare the optimizer state dicts
    assert are_optimizers_equal(original_optimizer_state, new_optimizer_state)
