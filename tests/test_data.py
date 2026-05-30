import math
from collections import Counter

import numpy as np
import pytest

from .adapters import run_get_batch


# 中文导读：
# 这个文件只测试数据 batch 采样。
# 需要实现接口：run_get_batch(dataset, batch_size, context_length, device)。
# 返回值应是两个 LongTensor：x 是连续 token 片段，y 是对应的下一 token 标签。


# 测试目标：
# 1. x/y shape 都是 batch_size x context_length。
# 2. 每个 y 都等于对应 x 向右移动一个 token。
# 3. 随机起点覆盖完整合法范围，不越界，也不要固定采样同一个位置。
# 4. device 参数确实被使用，传入非法 CUDA 设备时应暴露错误。
def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)
