import numpy
import torch
import torch.nn.functional as F
from einops import rearrange

# 中文导读：
# 这个文件验证 Transformer LM 的各个模型组件。
# 所有被测接口都来自 tests/adapters.py；你的实现可以放在 cs336_basics/ 中，
# 然后在 adapters.py 里把这些 run_* 函数转接到你的实现。
# 这些测试大多使用 tests/_snapshots/*.npz 中的参考输出做数值比较，
# 所以 shape、权重方向、归一化顺序、mask 语义和 RoPE 位置处理都要严格一致。

from .adapters import (
    run_embedding,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_scaled_dot_product_attention,
    run_silu,
    run_swiglu,
    run_transformer_block,
    run_transformer_lm,
)


# 需要实现接口：run_linear(d_in, d_out, weights, in_features)。
# 测试目标：用给定的无 bias 线性层权重，把最后一维 d_model 映射到 d_ff。
def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(output)


# 需要实现接口：run_embedding(vocab_size, d_model, weights, token_ids)。
# 测试目标：按 token id 从 embedding 矩阵取向量，输出 shape 是 token_ids 后面追加 d_model。
def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(output)


# 需要实现接口：run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features)。
# 测试目标：验证 FFN 中的 SwiGLU 子层，输入输出最后一维都应是 d_model。
def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


# 需要实现接口：run_scaled_dot_product_attention(Q, K, V, mask)。
# 测试目标：验证普通 3D attention 输入，mask 的最后两维对应 queries x keys。
def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-5,
    )


# 同样测试 run_scaled_dot_product_attention，但输入被整理成 4D：
# batch x head x seq x d。你的 attention 实现需要支持任意 leading dimensions。
def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-5,
    )


# 需要实现接口：run_multihead_self_attention(...)。
# 测试目标：把 Q/K/V projection、分头 attention、输出 projection 串起来；这个版本不使用 RoPE。
def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


# 需要实现接口：run_multihead_self_attention_with_rope(...)。
# 测试目标：在 MHA 中只对 Q/K 应用 RoPE，并正确处理 token_positions。
def test_multihead_self_attention_with_rope(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


# 需要实现接口：run_transformer_lm(...)。
# 测试目标：完整 Transformer LM 前向传播，输出 logits shape 为 batch x seq x vocab_size。
def test_transformer_lm(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


# 同样测试 run_transformer_lm。
# 测试目标：输入序列可以短于 context_length，不能假设总是固定最大长度。
def test_transformer_lm_truncated_input(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


# 需要实现接口：run_transformer_block(...)。
# 测试目标：单个 pre-norm Transformer block，包含 RMSNorm、带 RoPE 的 attention、SwiGLU FFN 和 residual。
def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):
    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-4,
    )


# 需要实现接口：run_rmsnorm(d_model, eps, weights, in_features)。
# 测试目标：只在最后一维做 RMSNorm，并乘上传入的 affine weight。
def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)

    numpy_snapshot.assert_match(actual_output, atol=1e-4)


# 需要实现接口：run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions)。
# 测试目标：RoPE 输出 shape 不变，位置由 token_positions 指定。
def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    output = run_rope(
        d_model, theta=theta, max_seq_len=n_queries, in_query_or_key=in_embeddings, token_positions=pos_ids
    )
    numpy_snapshot.assert_match(output, atol=1e-5)


# 需要实现接口：run_silu(in_features)。
# 测试目标：逐元素 SiLU 激活要和 PyTorch F.silu 数值一致。
def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-5)
