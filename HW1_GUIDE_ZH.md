# CS336 HW1 Basics 完成指南

这份文档是学习路线和工程组织说明，不包含作业解法代码。仓库里的 `AGENTS.md` 要求 AI 只做助教式指导；核心实现需要你自己写。建议把本文件当作推进清单：先读 handout，再按模块补实现，最后用测试和实验验证。

## 1. 作业范围

作业 PDF 是 [cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)。从目录和测试接口看，HW1 主要包含这些部分：

- BPE tokenizer：Unicode/bytes、pre-tokenization、BPE 训练、特殊 token、encode/decode、流式 encode。
- Transformer LM：Linear、Embedding、RMSNorm、SwiGLU、RoPE、scaled dot-product attention、causal MHA、pre-norm block、完整 LM。
- 训练工具：softmax、cross entropy、gradient clipping、AdamW、cosine LR schedule、batch sampling、checkpoint。
- 实验：TinyStories、OpenWebText、消融、生成文本、leaderboard 修改。

课件里最相关的入口：

- `lectures/lecture_01.py`：课程定位、tokenization。
- `lectures/lecture_02.py`：tensor、资源估算、优化器、训练循环。
- `lectures/lecture_03.pdf` 到 `lecture_05.pdf`：建议配合 handout 的 Transformer 架构、训练与实验章节阅读。

## 2. 仓库结构

当前作业目录是 `cs336_hw1`：

- `README.md`：环境、测试、数据下载说明。
- `cs336_assignment1_basics.pdf`：完整 handout。
- `tests/adapters.py`：测试入口。你需要把自己的实现接到这些 adapter 函数上。
- `tests/test_*.py`：测试规格。不要改测试。
- `cs336_basics/`：你的实现包。目前几乎是空的，适合按模块拆文件。
- `cs336_basics/pretokenization_example.py`：handout 提供的 chunk boundary 思路示例，可用于理解并行 pre-tokenization。

如果运行 `uv run ...` 时遇到 home cache 只读问题，可以把 uv cache 指到仓库或 `/tmp`，例如先设置 `UV_CACHE_DIR=.uv-cache`。这是环境问题，不是作业逻辑问题。

## 3. 推荐架构

你可以按责任边界拆模块，而不是把所有东西堆进 `tests/adapters.py`。一种清晰拆法如下：

- `cs336_basics/nn.py`：基础层和张量函数。
  负责 Linear、Embedding、RMSNorm、SiLU/SwiGLU、softmax、cross entropy、attention、RoPE。

- `cs336_basics/model.py`：模型组合。
  负责 causal MHA、TransformerBlock、TransformerLM。这里尽量只组合基础层，不重复实现底层数学。

- `cs336_basics/optim.py`：优化相关。
  负责 AdamW、learning-rate schedule、gradient clipping。

- `cs336_basics/data.py`：数据采样。
  负责从一维 token 数组中采样 `(x, y)` batch，并保证 device、shape、offset 关系正确。

- `cs336_basics/tokenizer.py`：BPE 训练和 tokenizer。
  负责 `train_bpe`、Tokenizer 的 `encode`、`decode`、`encode_iterable`，以及特殊 token 处理。

- `cs336_basics/checkpoint.py`：序列化。
  负责保存/加载 model state、optimizer state 和 iteration。

- `cs336_basics/train.py`：训练脚本。
  负责参数读取、数据加载、训练循环、日志、checkpoint、验证和生成。单元测试不一定直接覆盖它，但实验需要。

`tests/adapters.py` 建议只做薄薄的一层连接：接收测试传入的参数，实例化或调用你在 `cs336_basics/` 中的实现，然后返回测试期望的结果。这样调试时可以直接定位到真实模块。

## 4. Adapter 对应关系

| Adapter | 建议归属 | 重点不变量 |
| --- | --- | --- |
| `run_linear` / `run_embedding` | `nn.py` | 输出 shape、权重 shape、无 bias 约定 |
| `run_rmsnorm` / `run_silu` / `run_swiglu` | `nn.py` | 最后一维归一化、数值稳定、输入输出 shape 相同 |
| `run_scaled_dot_product_attention` | `nn.py` | 任意 leading dims、mask 广播、softmax 维度 |
| `run_rope` | `nn.py` | token position shape、偶数维旋转、cache 与 dtype/device 一致 |
| `run_multihead_self_attention*` | `model.py` | head reshape、causal mask、RoPE 只作用于 Q/K |
| `run_transformer_block` | `model.py` | pre-norm、residual 顺序、state dict key 映射 |
| `run_transformer_lm` | `model.py` | token embedding、层堆叠、final norm、LM head |
| `run_get_batch` | `data.py` | 随机 start index 范围、`y = x` 右移一位、device 生效 |
| `run_softmax` / `run_cross_entropy` | `nn.py` | 大 logits 不溢出、平均 loss |
| `run_gradient_clipping` | `optim.py` | 跳过 `grad is None`，按所有参数的总 L2 norm 缩放 |
| `get_adamw_cls` / LR schedule | `optim.py` | AdamW 状态、weight decay、warmup/cosine 边界 |
| checkpoint functions | `checkpoint.py` | model/optimizer state 完整恢复，iteration 原样返回 |
| `get_tokenizer` / `run_train_bpe` | `tokenizer.py` | bytes 级 vocab、merge 顺序、特殊 token、流式内存 |

## 5. 建议完成顺序

1. 环境和单测入口
   先确认能运行单个测试文件。不要一开始跑全量测试；全量失败信息太多。

2. 基础工具
   先完成 softmax、cross entropy、gradient clipping、`get_batch`。这些模块小，容易建立测试节奏。

3. 基础神经网络层
   再完成 Linear、Embedding、RMSNorm、SiLU、SwiGLU。每个函数都先写 shape 断言或临时检查，确认和 adapter docstring 一致。

4. Attention 与 RoPE
   先单独验证 scaled dot-product attention，再验证 RoPE，最后组合成 MHA。这里最容易错的是 mask 语义、head 维度重排、RoPE position 的 broadcasting。

5. Transformer block 和 LM
   先让单个 block 的 snapshot 过，再接完整 LM。失败时按 block 内部子模块逐段比对 shape 和数值范围。

6. Optimizer、LR schedule、checkpoint
   这些和模型相对解耦，适合在主模型测试通过后补齐。注意 AdamW 的实现细节需要严格对齐 handout。

7. Tokenizer 和 BPE
   BPE 最容易写慢，也最容易在特殊 token 上出边界错。建议单独处理，先追求小样例正确，再做速度和内存。

8. 训练循环与实验
   单元测试通过后再进入 TinyStories/OpenWebText 实验。训练脚本重点是可恢复、可记录、可复现实验配置。

## 6. Tokenizer 调试清单

- 内部表示优先用 bytes，而不是 Python 字符串。Unicode 文本先编码为 UTF-8 bytes，再进入 BPE。
- 特殊 token 需要作为不可拆分单元；训练和编码阶段的处理语义不同，分别对照 handout。
- 重叠特殊 token 时，长 token 应该优先匹配，否则会被短 token 抢先切开。
- BPE merge 顺序必须稳定；遇到频数相同的 pair 时，按 handout 指定的 tie-break 规则处理。
- `encode` 要通过 roundtrip 测试：`decode(encode(text)) == text`。
- `encode_iterable` 要避免一次性把大文件读入内存，测试里有 Linux 内存限制。
- 速度测试会卡住低效实现；如果小 corpus 超时，优先检查是否每轮 merge 都全量重扫了所有 pair。

## 7. Transformer 调试清单

- 给每个模块写清楚输入/输出 shape，特别是 `batch, seq, heads, d_head` 的位置。
- Linear/Embedding 的 adapter 测试使用传入权重；不要让随机初始化影响 snapshot。
- RMSNorm 只在最后一维做均方根归一化，eps 的位置按 handout。
- SwiGLU 的三个矩阵职责不同，确认输出回到 `d_model`。
- Attention 分数的最后两维应是 `queries, keys`；mask 应在 softmax 前影响这些分数。
- Causal mask 应禁止当前位置看未来 token，不应禁止看自己。
- RoPE 只作用于 Q 和 K，不作用于 V；`token_positions` 存在时优先使用传入位置。
- Full LM 支持短于 `context_length` 的输入，测试里会检查 truncated input。

## 8. 注释规范

建议把注释放在“容易出错但不显然”的地方：

- 模块顶部：说明这个文件的责任，不解释整个 Transformer。
- 类/函数 docstring：写输入 shape、输出 shape、是否有 bias、是否修改参数/梯度。
- Attention/RoPE：标注维度重排前后的 layout。
- Tokenizer：标注内部 bytes 表示、特殊 token 匹配规则、merge tie-break。
- Optimizer/checkpoint：标注哪些 state 会被原地修改或序列化。

不建议写的注释：

- “执行矩阵乘法”“返回结果”这类和代码完全重复的注释。
- 把 handout 公式整段复制到代码里。
- 在 adapter 里写大量逻辑注释；adapter 应该保持薄。

## 9. 测试策略

推荐从小到大跑测试：

- `uv run pytest tests/test_nn_utils.py`
- `uv run pytest tests/test_data.py`
- `uv run pytest tests/test_model.py -k "linear or embedding or rmsnorm or silu"`
- `uv run pytest tests/test_model.py -k "attention or rope"`
- `uv run pytest tests/test_model.py`
- `uv run pytest tests/test_optimizer.py`
- `uv run pytest tests/test_serialization.py`
- `uv run pytest tests/test_tokenizer.py`
- `uv run pytest tests/test_train_bpe.py`
- `uv run pytest`

如果一个 snapshot 测试失败，不要先调 tolerance。优先确认：

- shape 是否完全一致；
- dtype/device 是否意外变化；
- state dict key 是否映射到了正确子模块；
- mask、normalization、residual 的顺序是否和 handout 一致；
- 是否在 adapter 测试中引入了随机初始化。

## 10. 实验阶段交付物

单测通过后，再准备实验记录。每次实验至少记录：

- 数据集、tokenizer、vocab size、context length。
- 模型规模：层数、head 数、`d_model`、`d_ff`、参数量。
- 优化参数：batch size、learning rate、warmup、cosine steps、weight decay、gradient clipping。
- 训练预算：iteration、tokens seen、设备、耗时。
- 指标：train loss、valid loss、perplexity、生成样例。
- 修改项：每个 ablation 只改一个因素，方便解释。

判断完成的标准：

- 所有要求的 adapter 测试通过。
- 训练循环能从 checkpoint 恢复，并且恢复后 iteration 和 optimizer state 正常。
- TinyStories 实验有可复现配置和结果。
- OpenWebText/leaderboard 修改遵守 handout 规则。
- 你的 README 或实验笔记能解释每个关键设计选择，而不只是贴最终数字。
