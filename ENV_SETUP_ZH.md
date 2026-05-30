# CS336 HW1 uv 环境配置记录

这份文档记录本次环境配置过程，以及之后在本机或 GPU 服务器上重新配置环境时需要执行的命令。

## 1. 当前配置结果

本次在 `cs336_hw1` 目录下完成了：

- 使用 `uv sync` 创建项目虚拟环境 `.venv`。
- 使用本地 uv cache 目录 `.uv-cache`，避免写入 home 目录下可能只读的默认 cache。
- 安装了 `uv.lock` 中锁定的依赖，包括 `torch==2.11.0`、`numpy==2.4.4`、`tiktoken==0.12.0`、`pytest==9.0.2` 等。

验证结果：

```text
Python: 3.12.3
PyTorch: 2.11.0+cu130
CUDA available on current machine: False
```

当前机器没有可用 CUDA 设备，所以本地适合跑 CPU 单元测试和小规模调试。需要 GPU 的训练实验应放到 GPU 服务器上跑。

## 2. 从零配置环境

进入作业目录：

```sh
cd /home/yszn/code/cs336/cs336_hw1
```

确认 `uv` 可用：

```sh
uv --version
```

创建本地 cache 目录：

```sh
mkdir -p .uv-cache
```

按锁文件同步环境：

```sh
UV_CACHE_DIR=.uv-cache uv sync
```

如果服务器或本机的 home 目录可写，也可以直接运行：

```sh
uv sync
```

但为了可复现和避免权限问题，推荐统一使用 `UV_CACHE_DIR=.uv-cache`。

## 3. 验证环境

同步完成后，检查 Python、PyTorch 和 CUDA 状态：

```sh
UV_CACHE_DIR=.uv-cache uv run python -c "import sys, torch; print(sys.version.split()[0]); print(torch.__version__); print(torch.cuda.is_available())"
```

预期：

- Python 版本满足 `pyproject.toml` 的 `>=3.12,<3.14`。
- PyTorch 能正常 import。
- 在 GPU 服务器上，`torch.cuda.is_available()` 应该输出 `True`。

如果在 GPU 服务器上输出 `False`，优先检查：

- 是否申请到了 GPU 节点，而不是登录节点。
- `nvidia-smi` 是否能看到 GPU。
- 服务器驱动是否支持当前 PyTorch/CUDA wheel。
- 作业是否需要通过调度系统启动，例如 `srun`、`sbatch` 或平台提供的交互式 GPU shell。

## 4. 运行测试

单个测试文件：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=.uv-cache uv run pytest tests/test_nn_utils.py
```

全量测试：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=.uv-cache uv run pytest
```

建议在实现早期只跑小测试，避免大量 `NotImplementedError` 淹没关键信息。

## 5. GPU 服务器重新配置流程

在 GPU 服务器上重新配置时，推荐流程如下：

1. 克隆或同步仓库到服务器。
2. 进入 `cs336_hw1` 目录。
3. 确认 `uv --version` 可用；如果没有 uv，按服务器允许的方式安装。
4. 执行 `mkdir -p .uv-cache`。
5. 执行 `UV_CACHE_DIR=.uv-cache uv sync`。
6. 执行环境验证命令，确认 `torch.cuda.is_available()` 为 `True`。
7. 先跑轻量测试，再跑训练脚本。

常用命令汇总：

```sh
cd /path/to/cs336_hw1
mkdir -p .uv-cache
UV_CACHE_DIR=.uv-cache uv sync
UV_CACHE_DIR=.uv-cache uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=.uv-cache uv run pytest tests/test_nn_utils.py
```

## 6. 下载数据

README 中给出的数据下载命令如下。建议在 GPU 服务器或有足够磁盘的机器上下载：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

如果网络受限，可以先在能访问 Hugging Face 的机器下载，再把 `data/` 目录传到服务器。

## 7. 常见问题

### uv 试图写入 home cache 失败

现象类似：

```text
Could not create temporary file
Read-only file system at /home/.../.cache/uv
```

解决：

```sh
mkdir -p .uv-cache
UV_CACHE_DIR=.uv-cache uv sync
```

### 下载依赖失败

如果报网络错误或 PyPI 连接失败，通常是服务器网络限制。处理方式：

- 在允许联网的节点运行 `uv sync`。
- 使用学校/集群提供的代理或镜像。
- 让调度系统分配到允许外网的交互式节点。

### `.venv` 占用空间较大

本次 `.venv` 大约 4.7G，主要来自 PyTorch 和 CUDA wheel。GPU 服务器上需要提前确认作业目录或 scratch 目录空间足够。

### pytest 被系统插件干扰

本机运行 pytest 时，系统里的 ROS pytest 插件会自动加载，并可能因为缺少 `lark` 等依赖而在收集测试前失败。这个问题和作业代码无关。

解决方式是在 pytest 命令前加：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

例如：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=.uv-cache uv run pytest --collect-only -q
```

### 想重新创建环境

如果环境损坏，可以删除 `.venv` 和本地 cache 后重建：

```sh
rm -rf .venv .uv-cache
mkdir -p .uv-cache
UV_CACHE_DIR=.uv-cache uv sync
```

注意：这个命令会删除本地虚拟环境和依赖缓存，但不会删除你的代码。
