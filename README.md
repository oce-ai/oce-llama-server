# oce-llama-server

单进程、单端口，同时提供**文本嵌入**与**重排**服务，基于
[llama.cpp](https://github.com/ggml-org/llama.cpp) 在本地运行 GGUF 模型（GPU 优先）。

| 端点 | 说明 |
|---|---|
| `POST /v1/embeddings` | OpenAI 兼容的文本嵌入 |
| `POST /v1/rerank` | Jina / Cohere 风格的重排 |
| `GET /v1/models` | OpenAI 兼容的模型列表 |
| `GET /health` | 服务与模型状态 |

两个模型各持独立的 llama context 和一把锁：嵌入与重排请求可并行执行、互不阻塞；
同类请求之间串行排队。

## 模型文件

| 用途 | 模型 | 默认路径（相对模型根目录） |
|---|---|---|
| 嵌入（L2 归一化，1024 维） | F2LLM-v2-0.6B | `F2LLM-v2-0.6B.Q4_K_M.gguf` |
| 重排（listwise） | jina-reranker-v3.5 | `jina-reranker-v3.5/jina-reranker-v3.5-Q4_K_M.gguf` |

重排模型的打分头在 GGUF **外部**：`jina-reranker-v3.5/projector.safetensors`（BF16 权重）
必须一并下载，缺少它无法计算相关性分数。默认目录结构（`EMBED_DIR` 可覆盖根目录）：

```
models/
  F2LLM-v2-0.6B.Q4_K_M.gguf
  jina-reranker-v3.5/
    jina-reranker-v3.5-Q4_K_M.gguf
    projector.safetensors
```

## 快速开始

### 1. 安装依赖

需要 Python ≥ 3.10（`.python-version` 锁定 3.12），用 [uv](https://docs.astral.sh/uv/) 管理：

```bash
uv venv
uv sync        # 安装 fastapi / uvicorn / numpy
```

### 2. 安装 llama-cpp-python（GPU 加速）

llama-cpp-python **不在 pyproject.toml 里**：它需要匹配本机 GPU 的预编译 wheel，
写进依赖会让 `uv sync` 触发源码编译（需要 nvcc，慢且易失败）。用自带脚本从
GitHub release 自动探测并安装：

```bash
uv run python scripts/install_llama_cpp.py                 # 自动探测 CUDA/Metal/ROCm，装最佳 wheel
uv run python scripts/install_llama_cpp.py --dry-run       # 只打印将执行的命令，不实际安装
uv run python scripts/install_llama_cpp.py --version 0.3.35
uv run python scripts/install_llama_cpp.py --backend cpu   # 强制后端，跳过探测
uv run python scripts/install_llama_cpp.py --no-wheels     # 跳过 wheel，直接源码编译
```

找不到预编译 wheel 时自动回退 PyPI 源码编译。不常见的 GPU 架构（如 P40/P100）
源码编译需显式指定计算能力，否则默认 kernel 可能不含你的卡：

```bash
# Pascal (P40 / P100, sm_61)
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61" uv pip install llama-cpp-python
```

设置 `GITHUB_TOKEN` 环境变量可提高 GitHub API 速率限制（匿名 60 次/小时）。

### 3. 启动

```bash
uv run app.py
```

模型不在默认 `models/` 目录时，用环境变量指定根目录。Windows PowerShell：

```powershell
$env:EMBED_DIR = "D:\path\to\models"
uv run app.py
```

Linux / macOS：

```bash
EMBED_DIR=/path/to/models uv run app.py
```

启动成功会输出类似：

```
INFO oce-llama-server: ready in 0.9s (embed 32768 ctx/1024 dim, rerank 32768 ctx)
INFO:     Uvicorn running on http://127.0.0.1:8994 (Press CTRL+C to quit)
```

### 4. 验证

`/health` 不需要鉴权，`/v1/*` 需要带 `Authorization` 头（下面是默认 key）：

```bash
curl http://127.0.0.1:8994/health

curl http://127.0.0.1:8994/v1/models -H "Authorization: Bearer sk-oce-llama-server"

curl -X POST http://127.0.0.1:8994/v1/embeddings -H "Authorization: Bearer sk-oce-llama-server" -H "Content-Type: application/json" -d '{"input": "hello world"}'

curl -X POST http://127.0.0.1:8994/v1/rerank -H "Authorization: Bearer sk-oce-llama-server" -H "Content-Type: application/json" -d '{"query": "猫", "documents": ["狗", "猫科动物", "汽车"]}'
```

## 配置

所有默认值只在 `config.py` 一处定义，环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `127.0.0.1` | 绑定地址 |
| `PORT` | `8994` | 服务端口 |
| `API_KEY` | `sk-oce-llama-server` | 访问 `/v1/*` 的 Bearer token |
| `EMBED_DIR` | `<项目根>/models` | 模型文件根目录 |
| `RERANK_DIR` | `<EMBED_DIR>/jina-reranker-v3.5` | 重排模型所在目录 |
| `EMBED_MODEL_PATH` | `<EMBED_DIR>/F2LLM-v2-0.6B.Q4_K_M.gguf` | 嵌入模型 GGUF 路径 |
| `RERANK_MODEL_DIR` | `<RERANK_DIR>/jina-reranker-v3.5-Q4_K_M.gguf` | 重排模型 GGUF 路径（历史命名，实际是文件路径） |
| `PROJECTOR_PATH` | `<RERANK_DIR>/projector.safetensors` | 重排打分头权重路径 |
| `EMBED_CTX` | `32768` | 嵌入上下文长度，单条输入超限报 400 |
| `RERANK_CTX` | `32768` | 重排上下文长度，整个 prompt 超限报 400 |
| `N_GPU_LAYERS` | `99` | GPU 卸载层数，两个模型共用；`0` 为纯 CPU |

API 响应里的 `model` 字段**不通过环境变量配置**：由模型文件名自动推导（去掉量化
后缀、统一参数量写法），如 `F2LLM-v2-0.6B.Q4_K_M.gguf` → `F2LLM-v2-0.6b`、
`jina-reranker-v3.5-Q4_K_M.gguf` → `jina-reranker-v3.5`。请求体传入的 `model` 字段优先。

运行 `uv run python config.py` 可打印当前生效的完整配置（JSON），排查路径问题时先跑它。

## API

### 鉴权

`/v1/*` 全部需要 `Authorization: Bearer <API_KEY>` 头，缺失或错误返回 **401**：

```bash
curl http://127.0.0.1:8994/v1/models -H "Authorization: Bearer sk-oce-llama-server"
```

`/health` 不鉴权，供负载均衡/监控探活。默认的 `sk-oce-llama-server` 只为本地开箱
即用，**暴露到网络前务必用 `API_KEY` 环境变量改掉**。

### POST /v1/embeddings

请求：`input` 接受字符串或字符串数组；`model` 可选，原样回显到响应。

```json
{"input": ["hello world", "another text"], "model": "embed"}
```

响应（OpenAI 兼容）：

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.013, -0.081, "..."]},
    {"object": "embedding", "index": 1, "embedding": ["..."]}
  ],
  "model": "F2LLM-v2-0.6b",
  "usage": {"prompt_tokens": 7, "total_tokens": 7}
}
```

向量为 L2 归一化的 1024 维 float32。批量输入逐条嵌入（结果与单条调用完全一致），
不做内部静默截断。

### POST /v1/rerank

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | string | 必填 |
| `documents` | string[] | 文档列表；`texts` 为兼容别名，二选一 |
| `top_n` | int | 可选，只返回前 N 条 |
| `instruction` | string | 可选，重排指令，注入 prompt |
| `return_documents` | bool | 默认 `false`，为 `true` 时结果带回原文 |
| `model` | string | 可选，原样回显到响应 |

响应（按相关性降序）：

```json
{
  "model": "jina-reranker-v3.5",
  "object": "list",
  "results": [
    {"index": 1, "relevance_score": 0.397},
    {"index": 0, "relevance_score": -0.138}
  ]
}
```

重排是 **listwise** 的：所有文档打包进同一个 prompt 一次推理，保留文档间的注意力，
与模型训练形式一致。分数是投影后向量的余弦相似度，范围 `[-1, 1]`——不是概率，
只有同批文档内的相对顺序有意义。文档里的特殊 token 字面量（`<|rerank_token|>`、
`<|embed_token|>`）会被自动移除，避免污染 prompt 结构。

### GET /v1/models

列出本服务提供的模型（OpenAI 兼容格式）。`id` 与 `/v1/embeddings`、`/v1/rerank`
响应里的 `model` 字段一致，客户端拿到后可直接回填使用。该端点只依赖 `config`，
模型加载完成前也能查询。

```json
{
  "object": "list",
  "data": [
    {"id": "F2LLM-v2-0.6b", "object": "model", "created": 1788855512, "owned_by": "oce-llama-server"},
    {"id": "jina-reranker-v3.5", "object": "model", "created": 1789054089, "owned_by": "oce-llama-server"}
  ]
}
```

`created` 为模型文件的修改时间（Unix 秒）。

### GET /health

模型就绪返回 `200`，加载中返回 `503`：

```json
{"status": "ok", "embed_ctx": 32768, "embed_dim": 1024, "rerank_ctx": 32768}
```

### 错误行为

- 输入超出 `EMBED_CTX` / 重排 prompt 超出 `RERANK_CTX`：返回 **400** 并说明超出多少，
  **不静默截断**
- 空 `input` / 空 `documents`：400
- 模型未加载完成：503
- 其他推理异常：500，详情见服务端日志

## 参数约束（改 models.py 前必读）

有三个 llama.cpp 参数**失效时不报错，只会让输出数值悄悄改变**：

1. `flash_attn=True` —— 两个模型都必须开。关掉后注意力数值路径不同，嵌入向量
   和重排分数都会漂移。
2. `n_batch=n_ctx` —— 必须 ≥ 实际输入长度，否则超长部分被静默丢弃，嵌入结果
   变成截断文本的向量，不报任何错。
3. `pooling_type` —— 嵌入用 `LAST`（取末 token），重排用 `NONE`（要每个 token
   位置的 hidden state 来挑 `<|embed_token|>` 处）。设错不会崩溃，只会取错位置，
   分数悄悄变成垃圾。

## 项目结构

```
oce-llama-server/
├── app.py                      # FastAPI 入口：端点、请求模型、错误处理
├── config.py                   # 配置单一来源（环境变量 + 默认值），可独立运行打印配置
├── models.py                   # 模型封装：嵌入、重排、projector 解析、双锁注册表
├── scripts/
│   └── install_llama_cpp.py    # llama-cpp-python 预编译 wheel 自动安装
├── pyproject.toml              # uv 项目配置（package = false）
├── uv.lock                     # 依赖锁
└── .python-version             # Python 版本锁定（3.12）
```

`models.py` 里的 `load_projector` 用纯 numpy 手动解析 safetensors（BF16 左移 16 位
补成 float32）：numpy 没有 bfloat16 类型，`safetensors` 库的 numpy framework 读
BF16 会直接报错，pt framework 又要求装 torch，所以不依赖两者。

## 故障排查

### 启动报 "Model path does not exist"

模型目录不对。设置 `EMBED_DIR`（见「快速开始 → 启动」），或跑 `uv run python config.py`
查看当前解析出的路径。注意嵌入模型默认在根目录下，重排模型默认在
`<EMBED_DIR>/jina-reranker-v3.5/` 子目录下。

### 启动日志出现 "pooling_type is [-1], but [0] was specified"

无害警告：重排模型 GGUF 元数据没写默认 pooling，代码显式指定了 `NONE`，
正是需要的行为，可忽略。

### llama-cpp-python 安装失败

```bash
nvidia-smi                                            # 确认驱动支持的 CUDA 版本
uv run python scripts/install_llama_cpp.py --dry-run   # 预览将选择的 wheel
uv run python scripts/install_llama_cpp.py --backend cuda
uv run python scripts/install_llama_cpp.py --no-wheels # 强制源码编译（需要 nvcc）
```

### Windows 启动报 "Failed to load shared library ... llama.dll"

```
RuntimeError: Failed to load shared library
'...\.venv\Lib\site-packages\llama_cpp\lib\llama.dll':
Could not find module '...\llama.dll' (or one of its dependencies).
```

**关键是报错末尾的 `(or one of its dependencies)`**：`llama.dll` 文件本身在，缺的是它
依赖的 CUDA 运行时库（`cudart64_*.dll`、`cublas64_*.dll`、`cublasLt64_*.dll`）。
CUDA 版的 llama-cpp-python wheel **不打包**这些库，所以加载连带失败，Windows 却把错
报在 `llama.dll` 上，很有迷惑性。

解决：去 [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 下载与
**wheel 的 CUDA 版本一致**的 cudart 包，解压后把里面的 dll 放进 `llama_cpp\lib\`
（即和 `llama.dll` 同目录）：

```
# 例：wheel 是 CUDA 13.x 的，就下对应 cudart 包
cudart-llama-bin-win-cuda-13.3-x64.zip
  └─ cudart64_*.dll / cublas64_*.dll / cublasLt64_*.dll  →  复制到 .venv\Lib\site-packages\llama_cpp\lib\
```

不知道 wheel 是哪个 CUDA 版本时，跑安装脚本预览，tag 里的 `-cu13x` 就是：

```bash
uv run python scripts/install_llama_cpp.py --dry-run   # 看选中的 wheel tag, 如 ...-cu132
```

> cudart 版本号要和 wheel 对得上（wheel 是 cu132 就配 CUDA 13.x 的 cudart），否则
> 仍会因 dll 版本不符而加载失败。装完重启服务即可。

### Linux + conda 启动报 "GLIBCXX_3.4.30 not found"

```
RuntimeError: Failed to load shared library
'.../site-packages/llama_cpp/lib/libllama.so':
.../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
(required by .../llama_cpp/lib/libggml-cuda.so.0)
```

conda 自带的 `libstdc++.so.6` 版本太老，不含 CUDA wheel 需要的 `GLIBCXX_3.4.30`
符号（gcc 12+ 才提供）。注意报错指向的 `libstdc++.so.6` 是 **conda 环境里**的那个
（在 `$CONDA_PREFIX/lib/`），系统自带的可能反而是新的——所以不能只看系统版本。

解决：装 conda-forge 的 `libstdcxx-ng` 覆盖老版本：

```bash
conda install -c conda-forge libstdcxx-ng -y
```

验证（确认 `$CONDA_PREFIX/lib/libstdc++.so.6` 已包含所需符号）：

```bash
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
```

> 若装完仍报错，检查 `LD_LIBRARY_PATH` 是否把另一个旧的 `libstdc++.so.6` 排在了
> conda 环境前面；必要时 `unset LD_LIBRARY_PATH` 后重启服务再试。

### 显存不足 / 想纯 CPU

降低上下文长度或关闭 GPU 卸载：

```bash
EMBED_CTX=8192 RERANK_CTX=16384 uv run app.py
N_GPU_LAYERS=0 uv run app.py
```

## 依赖

- Python ≥ 3.10（锁定 3.12）
- `fastapi`、`uvicorn`、`numpy`（`uv sync` 安装）
- `llama-cpp-python`（`scripts/install_llama_cpp.py` 安装，刻意不进 pyproject.toml）

## 性能参考

RTX 5070 实测（Q4_K_M 量化）：

| 场景 | 耗时 |
|---|---|
| 嵌入 32 条批量 | 207 ms |
| 嵌入 8 并发 × 8 条 | 0.35 s |
| 重排 80 篇 | 0.88 s |
| 重排 8 并发 × 40 篇 | 3.03 s |
| 嵌入 + 重排同时请求 | 墙钟 0.38 s |

最后一行是单进程双模型的收益：两个模型 context 独立、各持一把锁，同一端口上的
嵌入与重排真正并行。

显存占用主要由 ctx 决定（KV cache 约 112 KiB/token）：默认两个模型都是 32768，
KV cache 合计约 7 GB；显存紧张时优先降 `EMBED_CTX` / `RERANK_CTX`（见「显存不足」）。
