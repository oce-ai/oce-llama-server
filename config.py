"""集中配置: 所有环境变量读取与默认值都在这一个文件里。

此前默认值散落在 app.py / serve.sh / serve.ps1 / llm-server.service 四处,
各自维护一份 (端口 8994、ctx 32768/32768 ...), 改一个要同步四个地方, 极易
漂移。现在统一以这里为准: 启动脚本只负责找解释器、守护、日志, 不再写死任何
业务默认值; systemd 单元也不再重复 ctx 之类。

环境变量名保持向后兼容 (PORT / HOST / MODELS_DIR / EMBED_CTX / ...)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from os import getenv
from os.path import abspath, join, dirname
from pathlib import Path

_HERE = dirname(abspath(__file__))
_QUANT_SEG = re.compile(
    r'[.\-_]'
    r'(?:'
    r'[iI][qQ]\d+(?:[._\-][A-Za-z0-9]+)*'  # IQ4_XS, IQ2_M, IQ1_S
    r'|[qQ]\d+(?:[._\-][A-Za-z0-9]+)*'  # Q4_K_M, Q8_0, Q4_0, Q5_K_S
    r'|BF16|FP16|F16|FP32|F32'  # 浮点精度
    r')'
    r'(?=[.\-_]|$)'  # 必须是完整段
)
_SHARD_RE = re.compile(r'-\d{5}-of-\d{5}$')


def _env_int(name: str, default: int) -> int:
    raw = getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"环境变量 {name} 需要整数, 当前为 {raw!r}")


def _extract_model_name(filename: str) -> str:
    # 1. 去掉目录、扩展名
    name = Path(filename).name
    name = re.sub(r'\.gguf$', '', name, flags=re.IGNORECASE)
    # 2. 去掉分片后缀
    name = _SHARD_RE.sub('', name)
    # 3. 从第一个量化段处截断
    m = _QUANT_SEG.search(name)
    if m:
        name = name[:m.start()]
    # 4. 把参数量的 "B" 统一成小写（只动结尾的数字+B）
    name = re.sub(r'(\d)\s*[Bb]$', r'\1b', name)
    # 5. 清理首尾多余的分隔符
    name = name.strip('._-')
    return name.lower()


@dataclass(frozen=True)
class Settings:
    """运行期配置。一次解析, 全程只读。"""

    host: str
    port: int
    # /v1/* 的鉴权 key; 默认值只为本地开箱即用, 暴露到网络前务必用 API_KEY 改掉。
    api_key: str
    models_dir: str
    embed_ctx: int
    rerank_ctx: int
    n_gpu_layers: int
    embed_model_path: str
    rerank_model_path: str
    projector_path: str
    # API 响应里的 model 字段标签; 中性默认值, 部署时用环境变量改成真实标识。
    embed_model_name: str
    rerank_model_name: str


def load_settings() -> Settings:
    models_dir = getenv("EMBED_DIR") or join(_HERE, "models")
    rerank_dir = getenv("RERANK_DIR", join(models_dir, "jina-reranker-v3.5"))
    embed_model_path = getenv(
        "EMBED_MODEL_PATH",
        join(models_dir, "F2LLM-v2-0.6B.Q4_K_M.gguf"),
    )
    rerank_model_path = getenv(
        "RERANK_MODEL_DIR",
        join(rerank_dir, "jina-reranker-v3.5-Q4_K_M.gguf"),
    )

    return Settings(
        host=getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8994),
        api_key=getenv("API_KEY") or "sk-oce-llama-server",
        models_dir=models_dir,
        embed_ctx=_env_int("EMBED_CTX", 32768),
        rerank_ctx=_env_int("RERANK_CTX", 32768),
        n_gpu_layers=_env_int("N_GPU_LAYERS", 99),
        embed_model_path=embed_model_path,
        rerank_model_path=rerank_model_path,
        projector_path=getenv("PROJECTOR_PATH", join(rerank_dir, "projector.safetensors")),
        embed_model_name=_extract_model_name(embed_model_path),
        rerank_model_name=_extract_model_name(rerank_model_path),
    )


# 进程级单例: import 即解析。需要的话可调用 load_settings() 重新读取。
settings = load_settings()

if __name__ == "__main__":
    import json
    from dataclasses import asdict

    print(json.dumps(asdict(settings), indent=2, ensure_ascii=False))
