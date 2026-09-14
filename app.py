"""本地推理服务: 一个进程、一个端口, 同时提供嵌入与重排。

  POST /v1/embeddings   OpenAI 兼容
  POST /v1/rerank       Jina/Cohere 兼容
  GET  /v1/models       OpenAI 兼容模型列表
  GET  /health          状态
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import threading
import time
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from models import EmbeddingModel, ModelRegistry, RerankerModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("oce-llama-server")

# 所有配置 (端口/ctx/模型路径/标签) 都来自 config.settings, 默认值集中在那一处。
registry: Optional[ModelRegistry] = None
_load_lock = threading.Lock()


def load_models() -> ModelRegistry:
    """幂等加载两个模型。"""
    global registry
    with _load_lock:
        if registry is not None:
            return registry
        t0 = time.perf_counter()
        logger.info("loading embedding model: %s", settings.embed_model_path)
        embedder = EmbeddingModel(settings.embed_model_path, n_ctx=settings.embed_ctx,
                                  n_gpu_layers=settings.n_gpu_layers)
        logger.info("loading reranker: %s", settings.rerank_model_path)
        reranker = RerankerModel(settings.rerank_model_path, settings.projector_path,
                                 n_ctx=settings.rerank_ctx,
                                 n_gpu_layers=settings.n_gpu_layers)
        registry = ModelRegistry(embedder, reranker)
        logger.info(
            "ready in %.1fs (embed %d ctx/%d dim, rerank %d ctx)",
            time.perf_counter() - t0, embedder.n_ctx, embedder.n_embd, reranker.n_ctx,
        )
        return registry


app = FastAPI(title="oce-llama-server")


async def require_api_key(authorization: Optional[str] = Header(None)) -> None:
    """校验 ``Authorization: Bearer <API_KEY>``, 不通过返回 401。

    只挂在 /v1/* 端点上; /health 保持无鉴权, 供负载均衡/监控探活。用
    hmac.compare_digest 做定长比较, 避免逐字节短路比较泄露 key 的时序信息。
    """
    if not authorization:
        raise HTTPException(
            401, "missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        token.strip(), settings.api_key
    ):
        raise HTTPException(
            401, "invalid api key", headers={"WWW-Authenticate": "Bearer"}
        )


@app.get("/health")
async def health():
    if registry is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {
        "status": "ok",
        "embed_ctx": registry.embedder.n_ctx,
        "embed_dim": registry.embedder.n_embd,
        "rerank_ctx": registry.reranker.n_ctx,
    }


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    """OpenAI 兼容的模型列表。

    模型名来自 config (由文件名推导), 不依赖 registry, 因此模型加载完成前
    也能查询。`id` 与 /v1/embeddings、/v1/rerank 响应里的 model 字段一致,
    客户端拿到后可直接回填使用。
    """

    def _created(path: str) -> int:
        try:
            return int(os.path.getmtime(path))
        except OSError:
            return 0

    return {
        "object": "list",
        "data": [
            {
                "id": settings.embed_model_name,
                "object": "model",
                "created": _created(settings.embed_model_path),
                "owned_by": "oce-llama-server",
            },
            {
                "id": settings.rerank_model_name,
                "object": "model",
                "created": _created(settings.rerank_model_path),
                "owned_by": "oce-llama-server",
            },
        ],
    }


# ─────────────────────────── 嵌入 (OpenAI 兼容) ───────────────────────────


class EmbeddingRequest(BaseModel):
    input: Any = Field(..., description="字符串或字符串数组")
    model: Optional[str] = None


@app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
async def embeddings(req: EmbeddingRequest):
    if registry is None:
        raise HTTPException(503, "models not loaded")

    texts = req.input if isinstance(req.input, list) else [req.input]
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    if not texts:
        raise HTTPException(400, "input must not be empty")

    try:
        # 逐条嵌入 (与单条调用结果一致), 放线程池避免阻塞事件循环
        vecs = await asyncio.to_thread(registry.embed_batch, texts)
    except ValueError as e:
        # 超长输入: 显式 400, 不静默截断
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("embedding failed")
        raise HTTPException(500, "embedding failed - see server log")

    data = [
        {"object": "embedding", "index": i, "embedding": v.tolist()}
        for i, v in enumerate(vecs)
    ]
    tokens = await asyncio.to_thread(
        lambda: sum(registry.embedder.n_tokens(t) for t in texts)
    )
    return {
        "object": "list",
        "data": data,
        "model": req.model or settings.embed_model_name,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


# ─────────────────────────── 重排 (Jina/Cohere 风格) ───────────────────────────


class RerankRequest(BaseModel):
    query: str
    documents: Optional[List[str]] = None
    texts: Optional[List[str]] = None  # 兼容别名
    top_n: Optional[int] = None
    instruction: Optional[str] = None
    return_documents: bool = False
    model: Optional[str] = None


@app.post("/v1/rerank", dependencies=[Depends(require_api_key)])
async def rerank(req: RerankRequest):
    if registry is None:
        raise HTTPException(503, "models not loaded")

    docs = req.documents if req.documents is not None else req.texts
    if not docs:
        raise HTTPException(400, "documents must not be empty")

    try:
        results = await asyncio.to_thread(
            registry.rerank, req.query, docs, req.top_n, req.instruction
        )
    except ValueError as e:
        # 超 n_ctx / 特殊 token 缺失: 显式错误, 不静默出错
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("rerank failed")
        raise HTTPException(500, "rerank failed - see server log")

    return {
        "model": req.model or settings.rerank_model_name,
        "object": "list",
        "results": [
            {
                "index": r["index"],
                "relevance_score": r["relevance_score"],
                **({"document": r["document"]} if req.return_documents else {}),
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    import uvicorn

    # 单进程单端点: 嵌入与重排都在同一端口。两个模型各持独立 ctx 和锁,
    # 所以嵌入和重排的请求可以真正并行, 互不阻塞。
    load_models()
    logger.info("embed : http://%s:%d/v1/embeddings", settings.host, settings.port)
    logger.info("rerank: http://%s:%d/v1/rerank", settings.host, settings.port)
    try:
        uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("shutting down")
