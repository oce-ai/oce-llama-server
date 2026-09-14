"""F2LLM 嵌入 与 jina-reranker-v3 重排的进程内封装。

两个模型各持独立的 llama context, 由 ModelRegistry 分别加锁, 因此嵌入与
重排可以并行执行。

改动 llama.cpp 参数前请先读 README 的 "参数约束" 一节: 其中有三个参数
失效时不会报错, 只会让输出数值悄悄改变。
"""
from __future__ import annotations

import json
import struct
import threading
from typing import Dict, List, Optional

import numpy as np
from llama_cpp import Llama, llama_cpp
from numpy.ctypeslib import as_array


class MLPProjector:
    """jina-reranker-v3 的打分头: 1024 -> 512 -> ReLU -> 512。"""

    def __init__(self, w0: np.ndarray, w2: np.ndarray):
        self.w0 = w0.astype(np.float32)
        self.w2 = w2.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x @ self.w0.T) @ self.w2.T


def _read_safetensor(path: str, key: str) -> np.ndarray:
    """纯 numpy 读取 safetensors 里的单个张量。

    不依赖 torch / safetensors 的 numpy framework: numpy 本身没有 bfloat16 类型
    (任何版本都没有, 它是 torch / ml_dtypes 的类型), 所以 safetensors 的 numpy
    framework 读 BF16 张量会抛 "data type 'bfloat16' not understood", 而 pt
    framework 又需要装 torch。这里自己解析 header + 按 offset 读原始字节, BF16
    通过左移 16 位补成 float32。
    """
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        meta = header[key]
        dtype, shape = meta["dtype"], meta["shape"]
        start, end = meta["data_offsets"]
        fh.seek(8 + header_len + start)
        raw = fh.read(end - start)

    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F64":
        arr = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    else:
        raise ValueError(f"projector 含不支持的 dtype: {dtype}")

    return arr.reshape(shape).astype(np.float32)


def load_projector(path: str) -> MLPProjector:
    return MLPProjector(
        _read_safetensor(path, "projector.0.weight"),
        _read_safetensor(path, "projector.2.weight"),
    )


def sanitize_input(text: str, special_tokens: Dict[str, str]) -> str:
    """移除文本中出现的特殊 token 字面量, 避免污染 prompt 结构。"""
    for tok in special_tokens.values():
        text = text.replace(tok, "")
    return text


def format_rerank_prompt(
    query: str,
    docs: List[str],
    instruction: Optional[str] = None,
    special_tokens: Optional[Dict[str, str]] = None,
) -> str:
    """按官方 listwise 格式打包: 所有 passage 放进同一个 prompt。

    相比逐篇(pointwise)打分, listwise 保留了 passage 之间的注意力,
    与模型训练时的输入形式一致。
    """
    special_tokens = special_tokens or {}
    query = sanitize_input(query, special_tokens)
    docs = [sanitize_input(d, special_tokens) for d in docs]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages "
        "based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"

    doc_emb_token = special_tokens["doc_embed_token"]
    query_emb_token = special_tokens["query_embed_token"]

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    if instruction:
        prompt += f"<instruct>\n{instruction}\n</instruct>\n"

    prompt += "\n".join(
        f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>'
        for i, doc in enumerate(docs)
    ) + "\n"
    prompt += f"<query>\n{query}{query_emb_token}\n</query>"

    return prefix + prompt + suffix


class EmbeddingModel:
    """F2LLM 嵌入模型, 输出 L2 归一化的 1024 维向量。"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        kwargs = dict(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_ubatch=min(n_ctx, 2048),
            n_gpu_layers=n_gpu_layers,
            flash_attn=True,  # 必须为 True, 见 README "参数约束"
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
            verbose=verbose,
        )
        if n_threads:
            kwargs["n_threads"] = n_threads
        self.llm = Llama(**kwargs)
        self._ctx_ptr = self.llm._ctx.ctx
        self._n_embd = llama_cpp.llama_n_embd(self.llm._model.model)

    @property
    def n_ctx(self) -> int:
        return self.llm.n_ctx()

    @property
    def n_embd(self) -> int:
        return self._n_embd

    def n_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True))

    def embed(self, text: str) -> np.ndarray:
        tokens = self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)
        if len(tokens) > self.n_ctx:
            raise ValueError(
                f"input is {len(tokens)} tokens > n_ctx={self.n_ctx}; "
                f"split the text or raise EMBED_CTX"
            )

        self.llm.reset()
        self.llm._ctx.kv_cache_clear()
        self.llm._batch.reset()
        self.llm._batch.add_sequence(tokens, 0, True)
        self.llm._ctx.decode(self.llm._batch)

        ptr = llama_cpp.llama_get_embeddings_seq(self._ctx_ptr, 0)
        if not ptr:
            raise RuntimeError("llama_get_embeddings_seq returned NULL")
        v = np.asarray(ptr[: self._n_embd], dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """逐条嵌入。

        不用 llama_cpp 的批量 embed(): 它在多条输入时会共用一次 decode, 且
        内部按 n_batch 静默截断。逐条调用结果与单条完全一致, 并让超长输入
        能被显式拒绝。
        """
        return [self.embed(t) for t in texts]


class RerankerModel:
    """jina-reranker-v3 listwise 重排, 打分头是 GGUF 外部的 projector.safetensors。"""

    DOC_EMBED_TOKEN_ID = 151670
    QUERY_EMBED_TOKEN_ID = 151671

    def __init__(
        self,
        model_path: str,
        projector_path: str,
        n_ctx: int = 32768,
        n_ubatch: int = 512,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        kwargs = dict(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_batch=n_ctx,  # 必须 >= n_ctx, 见 README "参数约束"
            n_ubatch=min(n_ctx, n_ubatch),
            n_gpu_layers=n_gpu_layers,
            flash_attn=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE,
            logits_all=False,
            verbose=verbose,
        )
        if n_threads:
            kwargs["n_threads"] = n_threads
        self.llm = Llama(**kwargs)
        self.projector = load_projector(projector_path)
        self._ctx_ptr = self.llm._ctx.ctx
        self._n_embd = llama_cpp.llama_n_embd(self.llm._model.model)
        self._special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>",
        }

    @property
    def n_ctx(self) -> int:
        return self.llm.n_ctx()

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        prompt = format_rerank_prompt(
            query, documents, instruction=instruction, special_tokens=self._special_tokens
        )

        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        tokens_array = np.asarray(tokens)

        if len(tokens) > self.n_ctx:
            raise ValueError(
                f"prompt is {len(tokens)} tokens > n_ctx={self.n_ctx}; "
                f"send fewer/shorter documents or raise RERANK_CTX"
            )

        q_pos = np.where(tokens_array == self.QUERY_EMBED_TOKEN_ID)[0]
        d_pos = np.where(tokens_array == self.DOC_EMBED_TOKEN_ID)[0]
        if len(q_pos) == 0:
            raise ValueError(f"query embed token {self.QUERY_EMBED_TOKEN_ID} not found")
        if len(d_pos) != len(documents):
            raise ValueError(f"expected {len(documents)} doc tokens, found {len(d_pos)}")

        # hidden[0] = query, hidden[1:] = 各文档 (顺序与 documents 一致)
        want = np.concatenate(([q_pos[0]], d_pos))

        self.llm.reset()
        self.llm.eval(tokens)

        ptr = llama_cpp.llama_get_embeddings(self._ctx_ptr)
        if not ptr:
            raise RuntimeError("llama_get_embeddings returned NULL")
        view = as_array(ptr, shape=(len(tokens), self._n_embd))
        hidden = view[want].astype(np.float32, copy=False)

        q = self.projector(hidden[0:1])[0]
        doc_emb = self.projector(hidden[1:])

        q_norm = float(np.linalg.norm(q)) + 1e-12
        d_norms = np.linalg.norm(doc_emb, axis=-1) + 1e-12
        scores = (doc_emb @ q) / (d_norms * q_norm)

        results = [
            {"index": idx, "relevance_score": float(score), "document": doc}
            for idx, (doc, score) in enumerate(zip(documents, scores))
        ]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_n] if top_n is not None else results


class ModelRegistry:
    """持有两个模型及各自的锁。

    一个 llama context 不是线程安全的, 所以每个模型用一把锁串行化自己的推理。
    两个模型的 context 相互独立, 因此嵌入与重排可以并行, 互不阻塞。
    """

    def __init__(self, embedder: EmbeddingModel, reranker: RerankerModel):
        self.embedder = embedder
        self.reranker = reranker
        self._embed_lock = threading.Lock()
        self._rerank_lock = threading.Lock()

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        with self._embed_lock:
            return self.embedder.embed_batch(texts)

    def rerank(self, *args, **kwargs) -> List[Dict]:
        with self._rerank_lock:
            return self.reranker.rerank(*args, **kwargs)

    def n_tokens(self, text: str) -> int:
        with self._embed_lock:
            return self.embedder.n_tokens(text)
