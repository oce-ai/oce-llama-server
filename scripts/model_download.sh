#!/bin/bash
set -euo pipefail

# ========== 定位路径 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"
RERANKER_DIR="${MODELS_DIR}/jina-reranker-v3.5"

# ========== 模型 URL ==========
EMBED_MODEL_URL="https://huggingface.co/mradermacher/F2LLM-v2-0.6B-GGUF/resolve/main/F2LLM-v2-0.6B.Q4_K_M.gguf"
EMBED_MODEL_NAME="F2LLM-v2-0.6B.Q4_K_M.gguf"

RERANKER_BASE="https://huggingface.co/jinaai/jina-reranker-v3.5-GGUF/resolve/main"
RERANKER_MODEL_NAME="jina-reranker-v3.5-Q4_K_M.gguf"
RERANKER_PROJECTOR_NAME="projector.safetensors"

# ========== 可选：HuggingFace 镜像 ==========
# export HF_ENDPOINT=https://hf-mirror.com

# ========== 1. 创建目录 ==========
mkdir -p "${MODELS_DIR}"
mkdir -p "${RERANKER_DIR}"

# ========== 2. 下载 embedding 模型 ==========
if [ -f "${MODELS_DIR}/${EMBED_MODEL_NAME}" ]; then
    echo ">>> 已存在 ${EMBED_MODEL_NAME}，跳过下载"
else
    echo ">>> 下载 ${EMBED_MODEL_NAME}"
    wget -c -O "${MODELS_DIR}/${EMBED_MODEL_NAME}" "${EMBED_MODEL_URL}"
fi

# ========== 3. 下载 reranker 模型 ==========
if [ -f "${RERANKER_DIR}/${RERANKER_MODEL_NAME}" ]; then
    echo ">>> 已存在 ${RERANKER_MODEL_NAME}，跳过下载"
else
    echo ">>> 下载 ${RERANKER_MODEL_NAME}"
    wget -c -O "${RERANKER_DIR}/${RERANKER_MODEL_NAME}" \
        "${RERANKER_BASE}/${RERANKER_MODEL_NAME}"
fi

# ========== 4. 下载 projector ==========
if [ -f "${RERANKER_DIR}/${RERANKER_PROJECTOR_NAME}" ]; then
    echo ">>> 已存在 ${RERANKER_PROJECTOR_NAME}，跳过下载"
else
    echo ">>> 下载 ${RERANKER_PROJECTOR_NAME}"
    wget -c -O "${RERANKER_DIR}/${RERANKER_PROJECTOR_NAME}" \
        "${RERANKER_BASE}/${RERANKER_PROJECTOR_NAME}"
fi

# ========== 5. 完成 ==========
echo ""
echo ">>> 全部完成。模型目录结构："
ls -lh "${MODELS_DIR}"
echo "---"
ls -lh "${RERANKER_DIR}"