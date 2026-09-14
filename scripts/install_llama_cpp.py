#!/usr/bin/env python3
"""为当前环境安装匹配本机 GPU 的 llama-cpp-python 预编译 wheel。

llama-cpp-python 官方在 GitHub release 里为每个版本发布了多种后端变体的
预编译 wheel (CUDA 各版本 / Vulkan / Metal / ROCm), 但 PyPI 上只有需要
现场编译的源码包。本脚本:

  1. 探测本机后端 (NVIDIA CUDA / Apple Metal / AMD ROCm / 其它);
  2. 扫描该版本在 GitHub release 的所有变体 tag 与 assets;
  3. 按 "平台 wheel tag 匹配 + 后端优先级 + CUDA 版本接近度" 选出最佳 wheel;
  4. 用 `uv pip install <wheel-url>` 安装;
  5. 找不到任何预编译 wheel 时, 回退到 PyPI 源码编译 (需要本机有 nvcc/工具链)。

只依赖标准库。设计为可反复运行 (幂等) 与 --dry-run 预览。

用法:
    python scripts/install_llama_cpp.py                 # 自动探测, 装最佳 wheel
    python scripts/install_llama_cpp.py --dry-run       # 只打印将执行的命令
    python scripts/install_llama_cpp.py --version 0.3.9 # 指定版本
    python scripts/install_llama_cpp.py --backend cpu   # 强制后端 (跳过 GPU)
    python scripts/install_llama_cpp.py --no-wheels     # 直接走源码编译回退

环境变量:
    GITHUB_TOKEN   有则带上, 提高 API 速率限制 (匿名 60 次/小时)。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

REPO = "abetlen/llama-cpp-python"
API = f"https://api.github.com/repos/{REPO}"

# 各后端的变体 tag 后缀 (v{version}{suffix})。顺序不代表优先级 —— 优先级由
# pick_wheel 里的打分决定 (例如 CUDA 变体要按与驱动版本的接近度排序)。
CUDA_SUFFIX_RE = re.compile(r"cu(\d{3})$")  # -cu132 / -cu124 ...


def detect_cuda_version() -> tuple[int, int] | None:
    """返回 NVIDIA 驱动支持的最高 CUDA runtime 版本 (major, minor), 无则 None。

    解析 `nvidia-smi` 输出右上角的 "CUDA Version: 13.0"。这是驱动能跑的上限,
    用来过滤掉需要更高驱动的 cu 变体。
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return None
    m = re.search(r"CUDA\s*(UMD)?\s*Version:\s*(\d+)\.(\d+)", out)
    if m:
        return int(m.group(2)), int(m.group(3))
    return None


def detect_backend() -> tuple[str, tuple[int, int] | None]:
    """探测本机后端: ('cuda'|'metal'|'rocm'|'cpu', cuda_version_or_None)。"""
    cuda = detect_cuda_version()
    if cuda is not None:
        return "cuda", cuda
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "metal", None
    if system == "Linux" and (os.path.isdir("/opt/rocm") or shutil.which("rocminfo")):
        return "rocm", None
    return "cpu", None


def wheel_platform_ok(name: str) -> bool:
    """判断一个 wheel 文件名是否匹配当前 OS/架构。用子串匹配, 足够鲁棒。"""
    system = platform.system()
    machine = platform.machine().lower()
    n = name.lower()
    if not n.endswith(".whl"):
        return False
    if system == "Windows":
        return "win_amd64" in n or ("win_arm64" in n and machine == "arm64")
    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "macosx" in n and "arm64" in n
        return "macosx" in n and "x86_64" in n
    # Linux / 其它 Unix
    arch = "aarch64" if machine in ("arm64", "aarch64") else machine
    return ("manylinux" in n or "linux" in n) and arch in n



def fetch_release_tags(version: str) -> list[dict]:
    """获取某版本的所有变体 tag (v{version} / v{version}-cu132 / -metal ...)。

    返回: [{"ref": "refs/tags/v0.3.35-cu132", ...}, ...]。
    """
    url = f"{API}/git/matching-refs/tags/v{version}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise SystemExit(f"GitHub API 错误 {e.code}: {e.read().decode()}")
    except Exception as e:
        raise SystemExit(f"无法访问 GitHub API: {e}")


def fetch_release_assets(tag: str) -> list[dict]:
    """获取某个 tag 的所有 assets (预编译 wheel / 源码包)。

    返回: [{"name": "llama_cpp_python-0.3.35-...-win_amd64.whl",
            "browser_download_url": "https://...", "size": 123456}, ...]。
    """
    url = f"{API}/releases/tags/{tag}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("assets", [])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        print(f"警告: tag {tag} 的 release 不存在或访问失败 ({e.code})", file=sys.stderr)
        return []
    except Exception as e:
        print(f"警告: 无法获取 {tag} 的 assets: {e}", file=sys.stderr)
        return []


def score_variant(tag_suffix: str, backend: str, cuda_ver: tuple[int, int] | None) -> int:
    """为变体 tag 后缀打分: 分数越高越优先。

    策略:
      - CUDA: 版本越接近驱动上限越好 (cu132 > cu130 > cu124 ...)
      - Metal / ROCm / Vulkan: 固定分数
      - 无后缀 (v{version}): CPU fallback, 最低分
    """
    if not tag_suffix:
        return 0  # CPU 或者 base build
    if backend == "cuda" and cuda_ver:
        m = CUDA_SUFFIX_RE.search(tag_suffix)
        if m:
            # cu132 -> 1302, cu124 -> 1204 ...
            cu_code = int(m.group(1))
            cu_major, cu_minor = cu_code // 10, cu_code % 10
            driver_major, driver_minor = cuda_ver
            driver_code = driver_major * 100 + driver_minor * 10
            # 分数 = 10000 - |驱动 - cu 变体| (越接近越高)
            return 10000 - abs(driver_code - cu_code)
        return 0
    if backend == "metal" and tag_suffix == "-metal":
        return 9000
    if backend == "rocm" and ("rocm" in tag_suffix or "hip" in tag_suffix):
        return 8000
    if "vulkan" in tag_suffix:
        return 7000  # 通用后备, 分数低于原生 GPU 后端
    return 0


def pick_wheel(version: str, backend: str, cuda_ver: tuple[int, int] | None) -> str | None:
    """为本机选出最佳 wheel URL, 找不到返回 None。"""
    tags_data = fetch_release_tags(version)
    if not tags_data:
        print(f"GitHub 上找不到版本 v{version} 的任何 release tag。", file=sys.stderr)
        return None

    candidates = []
    for tag_info in tags_data:
        tag = tag_info["ref"].replace("refs/tags/", "")
        suffix = tag.replace(f"v{version}", "")  # "" / "-cu132" / "-metal" ...
        assets = fetch_release_assets(tag)
        for asset in assets:
            name = asset["name"]
            if not wheel_platform_ok(name):
                continue
            score = score_variant(suffix, backend, cuda_ver)
            candidates.append((score, asset["browser_download_url"], name, tag))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_url, best_name, best_tag = candidates[0]
    print(f"选中: {best_name} (tag={best_tag}, score={best_score})")
    return best_url


def install_wheel(url: str, dry_run: bool):
    """用 uv pip install <url> 安装 wheel。"""
    cmd = ["uv", "pip", "install", url]
    print(f"运行: {' '.join(cmd)}")
    if dry_run:
        print("(--dry-run, 不实际执行)")
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"安装失败, 退出码 {result.returncode}")


def fallback_source_install(dry_run: bool):
    """回退到 PyPI 源码编译 (CMAKE_ARGS + uv pip install llama-cpp-python)。"""
    print("\n未找到预编译 wheel, 回退 PyPI 源码编译 (需要本机有 nvcc 或等价工具链)。")
    cmake_args = os.getenv("CMAKE_ARGS", "")
    backend, cuda = detect_backend()
    if backend == "cuda" and not cmake_args:
        cmake_args = "-DGGML_CUDA=on"
        print(f"探测到 CUDA {cuda[0]}.{cuda[1]}, 默认 CMAKE_ARGS={cmake_args!r}")
    elif backend == "metal" and not cmake_args:
        cmake_args = "-DGGML_METAL=on"
        print(f"探测到 Metal, 默认 CMAKE_ARGS={cmake_args!r}")
    elif backend == "rocm" and not cmake_args:
        cmake_args = "-DGGML_HIPBLAS=on"
        print(f"探测到 ROCm, 默认 CMAKE_ARGS={cmake_args!r}")

    env = os.environ.copy()
    if cmake_args:
        env["CMAKE_ARGS"] = cmake_args

    cmd = ["uv", "pip", "install", "llama-cpp-python"]
    print(f"运行: CMAKE_ARGS={cmake_args!r} {' '.join(cmd)}")
    if dry_run:
        print("(--dry-run, 不实际执行)")
        return
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(f"源码编译失败, 退出码 {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="0.3.35", help="llama-cpp-python 版本 (默认 0.3.35)")
    parser.add_argument("--backend", choices=["cuda", "metal", "rocm", "cpu"], help="强制后端, 跳过自动探测")
    parser.add_argument("--no-wheels", action="store_true", help="跳过 GitHub wheel 扫描, 直接走源码编译")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令, 不实际安装")
    args = parser.parse_args()

    backend, cuda_ver = (args.backend, None) if args.backend else detect_backend()
    print(f"后端: {backend}", end="")
    if backend == "cuda" and cuda_ver:
        print(f" (驱动支持 CUDA {cuda_ver[0]}.{cuda_ver[1]})")
    else:
        print()

    if args.no_wheels:
        fallback_source_install(args.dry_run)
        return

    wheel_url = pick_wheel(args.version, backend, cuda_ver)
    if wheel_url:
        install_wheel(wheel_url, args.dry_run)
    else:
        fallback_source_install(args.dry_run)


if __name__ == "__main__":
    main()
