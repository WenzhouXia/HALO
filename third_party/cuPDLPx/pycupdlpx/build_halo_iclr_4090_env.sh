#!/usr/bin/env bash
# build.sh for cuPDLPx — auto-detect GPU arch and build SASS-only
set -euo pipefail

# ==========================================
# ==========================================

# if [[ -d "/usr/local/cuda-12.4/lib64" ]]; then
#   export LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
# elif [[ -d "/usr/local/cuda/lib64" ]]; then
#   export LIBRARY_PATH="/usr/local/cuda/lib64:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
# fi

export CFLAGS="-fno-lto"
export CXXFLAGS="-fno-lto"

# ==========================================

usage() {
  cat <<'USAGE'
Usage:
  ./build.sh [release|debug|clean|test]

Env overrides (optional):
  ARCH_LIST="80-real;86-real;89-real;90-real"
  EXTRA_CMAKE_FLAGS="..."
  CUDA_HOME=/usr/local/cuda-12.4
USAGE
}

CMD="${1:-release}"
case "$CMD" in
  release|debug|clean|test) ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 1 ;;
esac

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "❌ 请先 'conda activate <your_env>' 再运行此脚本！" >&2
  exit 1
fi

ROOT_DIR="$(pwd)"
BUILD_DIR="$ROOT_DIR/build"
PYTHON="$(command -v python)"
SITE_PACKAGES="$($PYTHON - << 'PY'
import sysconfig
print(sysconfig.get_paths()["platlib"])
PY
)"

echo "📁 ROOT      : $ROOT_DIR"
echo "🏗  BUILD     : $BUILD_DIR"
echo "🐍 PYTHON    : $PYTHON"
echo "📦 SITE-PKG  : $SITE_PACKAGES"
echo "📦 CONDA     : ${CONDA_PREFIX}"
echo

if [[ "$CMD" == "clean" ]]; then
  echo "🧹 清理 $BUILD_DIR"
  rm -rf "$BUILD_DIR"
  exit 0
fi

CMAKE_BUILD_TYPE="Release"
C_FLAGS="-O3 -fno-lto"
CXX_FLAGS="-O3 -fno-lto"
if [[ "$CMD" == "debug" ]]; then
  CMAKE_BUILD_TYPE="Debug"
  C_FLAGS="-g -O0 -fno-lto"
  CXX_FLAGS="-g -O0 -fno-lto"
fi

map_cc_to_arch() {
  local cc="$1"
  case "$cc" in
    8.0|8.0*) echo "80-real" ;;  # A100
    8.6|8.6*) echo "86-real" ;;
    8.9|8.9*) echo "89-real" ;;  # 4090 / 4060Ti / Ada Lovelace
    9.0|9.0*) echo "90-real" ;;  # H100 / Hopper
    *)        echo "" ;;
  esac
}

if [[ -z "${ARCH_LIST:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    archs=()
    if [[ -n "$caps" ]]; then
      unique_caps=$(echo "$caps" | awk '!seen[$0]++')
      while read -r cc; do
        [[ -z "$cc" ]] && continue
        a=$(map_cc_to_arch "$cc" || true)
        [[ -n "$a" ]] && archs+=("$a")
      done <<< "$unique_caps"
    fi
    if [[ ${#archs[@]} -eq 0 ]]; then
      archs=("89-real")
      echo "⚠️  未能探测到 compute_cap，默认使用: ${archs[*]}"
    fi
    ARCH_LIST="$(IFS=';'; echo "${archs[*]}")"
  else
    ARCH_LIST="89-real"
    echo "⚠️  未找到 nvidia-smi，默认使用 ARCH_LIST=${ARCH_LIST}"
  fi
else
  echo "ℹ️  使用外部指定的 ARCH_LIST=${ARCH_LIST}"
fi

: "${EXTRA_CMAKE_FLAGS:=}"

echo "🔧 配置类型  : $CMAKE_BUILD_TYPE"
echo "🔧 GPU 架构  : ${ARCH_LIST}   (SASS-only, 无 PTX)"
echo "🔧 LTO 优化  : 已禁用 (避免 GCC 版本冲突)"
echo "🔧 额外参数  : ${EXTRA_CMAKE_FLAGS:-<none>}"
echo

if [[ -z "${CUDA_HOME:-}" ]] && [[ -d "/usr/local/cuda-12.4" ]]; then
    export CUDA_HOME="/usr/local/cuda-12.4"
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  echo "🧭 使用 CUDA_HOME: $CUDA_HOME"
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "🧰 nvcc: $(nvcc --version | head -n1)"
else
  echo "❌ 未找到 nvcc，请设置 CUDA_HOME 或 PATH！" >&2
  exit 1
fi
echo

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DBUILD_CUDA=ON \
  -DBUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PYTHON" \
  -DCMAKE_C_FLAGS="$C_FLAGS" \
  -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN:\$ORIGIN/../.." \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH_LIST}" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE \
  -Dpybind11_LTO=OFF \
  ${EXTRA_CMAKE_FLAGS}

cmake --build "$BUILD_DIR" --parallel "$(nproc)"

EXT_SO="$(find "$BUILD_DIR" -type f -name 'pycupdlpx*.so*' | head -n1 || true)"
if [[ -z "${EXT_SO}" ]]; then
  echo "❌ 未找到 Python 扩展模块 (pycupdlpx*.so)。"
  exit 1
fi

echo "🧽 清理旧的 pycupdlpx*.so"
rm -f "$SITE_PACKAGES"/pycupdlpx*.so 2>/dev/null || true

echo "📥 复制扩展模块：$EXT_SO → $SITE_PACKAGES/"
cp "$EXT_SO" "$SITE_PACKAGES/"

echo "📥 复制核心库到 $CONDA_PREFIX/lib"
mkdir -p "$CONDA_PREFIX/lib"
find "$BUILD_DIR" -maxdepth 3 -type f \
  \( -name 'libcupdlpx*.so*' -o -name 'libcudalin.so*' -o -name 'libcupdlpx*.dylib*' \) \
  -exec cp -P {} "$CONDA_PREFIX/lib" \; || true

echo
echo "🧪 测试 import："
"$PYTHON" - << 'PY'
import sys, importlib.util
print("sys.path[0] =", sys.path[0])
try:
    import pycupdlpx
    print("  ✓ pycupdlpx 位于:", pycupdlpx.__file__)
    print("  ✓ 符号列表:", [x for x in dir(pycupdlpx) if not x.startswith("_")][:10], "...")
except Exception as e:
    print("  ✗ import 失败：", e)
    raise
PY

if [[ "$CMD" == "test" ]]; then
  echo
  echo "▶ 运行测试脚本：python test_dual_ot_loadData.py"
  set -x
  "$PYTHON" "$ROOT_DIR/test_dual_ot_loadData.py"
  set +x
fi

echo
echo "✅ 构建 & 安装完成（$CMAKE_BUILD_TYPE）"