#!/usr/bin/env bash
set -e # 遇到错误立即停止

# ================= 配置区域 =================
# C++ 源码所在的绝对路径
SOURCE_DIR="$(dirname "$0")/../cuPDLPx/pycupdlpx"
# 构建过程的中间目录
BUILD_DIR="$(dirname "$0")/build_cache_cupdlpx"

# ================= CUDA 环境设置 =================
# 1. 智能查找 CUDA Toolkit (优先 12.5，跳过损坏的 12.4)
CUDA_HOME="" # 初始化
if [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME="/usr/local/cuda-12.4"
elif [ -d "/usr/local/cuda-12.5" ]; then
    export CUDA_HOME="/usr/local/cuda-12.5"
else
    echo "❌ 未找到可靠的 CUDA Toolkit (建议安装 12.5 或其他完整版)"
    echo "   跳过检测，尝试直接使用系统/mamba环境..."
fi

# 2. 修正 PATH (确保 nvcc 和 cicc 都能被找到)
if [ -n "$CUDA_HOME" ]; then
    echo "🔋 使用 CUDA: $CUDA_HOME"
    export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

# ================= Mamba 环境检查与路径获取 =================

# 检查是否在 conda/mamba 环境中运行 (CONDA_PREFIX 是关键)
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "❌ 错误: 未在 micromamba/conda 虚拟环境中运行。"
    echo "   请先激活你的环境，例如: micromamba activate <env_name>"
    exit 1
fi

# 获取当前激活的 micromamba 环境路径
MAMBA_ENV_ROOT="$CONDA_PREFIX"
PYTHON_EXEC="$MAMBA_ENV_ROOT/bin/python"
# 获取 site-packages 路径
SITE_PACKAGES=$($PYTHON_EXEC -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TARGET_LIB_DIR="$MAMBA_ENV_ROOT/lib"

echo "🚀 [Build] 开始构建 cuPDLPx..."
echo "🐍 Python Executable: $PYTHON_EXEC"
echo "📦 目标安装位置: $SITE_PACKAGES"


# ================= 极速 GPU 架构检测 =================
ARCH_LIST="89-real;86-real"
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -n "$COMPUTE_CAP" ]; then
        ARCH_LIST="${COMPUTE_CAP}-real"
        echo "🎮 检测到 GPU 算力: sm_${COMPUTE_CAP}"
    fi
fi

# ================= CMake 构建流程 =================
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 1. 生成构建文件
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DBUILD_CUDA=ON \
    -DBUILD_PYTHON=ON \
    -DPython_EXECUTABLE="$PYTHON_EXEC" \
    -DCMAKE_C_FLAGS="-O3 -fno-lto" \
    -DCMAKE_CXX_FLAGS="-O3 -fno-lto" \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH_LIST}" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE \
    -Dpybind11_LTO=OFF

# 2. 开始多线程编译
echo "🔨 正在编译 (使用 $(nproc) 线程)..."
cmake --build "$BUILD_DIR" --parallel "$(nproc)"

# ================= 安装与搬运 =================
echo "📥 正在安装..."

# 1. 搬运 Python 扩展 (.so) 到 site-packages
find "$BUILD_DIR" -type f -name 'pycupdlpx*.so*' -exec cp -v {} "$SITE_PACKAGES/" \;

# 2. 搬运依赖的动态库 (.so) 到 Mamba 环境的 lib 目录
mkdir -p "$TARGET_LIB_DIR"
find "$BUILD_DIR" -maxdepth 3 -type f \
    \( -name 'libcupdlpx*.so*' -o -name 'libcudalin.so*' -o -name 'libcupdlpx*.dylib*' \) \
    -exec cp -P {} "$TARGET_LIB_DIR" \;

echo "🎉 依赖库已放入 $TARGET_LIB_DIR，mamba环境激活后会自动加载。"

# ================= 验证 =================
echo "✅ 编译完成！验证 import..."
$PYTHON_EXEC -c "import pycupdlpx; print(f'成功加载: {pycupdlpx.__file__}')"