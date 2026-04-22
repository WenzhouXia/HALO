#!/usr/bin/env bash
set -e  # 遇到错误立即停止

# ================= 配置区域 =================
# HPR-LP 源码绝对路径
SOURCE_DIR="$(dirname "$0")/../HPR-LP-C"
# ================= CUDA 环境设置 =================
# 1. 智能查找 CUDA Toolkit
CUDA_HOME="" # 初始化
if [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME="/usr/local/cuda-12.4"
elif [ -d "/usr/local/cuda-12.5" ]; then
    export CUDA_HOME="/usr/local/cuda-12.5"
else
    echo "❌ 未找到可靠的 CUDA Toolkit"
fi

# 2. 设置 CUDA 相关的环境变量
if [ -n "$CUDA_HOME" ]; then
    echo "🔋 使用 CUDA: $CUDA_HOME"
    export CUDA_PATH="$CUDA_HOME"
    # 添加 CUDA bin 路径到 PATH
    export PATH="${CUDA_HOME}/bin:${PATH}"
    # 添加 CUDA 库路径到 LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

# 3. 指定主机编译器 (通常在 mamba 环境中不需要，但保留以防万一)
export HOST_COMPILER="/usr/bin/g++"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"

# ================= Mamba 环境检查与路径获取 =================
# 检查是否在 micromamba/conda 环境中运行 (CONDA_PREFIX 是关键)
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "❌ 错误: 未在 micromamba/conda 虚拟环境中运行。"
    echo "   请先激活你的环境，例如: micromamba activate <env_name>"
    exit 1
fi

MAMBA_ENV_ROOT="$CONDA_PREFIX"
PYTHON_EXEC="$MAMBA_ENV_ROOT/bin/python"

echo "🚀 [Build HPR-LP] 开始构建..."
echo "🐍 Python Executable: $PYTHON_EXEC"


# ================= 极速 GPU 架构检测 =================
TARGET_SM="89"
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -n "$COMPUTE_CAP" ]; then
        TARGET_SM="$COMPUTE_CAP"
        echo "🎮 检测到 GPU 算力: sm_${TARGET_SM}"
    fi
fi

# ================= 编译 C++ 核心 (Make) =================
echo "📂 跳转到源码目录: $SOURCE_DIR"
cd "$SOURCE_DIR"

# echo "🧹 清理旧构建..."
# # 使用 `|| true` 确保即使 clean 失败脚本也不会立即停止
# make clean > /dev/null 2>&1 || true

# echo "🔨 正在编译 C++ 核心库 (make)..."
# make GPU_SM="$TARGET_SM" HOST_COMPILER="$CXX" -j$(nproc)

# echo "✅ C++ 核心库编译成功"
# # ================= 安装 Python 绑定 =================
# cd bindings/python
# <<<< 修改开始：增加 CMake 配置流程 >>>>
echo "🏗️  配置 CMake (构建目录: build_cpp)..."
# 如果构建目录存在且包含旧的缓存，先清理它
echo "🧹 清理旧构建缓存..."
if [ -d "build_cpp" ]; then
    rm -rf build_cpp
    echo "✅ 旧构建缓存已清理"
fi
mkdir -p build_cpp && cd build_cpp

# 显式指定 CUDA 路径和编译器
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="$TARGET_SM" \
    -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_C_COMPILER="$CC"

echo "🔨 正在编译 C++ 核心库..."
make -j$(nproc)
# <<<< 修改结束 >>>>
echo "✅ C++ 核心库编译成功"

# ================= 安装 Python 绑定 =================
# 注意：这里需要根据你 build_cpp 的位置调整返回路径
cd ../bindings/python


echo "🧹 清理 Python 临时文件..."
rm -rf build dist *.egg-info

echo "📥 正在使用 pip 安装 Python 绑定到 Mamba 环境..."
# 使用明确的 Python 解释器执行 pip
$PYTHON_EXEC -m pip install . -v

# ================= 自动配置运行时路径 (已移除) =================
# Conda/Micromamba 会自动处理 LD_LIBRARY_PATH，不需要手动修改 .env 文件

echo "🎉 HPR-LP 构建 & 安装全部完成！"

# ================= 验证 =================
echo "🧪 运行示例脚本进行验证..."
# 从 bindings/python 目录返回到源码根目录
cd ../../
EXAMPLE_SCRIPT="bindings/python/examples/example_direct_lp.py"
if [ -f "$EXAMPLE_SCRIPT" ]; then
    $PYTHON_EXEC "$EXAMPLE_SCRIPT"
    echo "✅ 示例运行成功！"
else
    echo "⚠️ 未找到示例脚本，跳过验证。"
fi