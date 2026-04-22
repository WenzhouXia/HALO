# make clean && make && cd bindings/python && python -m pip install . && cd ../.. && python bindings/python/examples/example_direct_lp.py
# make clean && CUDA_PATH=$CONDA_PREFIX HOST_COMPILER=$CONDA_PREFIX/bin/g++ GPU_SM=89 make && cd bindings/python && python -m pip install . -v && cd ../.. && python /root/OT/MGPD/external/HPR-LP-C/bindings/python/examples/example_direct_lp.py
#!/bin/bash
set -e  # <-- 关键：出错时立即退出

# --- 环境变量设置 ---
# CONDA_PREFIX 应该会自动从你的激活环境中获取
# 但我们为了保险起见，还是明确设置它
export CONDA_PREFIX=${CONDA_PREFIX:-"/root/miniconda3/envs/halo_env"}
export CUDA_PATH=$CONDA_PREFIX
export CUDA_HOME=$CONDA_PREFIX # CMake 经常查找这个

# 现在 .../bin/gcc 和 .../bin/g++ 都存在了
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# --- 设置结束 ---

echo "=== 正在使用 CUDA_PATH: $CUDA_PATH"
echo "=== 正在使用 CXX (C++ 编译器): $CXX"
echo "=== 正在使用 CC (C 编译器): $CC"

# 确保你在 HPR-LP-C 的根目录
# (你之前的日志显示你在 .../external/HPR-LP-C#)
# 如果你不在那里，请先 cd 过去

make clean

# make 会继承 $CUDA_PATH 和 $CXX (通过 HOST_COMPILER)
echo "=== [1/3] 正在编译 C++ 核心库 (make)... ==="
make GPU_SM=89 HOST_COMPILER=$CXX

echo "=== [1/3] C++ 核心库编译成功 ==="

cd bindings/python
rm -rf build dist *.egg-info # 清理旧的编译缓存
echo "=== [2/3] 正在编译和安装 Python 绑定 (pip install)... ==="
python -m pip install . -v

echo "=== [2/3] Python 绑定安装成功 ==="

cd ../..
echo "=== [3/3] 正在运行 Python 示例... ==="
python /root/OT/MGPD/external/HPR-LP-C/bindings/python/examples/example_direct_lp.py

echo "=== [3/3] Python 示例运行成功 ==="
echo ""
echo ">>> 构建和测试全部完成 <<<"