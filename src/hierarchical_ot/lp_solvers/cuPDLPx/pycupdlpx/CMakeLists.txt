cmake_minimum_required(VERSION 3.20)
project(pycupdlpx LANGUAGES C CXX CUDA)

add_subdirectory(pybind11)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_language(CUDA)

set(CUPDLPX_DIR ${CMAKE_SOURCE_DIR}/../cupdlpx)

add_library(cupdlpx_core STATIC
  ${CUPDLPX_DIR}/io.c
  ${CUPDLPX_DIR}/utils.cu
  ${CUPDLPX_DIR}/solver.cu
  ${CUPDLPX_DIR}/cupdlpx.c
)
target_include_directories(cupdlpx_core PUBLIC ${CUPDLPX_DIR})
target_link_libraries(cupdlpx_core PUBLIC
  CUDA::cudart CUDA::cublas CUDA::cusparse ZLIB::ZLIB
)

pybind11_add_module(pycupdlpx
  cupdlpx_bindings.cpp
)
target_include_directories(pycupdlpx PRIVATE ${CUPDLPX_DIR})
target_link_libraries(pycupdlpx PRIVATE cupdlpx_core)

set_target_properties(pycupdlpx PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
