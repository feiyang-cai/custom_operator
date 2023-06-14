cmake_minimum_required(VERSION 3.5)
project(custom_dynamics)

set(CMAKE_CXX_STANDARD 17)

# Set the path to ONNX Runtime
set(ONNXRUNTIME_ROOT "/home/feiyang/Packages/onnxruntime")

# Include ONNX Runtime headers
include_directories(${ONNXRUNTIME_ROOT}/include/onnxruntime/)
include_directories(${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
include_directories(${ONNXRUNTIME_ROOT}/build/Linux/RelWithDebInfo/_deps/gsl-src/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# Add your custom op source file
add_library(custom_dynamics SHARED custom_dynamics_library.cc custom_dynamics_library.h)
