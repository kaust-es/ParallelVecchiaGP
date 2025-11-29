# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CudaToolchain.cmake
# @brief This file is used to set up the CUDA toolchain for compilation.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Khaled
# @date 2025-09-29

# Set CUDA compilation options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA architectures to be targeted
set(CUDA_ARCHITECTURES "35;50;72")

# Set CUDA optimization flags (matching old code performance)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# Find the CUDA toolkit
find_package(CUDAToolkit REQUIRED)
set(ENV{LDFLAGS} "-L$ENV{CUDA_DIR}/lib64")

# Add CUDA libraries to the global LIBS variable
list(APPEND LIBS CUDA::cudart CUDA::cublas CUDA::cusparse)
