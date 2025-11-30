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
# Note: Faster repository doesn't specify architectures (uses auto-detection)
# Keeping specific architectures for compatibility, but can be removed if needed
set(CUDA_ARCHITECTURES "35;50;72")

# Set CUDA optimization flags to match faster repository
# They use: -O3 with --compiler-options -uns --extended-lambda -allow-unsupported-compiler
# They also use: -ccbin $(CXX) to use the same C++ compiler
# They do NOT use: --use_fast_math or -march=native for CUDA
# Note: -ccbin is set automatically by CMake via CMAKE_CUDA_HOST_COMPILER
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options -uns --extended-lambda -allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --compiler-options -uns --extended-lambda -allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# Prefer system CUDA libraries when CUDA_HOME is set (matches faster Makefile approach)
# This ensures we use system CUDA libraries instead of Spack-installed ones
if(DEFINED ENV{CUDA_HOME})
    set(CUDA_ROOT $ENV{CUDA_HOME})
    message(STATUS "Using CUDA_HOME: ${CUDA_ROOT}")
    # Set CUDA paths to prefer system installation
    set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc")
    # Add system CUDA library path FIRST (so it's searched before Spack paths)
    link_directories("${CUDA_ROOT}/lib64")
    # Also check for lib (non-64-bit systems)
    if(EXISTS "${CUDA_ROOT}/lib")
        link_directories("${CUDA_ROOT}/lib")
    endif()
    # Set CUDAToolkit_ROOT to force CMake to use system CUDA
    set(CUDAToolkit_ROOT "${CUDA_ROOT}")
    set(ENV{CUDAToolkit_ROOT} "${CUDA_ROOT}")
endif()

# Find the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA host compiler to match the C++ compiler (matches faster repo: -ccbin $(CXX))
# This ensures CUDA uses the same compiler as the rest of the code (mpic++)
if(CMAKE_CXX_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()

# Add CUDA libraries to the global LIBS variable
# When CUDA_HOME is set, link directly with library names (matches Makefile: -lcudart -lcublas -lcusparse)
# This ensures the linker uses the system libraries from the -L paths we set above
if(DEFINED ENV{CUDA_HOME})
    # Link directly with library names, matching Makefile approach
    # The -L paths set above will ensure system libraries are found
    list(APPEND LIBS cudart cublas cusparse)
    message(STATUS "Linking CUDA libraries directly: cudart cublas cusparse (using system libraries from ${CUDA_ROOT})")
else()
    # Use CMake imported targets when CUDA_HOME is not set
    list(APPEND LIBS CUDA::cudart CUDA::cublas CUDA::cusparse)
endif()
