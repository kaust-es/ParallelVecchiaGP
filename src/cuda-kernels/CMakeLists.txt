# Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief CMakeLists.txt file for CUDA kernels
# @version 1.0.0
# @author Mahmoud ElKarargy
# @date 2025-10-18

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gpukernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/scaled_block_kernels.cu
        ${SOURCES}
        PARENT_SCOPE
        )