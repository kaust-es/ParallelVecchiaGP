# Copyright (c) 2025, King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGP is a software package, provided by King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief This file contains the CMake configuration for the kernels directory.
# @version 1.1.0
# @author Mahmoud ElKarargy
# @author Sameh Abdulah
# @date 2025-09-29

# Automatically add all kernels in the concrete directory.
file(GLOB ALL_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/concrete/*.cpp)

# Add the kernel.cpp file to the list with other kernels.
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/Kernel.cpp
        ${ALL_KERNELS}
        ${SOURCES}
        PARENT_SCOPE
        )