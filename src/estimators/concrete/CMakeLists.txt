
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Abouzeid
# @author Qilong Pan
# @date 2025-09-29

# Add source files to the parent scope
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ParallelEstimator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ParallelBlockEstimator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ScaledBlockEstimator.cpp
        ${SOURCES}
        PARENT_SCOPE
        )