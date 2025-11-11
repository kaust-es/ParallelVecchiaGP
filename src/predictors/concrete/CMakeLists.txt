# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.0.0
# @author Mahmoud ElKarargy
# @date 2025-09-29

# Add source files to the parent scope
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ScaledBlockPredictor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ParallelBlockPredictor.cpp
        ${SOURCES}
        PARENT_SCOPE
        )

