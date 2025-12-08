
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief CMake configuration file for Data Generators module
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Abouzeid
# @author Qilong Pan
# @date 2025-09-29

# Add source files to the parent scope
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/KnnConditioningUpdater.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SequentialConditioningUpdater.cpp
        ${SOURCES}
        PARENT_SCOPE
        )
