
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Abouzeid
# @author Qilong Pan
# @date 2025-09-29

# Include the concrete implementations of the VecchiaGB Descriptor class
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/concrete)

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/EstimatorFactory.cpp
        ${SOURCES}
        PARENT_SCOPE
        )