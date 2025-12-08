
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Abouzeid
# @author Qilong Pan
# @date 2025-09-29

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/Locations.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/VecchiaGBData.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Point.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/BatchMetadata.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/VecchiaMemoryBuffers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ClusterData.cpp
        ${SOURCES}
        PARENT_SCOPE
        )