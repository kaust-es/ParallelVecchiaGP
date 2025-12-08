
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Abouzeid
# @author Qilong Pan
# @date 2025-09-29

set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/BasselFunction.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/BatchPreparationUtility.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ByteHandler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CommunicatorMPI.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DistanceCalculationHelpers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LocationSorter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CSVUtils.cpp
        ${SOURCES}
        PARENT_SCOPE
        )