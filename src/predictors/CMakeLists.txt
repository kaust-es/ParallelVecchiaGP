# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief This file contains the CMake configuration for the predictors directory.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @date 2025-09-29

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/concrete)

# Add the PredictorFactory source file
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/PredictorFactory.cpp
        ${SOURCES}
        PARENT_SCOPE
        )



