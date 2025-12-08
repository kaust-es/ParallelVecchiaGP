# Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.0.0
# @brief CMake build script for Rcpp adapters
# @author Generated for R wrapper
# @date 2025-01-01

if (USE_R)
    set(SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/FunctionsAdapter.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/RcppExports.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/RcppModules.cpp
            ${SOURCES}
            PARENT_SCOPE
            )
endif()

