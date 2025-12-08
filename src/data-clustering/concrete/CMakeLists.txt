# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @version 1.0.0
# @brief CMake configuration for concrete clustering strategies
# @author Mahmoud ElKarargy
# @date 2025-10-20

set(CLUSTERING_CONCRETE_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/NoClusteringStrategy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LocalClusteringStrategy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DistributedClusteringStrategy.cpp
        PARENT_SCOPE
        )
