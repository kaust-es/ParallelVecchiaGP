
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief This file contains the CMake configuration for the hardware directory.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Abouzeid
# @author Sameh Abdulah
# @date 2025-09-29

# Add the Configurations.cpp file to the list of source files.
set(SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/VecchiaHardware.cpp
        ${SOURCES}
        PARENT_SCOPE
        )