# Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief This is a CMakeLists.txt file for a C++ project VecchiaGB.
# The project is a parallel high performance unified framework for geographical statistics on manycore systems.
# The file sets up variables and finds dependencies required for the project.
# It also provides options to enable building tests, building examples, building documentation.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Khaled
# @author Qilong Pan
# @date 2025-09-29

# TODO: an issue when we run ./clean_build.sh and there's a change in the CMakeLists.txt file. it can't find cuda and I need to run ./configure again.
# Set the minimum CMake version required to 3.20
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Set project options
# MPI is now required (removed USE_MPI option)
option(BUILD_TESTS "Option to enable building tests" OFF)
option(BUILD_EXAMPLES "Option to enable building examples" ON)
option(USE_R "Enable the use of R and Rcpp in the project" OFF)

# Cmake Module Paths
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

message("")
message("---------------------------------------- CUDA")
# Enable CUDA and include CudaToolchain
add_definitions(-DUSE_CUDA=TRUE)
enable_language(CUDA)
include(toolchains/CudaToolchain)

# Project Name and Version
project(VecchiaGP VERSION 1.0.0)

# Show the current version of CMake.
message(STATUS "CMAKE VERSION: ${CMAKE_VERSION}")
# Enable C++ language
enable_language(CXX)
# Get the current path of the project.
add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/")

# Set kernels path.
add_definitions(-DKERNELS_PATH="${PROJECT_SOURCE_DIR}/inst/include/kernels/concrete/")

# VecchiaGP depends on MPI (required)
# -------------------------------
message("")
message("---------------------------------------- MPI")
# Enable MPI and include MPI
add_definitions(-DUSE_MPI=TRUE)
message(STATUS "Trying to find MPI")
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
list(APPEND LIBS ${MPI_LIBRARIES})

# VecchiaGP depends on GFortran
# -------------------------------
include(ImportGFortran)

# VecchiaGP depends on GSL
# -------------------------------
include(ImportGSL)

# VecchiaGP depends on NLOPT
# -------------------------------
include(ImportNLOPT)

# VecchiaGP depends on LAPACK/BLASPP
# -------------------------------
include(ImportBLASPP)
include(ImportLapack)

# VecchiaGP Linear Algebra Backends
# -------------------------------
include(ImportMAGMA)
include(ImportKBLAS)

# Include directories for VecchiaGB
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inst/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prerequisites)
set(MY_LOGGER_PATH ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DMY_LOGGER_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

# VecchiaGP depends on R
# -------------------------------
if (USE_R)
    message("")
    message("---------------------------------------- Rcpp")
    # Find R and Rcpp using FindR Module
    find_package(R REQUIRED)
    if (${R_FOUND})
        message(STATUS "Using R technology")
        list(APPEND LIBS R)
        add_definitions(-DUSE_R)
    endif ()
endif ()

# Add src Directory to expose added libraries
add_subdirectory(src)

# Creates a new INTERFACE library target named ${PROJECT_NAME}_INTERFACE.
# The INTERFACE keyword specifies that this library will not be built, but instead will only be used for its properties.
add_library(${PROJECT_NAME}_INTERFACE INTERFACE)
target_link_libraries(${PROJECT_NAME}_INTERFACE INTERFACE ${PROJECT_NAME})

# Add linker options to the target
target_link_options(${PROJECT_NAME}_INTERFACE INTERFACE "SHELL:-Wl,--whole-archive $<TARGET_FILE:${PROJECT_NAME}> -Wl,--no-whole-archive")

# Add tests if enabled
if (${BUILD_TESTS})
    message(STATUS "Building Tests")
    include(ImportCatch2)
    include(Catch)
    include(CTest)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp-tests)
    enable_testing()
endif ()

# Add examples if enabled
if (BUILD_EXAMPLES)
    message(STATUS "Building Examples is Enabled")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif ()

# VecchiaGP Dependence export messages for users
# -------------------------------
message(" \n \t ** Configurations of VecchiaGP and installation of dependence is done successfully ** ")
message("\t - Export the following line to avoid re-install dependencies each time. -")
message("\t ----------------------------------------------------------------------------------------------------------------------------------- ")
message("\t export PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/MAGMA/lib/pkgconfig:$PKG_CONFIG_PATH")
message("\t export PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/KBLAS/lib/pkgconfig:$PKG_CONFIG_PATH")
message("\t export PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/BLASPP/lib/pkgconfig:$PKG_CONFIG_PATH")
message("\t export PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/LAPACK/lib/pkgconfig:$PKG_CONFIG_PATH")
message("\t export PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/GSL/lib/pkgconfig:$PKG_CONFIG_PATH")
message("\t export PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/NLOPT/lib/pkgconfig:${CMAKE_INSTALL_PREFIX}/NLOPT/lib64/pkgconfig:$PKG_CONFIG_PATH")
message("\t ----------------------------------------------------------------------------------------------------------------------------------- \n")

