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

# Project Name and Version
project(VecchiaGP VERSION 1.0.0 LANGUAGES C CXX)

# Show the current version of CMake.
message(STATUS "CMAKE VERSION: ${CMAKE_VERSION}")

message("")
message("---------------------------------------- CUDA")
# Enable CUDA and include CudaToolchain
add_definitions(-DUSE_CUDA=TRUE)
# Find CUDA compiler using environment variables and system paths
if(NOT CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDA_HOME})
        find_program(CMAKE_CUDA_COMPILER nvcc HINTS $ENV{CUDA_HOME}/bin NO_DEFAULT_PATH)
    endif()
    if(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDA_PATH})
        find_program(CMAKE_CUDA_COMPILER nvcc HINTS $ENV{CUDA_PATH}/bin NO_DEFAULT_PATH)
    endif()
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(CMAKE_CUDA_COMPILER nvcc)
    endif()
endif()
enable_language(CUDA)
include(toolchains/CudaToolchain)
# Get the current path of the project.
add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/")

# Set kernels path.
add_definitions(-DKERNELS_PATH="${PROJECT_SOURCE_DIR}/inst/include/kernels/concrete/")

# VecchiaGP depends on OpenMP (required)
# -------------------------------
message("")
message("---------------------------------------- OpenMP")
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    list(APPEND LIBS OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

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

# Add MPI compiler flags for optimization
if(MPI_CXX_COMPILE_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
endif()
if(MPI_CXX_LINK_FLAGS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
endif()

# VecchiaGP depends on GFortran
# -------------------------------
enable_language(Fortran)

include(ImportGFortran)

# VecchiaGP depends on GSL
# -------------------------------
include(ImportGSL)
# VecchiaGP depends on NLOPT
# -------------------------------
include(ImportNLOPT)

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

