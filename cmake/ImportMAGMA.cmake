# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file ImportMAGMA.cmake
# @brief Find and include MAGMA library as a dependency.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Khaled
# @date 2025-09-29

# Configuration settings for the integration of the NLOPT library
# 'name' is assigned to "NLOPT", serving as the identifier for this library within the script.
set(name "MAGMA")
# 'tag' defines the version tag of MAGMA to use.
# Using v2.9.0 (Jan 2025) for CUDA 12+ compatibility
set(tag "v2.9.0")
# 'version' specifies the version of the MAGMA library.
set(version "2.9.0")
# 'flag' is intended for additional configuration options during the build process.
set(flag "-DGPU_TARGET=Volta")
# 'is_cmake' indicates that MAGMA uses CMake for its build system, which is set to ON.
set(is_cmake ON)
# 'is_git' - Using GitHub repository
set(is_git ON)
# 'auto_gen' signals whether autogen scripts are required for the build process.
set(auto_gen OFF)
 # 'url' provides the location of the MAGMA GitHub repository.
set(url "https://github.com/icl-utk-edu/magma.git")

# Define MAGMA-specific hooks for BuildDependency

# Pre-build: Generate CMake files for git builds
macro(MAGMA_PreBuild srcpath flags)
    message(STATUS "MAGMA: Running make generate for git build")
    string(REGEX MATCH "GPU_TARGET=([A-Za-z0-9_,]+)" _ "${flags}")
    set(gpu "${CMAKE_MATCH_1}")
    if(NOT gpu)
        set(gpu "Volta")
    endif()
    file(WRITE "${srcpath}/make.inc" "BACKEND = cuda\nFORT = true\nGPU_TARGET = ${gpu}\n")
    execute_process(COMMAND make generate WORKING_DIRECTORY ${srcpath} 
        RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
    if(result)
        message(FATAL_ERROR "MAGMA make generate failed")
    endif()
endmacro()

# Post-build: Fix version in pkgconfig file
macro(MAGMA_PostBuild installpath tag)
    set(pc "${installpath}/lib/pkgconfig/magma.pc")
    if(EXISTS "${pc}")
        string(REPLACE "v" "" ver "${tag}")
        file(READ "${pc}" content)
        string(REPLACE "Version: 0.0.0" "Version: ${ver}" content "${content}")
        file(WRITE "${pc}" "${content}")
        message(STATUS "MAGMA: Fixed pkgconfig version to ${ver}")
    endif()
endmacro()

# Use standard ImportDependency (will call our hooks automatically)
include(macros/ImportDependency)
ImportDependency(${name} ${tag} ${version} ${url} "${flag}" "" ${is_cmake} ${is_git} ${auto_gen})

message(STATUS "${name} done")
