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
set(flag "-DGPU_TARGET=Ampere")
# 'is_cmake' indicates that MAGMA uses CMake for its build system, which is set to ON.
set(is_cmake ON)
# 'is_git' - Using tarball (OFF) instead of git because git requires running 'make generate' first
set(is_git OFF)
# 'auto_gen' signals whether autogen scripts are required for the build process.
set(auto_gen OFF)
 # 'url' provides the location of the MAGMA source tarball (v2.9.0 release).
set(url "https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.9.0.tar.gz")

# The 'ImportDependency' macro script, located in the 'macros' directory, is included for managing the import and setup of the MAGMA library.
include(macros/ImportDependency)
# The 'ImportDependency' macro is invoked with the above-defined parameters to handle the detection, fetching, and integration of MAGMA into the project.
ImportDependency(${name} ${tag} ${version} ${url} "${flag}" "" ${is_cmake} ${is_git} ${auto_gen})

# A status message is outputted to indicate the successful integration of the MAGMA library into the project.
message(STATUS "${name} done")
