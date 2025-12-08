# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file ImportKBLAS.cmake
# @brief Find and include KBLAS library as a dependency.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Khaled
# @date 2025-09-29

# Configuration settings for the integration of the NLOPT library
# 'name' is assigned to "NLOPT", serving as the identifier for this library within the script.
set(name "KBLAS")
# 'tag' defines "master" as the version tag of KBLAS, indicating the specific release to be utilized.
set(tag "master")
# 'flag' is intended for additional configuration options during the build process. A space is placed as a placeholder.
set(flag " ")
# 'is_cmake' indicates that KBLAS uses CMake for its build system, which is set to ON.
set(is_cmake ON)
# 'is_git' denotes that the KBLAS source code is hosted in a Git repository, which is set to ON.
set(is_git ON)
# 'auto_gen' signals whether autogen scripts are required for the build process, which is set to OFF for KBLAS.
set(auto_gen OFF)
# 'url' provides the location of the KBLAS source code repository on GitHub.
set(url "https://github.com/ecrc/kblas-gpu.git")

# The 'ImportDependency' macro script, located in the 'macros' directory, is included for managing the import and setup of the KBLAS library.
include(macros/ImportDependency)
# The 'ImportDependency' macro is invoked with the above-defined parameters to handle the detection, fetching, and integration of KBLAS into the project.
ImportDependency(${name} ${tag} "" ${url} "${flag}" "" ${is_cmake} ${is_git} ${auto_gen})

# A status message is outputted to indicate the successful integration of the KBLAS library into the project.
message(STATUS "${name} done")

