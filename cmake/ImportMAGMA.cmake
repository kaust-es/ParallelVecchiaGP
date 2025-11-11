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
# 'tag' defines "v2.7.2" as the version tag of MAGMA, indicating the specific release to be utilized.
set(tag "v2.7.2")
# 'version' specifies "2.7.2" as the version of the MAGMA library, ensuring compatibility with the project's requirements.
set(version "2.7.2")
# 'flag' is intended for additional configuration options during the build process. A space is placed as a placeholder.
set(flag "-DGPU_TARGET=Volta")
# 'is_cmake' indicates that MAGMA uses CMake for its build system, which is set to ON.
set(is_cmake ON)
# 'is_git' denotes that the MAGMA source code is hosted in a Git repository, which is set to ON.
set(is_git OFF)
# 'auto_gen' signals whether autogen scripts are required for the build process, which is set to OFF for NLOPT.
set(auto_gen OFF)
# 'url' provides the location of the MAGMA source code repository on GitHub.
set(url "https://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-2.7.2.tar.gz")

# The 'ImportDependency' macro script, located in the 'macros' directory, is included for managing the import and setup of the MAGMA library.
include(macros/ImportDependency)
# The 'ImportDependency' macro is invoked with the above-defined parameters to handle the detection, fetching, and integration of MAGMA into the project.
ImportDependency(${name} ${tag} ${version} ${url} "${flag}" "" ${is_cmake} ${is_git} ${auto_gen})

# A status message is outputted to indicate the successful integration of the MAGMA library into the project.
message(STATUS "${name} done")

