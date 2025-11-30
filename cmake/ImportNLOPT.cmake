
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file ImportNLOPT.cmake
# @brief Find and include NLOPT library as a dependency.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohayla Khaled
# @date 2025-09-29

# Configuration settings for the integration of the NLOPT library
# 'name' is assigned to "NLOPT", serving as the identifier for this library within the script.
set(name "NLOPT")
# 'tag' defines "v2.7.1" as the version tag of NLOPT, indicating the specific release to be utilized.
set(tag "v2.7.1")
# 'version' specifies "2.7.1" as the version of the NLOPT library, ensuring compatibility with the project's requirements.
set(version "2.7.1")
# 'flag' is intended for additional configuration options during the build process. A space is placed as a placeholder.
set(flag " ")
# 'is_cmake' indicates that NLOPT uses CMake for its build system, which is set to ON.
set(is_cmake ON)
# 'is_git' denotes that the NLOPT source code is hosted in a Git repository, which is set to ON.
set(is_git ON)
# 'auto_gen' signals whether autogen scripts are required for the build process, which is set to OFF for NLOPT.
set(auto_gen OFF)
# 'url' provides the location of the NLOPT source code repository on GitHub.
set(url "https://github.com/stevengj/nlopt")

# Prefer system NLOPT when NLOPT_ROOT is set (matches faster Makefile approach)
# Check for system NLOPT first before building dependencies
if(DEFINED ENV{NLOPT_ROOT})
    set(NLOPT_ROOT $ENV{NLOPT_ROOT})
    message(STATUS "Checking for system NLOPT at: ${NLOPT_ROOT}")
    
    # Try to find system NLOPT using FindNLOPT
    include(FindNLOPT)
    if(NLOPT_FOUND)
        message(STATUS "Using system NLOPT library")
        # Add system NLOPT to LIBS (matches faster Makefile: -lnlopt)
        list(APPEND LIBS ${NLOPT_LIBRARIES})
        # Add include and library directories
        if(NLOPT_INCLUDE_DIRS)
            include_directories(${NLOPT_INCLUDE_DIRS})
        endif()
        # Set NLOPT_FOUND flag to skip ImportDependency
        set(${name}_FOUND TRUE)
    endif()
endif()

# If system NLOPT was not found, use ImportDependency to build it
if(NOT ${name}_FOUND)
    # The 'ImportDependency' macro script, located in the 'macros' directory, is included for managing the import and setup of the NLOPT library.
    include(macros/ImportDependency)
    # The 'ImportDependency' macro is invoked with the above-defined parameters to handle the detection, fetching, and integration of NLOPT into the project.
    ImportDependency(${name} ${tag} ${version} ${url} "${flag}" "" ${is_cmake} ${is_git} ${auto_gen})
endif()

# A status message is outputted to indicate the successful integration of the NLOPT library into the project.
message(STATUS "${name} done")

