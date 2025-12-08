# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file ImportDependency.cmake
# @brief CMake script for importing and building external dependencies.
# @version 1.0.0    
# @author Mahmoud ElKarargy
# @author Amr Nasr
# @date 2023-12-28

# ImportDependency Macro:
# This macro is designed to check for the presence of an external dependency and install it if not found.
# It takes the following parameters:
# - raw_name: The name of the dependency.
# - tag: The tag of the dependency to fetch.
# - version: The version of the dependency.
# - url: The URL of the repository or source tarball.
# - flag: Additional flags to pass to the configure/make commands.
# - is_cmake: A boolean flag indicating whether the dependency uses CMake as its build system.
# - is_git: A boolean flag indicating whether the dependency is hosted on a git repository.
# - auto_gen: A boolean flag indicating whether to use autogen scripts or not.

# The macro checks whether the dependency is already included. If not, it attempts to find the package.
# If the package is found, it prints a message. If not, it calls the BuildDependency macro to fetch,
# configure, build, and install the dependency. Finally, it attempts to find the package again to validate the installation.

# Define a macro named ImportDependency for handling external dependencies. The macro checks for the dependency's presence and installs it if missing.
macro(ImportDependency name tag version url flag components is_cmake is_git auto_gen)

    # Check if the installation prefix is set to a system path (like /usr/) and warn the user about potential need for administrative privileges.
    if (CMAKE_INSTALL_PREFIX MATCHES "/usr/")
        message(WARNING "Installation path not specified. Please set the installation path using -DCMAKE_INSTALL_PREFIX=path/to/install or execute ./config.sh. Otherwise, please note that administrative privileges may be required to install in system paths.")
    endif ()

    # Convert the dependency name to uppercase for consistent messaging.
    string(TOUPPER ${name} capital_name)
    # Begin a section in the output to visually separate the handling of this dependency.
    message("")
    message("---------------------------------------- ${capital_name}")
    # Log the attempt to check for the specified version of the dependency.
    message(STATUS "Checking for ${capital_name} with Version ${version}")

    # Include the previously defined BuildDependency macro script for potential use.
    include(macros/BuildDependency)

    # If the dependency has not already been targeted for building in the current CMake process, proceed to check its presence.
    IF (NOT TARGET ${name})
        # Use the FindPkgConfig module to potentially use pkg-config for finding installed libraries.
        include(FindPkgConfig)
        find_package(PkgConfig QUIET)
        
        # Special handling for BLAS - check if it's already installed via LAPACK
        if(${name} STREQUAL "BLAS" AND EXISTS "${CMAKE_INSTALL_PREFIX}/_deps/LAPACK/lib/libopenblas.a")
            set(BLAS_LIBRARIES "${CMAKE_INSTALL_PREFIX}/_deps/LAPACK/lib/libopenblas.a")
            set(BLAS_FOUND TRUE)
            message("   Found ${capital_name} in LAPACK directory")
        else()
            # Attempt to find the specified version of the package quietly, without generating much output.
            find_package(${name} ${version} QUIET COMPONENTS ${components})
        endif()

        # If the package is found, output a message detailing the found configuration.
        if (${name}_FOUND)
            message("   Found ${capital_name}; ${${name}_DIR} ${${name}_LIBRARIES}")
        else ()
            # If the package is not found, notify and invoke BuildDependency to install it.
            message("   Can't find ${capital_name}, Installing it instead ..")
            BuildDependency(${name} ${url} ${tag} "${flag}" ${is_cmake} ${is_git} ${auto_gen})
            
            # Special handling for BLAS - set BLAS_LIBRARIES after building
            if(${name} STREQUAL "BLAS")
                # Check if BLAS was built in the expected locations (BuildDependency installs to ${CMAKE_INSTALL_PREFIX}/${capital_name})
                string(TOLOWER ${name} lower_name)
                if(EXISTS "${CMAKE_INSTALL_PREFIX}/${capital_name}/lib/libopenblas.a")
                    set(BLAS_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${capital_name}/lib/libopenblas.a")
                    set(BLAS_FOUND TRUE)
                    set(${name}_FOUND TRUE)
                    include_directories("${CMAKE_INSTALL_PREFIX}/${capital_name}/include")
                    link_directories("${CMAKE_INSTALL_PREFIX}/${capital_name}/lib")
                    message(STATUS "Set BLAS_LIBRARIES after build: ${BLAS_LIBRARIES}")
                elseif(EXISTS "${CMAKE_INSTALL_PREFIX}/${capital_name}/lib64/libopenblas.a")
                    set(BLAS_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${capital_name}/lib64/libopenblas.a")
                    set(BLAS_FOUND TRUE)
                    set(${name}_FOUND TRUE)
                    include_directories("${CMAKE_INSTALL_PREFIX}/${capital_name}/include")
                    link_directories("${CMAKE_INSTALL_PREFIX}/${capital_name}/lib64")
                    message(STATUS "Set BLAS_LIBRARIES after build: ${BLAS_LIBRARIES}")
                else()
                    # After attempting installation, forcibly attempt to find the package again, this time requiring its presence.
                    find_package(${name} ${version} REQUIRED COMPONENTS ${components})
                endif()
            else()
                # After attempting installation, forcibly attempt to find the package again, this time requiring its presence.
                find_package(${name} ${version} REQUIRED COMPONENTS ${components})
            endif()
        endif ()
    else ()
        # If the dependency target already exists, log that it's already been included.
        message(STATUS "${capital_name} already included")
    endif ()

    # Setup link and include directories based on the found or installed package configuration.
    # Add the dependency's library directories to the link directories for the current CMake target.
    if(${name}_LIBRARY_DIRS_DEP)
        link_directories(${${name}_LIBRARY_DIRS_DEP})
    endif()
    if(${name}_LIBRARY_DIRS)
        link_directories(${${name}_LIBRARY_DIRS})
    endif()
    # Note: ${name}_LIBRARIES contains library names, not directories, so don't use it with link_directories()
    # Add the dependency's include directories to the include path.
    if(${name}_INCLUDE_DIRS)
        include_directories(${${name}_INCLUDE_DIRS})
    endif()
    if(${name}_INCLUDE_DIRS_DEP)
        include_directories(AFTER ${${name}_INCLUDE_DIRS_DEP})
    endif()

    # If the dependency is not GSL, append its libraries to the list of libraries to be linked against.
    if(NOT ${name} STREQUAL "GSL")
        # For MAGMA, filter out CUDA libraries and magma_sparse that are handled separately
        if(${name} STREQUAL "MAGMA")
            # Filter out CUDA libraries and magma_sparse from MAGMA_LIBRARIES
            set(FILTERED_MAGMA_LIBS "")
            foreach(lib ${${name}_LIBRARIES})
                # Check if it's a CUDA library or magma_sparse (handle various formats: -lmagma_sparse, magma_sparse, or full path)
                if(NOT lib MATCHES "(cudart|cublas|cusparse|curand|cufft|cusolver|magma_sparse)")
                    list(APPEND FILTERED_MAGMA_LIBS ${lib})
                endif()
            endforeach()
            list(APPEND LIBS ${FILTERED_MAGMA_LIBS})
        else()
            list(APPEND LIBS ${${name}_LIBRARIES})
        endif()
    endif()
    # Append any additional dependency libraries to the list of libraries.
    if(${name}_LIBRARIES_DEP)
        list(APPEND LIBS ${${name}_LIBRARIES_DEP})
    endif()

endmacro()
