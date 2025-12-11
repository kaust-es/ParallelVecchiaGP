# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.

#[=======================================================================[.rst:
FindKBLAS
--------

Find the native KBLAS (Kernel BLAS) includes and libraries.

Imported Targets
^^^^^^^^^^^^^^^

If KBLAS is found, this module defines the following :prop_tgt:`IMPORTED`
targets::

 KBLAS::kblas      - The main KBLAS library

Result Variables
^^^^^^^^^^^^^^^

This module will set the following variables in your project::

 KBLAS_FOUND          - True if KBLAS found on the local system
 KBLAS_INCLUDE_DIRS   - Location of KBLAS header files
 KBLAS_LIBRARIES      - The KBLAS libraries
 KBLAS_VERSION        - The version of the discovered KBLAS install
 KBLAS_DEFINITIONS    - Additional compiler and linker flags for KBLAS

Hints
^^^^^

Set ``KBLAS_ROOT_DIR`` to a directory that contains a KBLAS installation.
#]=======================================================================]

# Try pkg-config first
find_package(PkgConfig QUIET)
pkg_check_modules(PC_KBLAS kblas QUIET)

# Apply pkg-config results if found
if (PC_KBLAS_FOUND)
    set(KBLAS_DEFINITIONS ${PC_KBLAS_CFLAGS_OTHER})
    set(KBLAS_INCLUDE_DIRS ${PC_KBLAS_INCLUDEDIR})
    set(KBLAS_LIBRARIES ${PC_KBLAS_LIBRARIES})
    set(KBLAS_VERSION ${PC_KBLAS_VERSION})
    message(STATUS "KBLAS FROM PKGConfig")
endif()

# Fallback if pkg-config failed
if (NOT PC_KBLAS_FOUND)

    # Allow override by user
    set(KBLAS_ROOT
        $ENV{KBLAS_DIR}
        ${CMAKE_INSTALL_PREFIX}/KBLAS
        CACHE PATH "Root directory of KBLAS installation"
    )

    # Check build directory first (where dependencies are installed via BuildDependency)
    # This takes precedence over user-specified paths
    set(KBLAS_BUILD_DEP_DIR "${CMAKE_BINARY_DIR}/_dep/KBLAS")
    
    # Find headers - check build directory first, then user-specified locations
    find_path(KBLAS_INCLUDE_DIR
        NAMES kblas.h
        PATHS ${KBLAS_BUILD_DEP_DIR}/include
              ${KBLAS_ROOT}/include
        NO_DEFAULT_PATH
    )
    
    # If not found in specific paths, try system paths
    if(NOT KBLAS_INCLUDE_DIR)
        find_path(KBLAS_INCLUDE_DIR
            NAMES kblas.h
            PATHS ${KBLAS_BUILD_DEP_DIR}/include
                  ${KBLAS_ROOT}/include
        )
    endif()

    # Find libraries - check build directory first, then user-specified locations
    find_library(KBLAS_LIBRARY
        NAMES kblasgpu kblas
        PATHS ${KBLAS_BUILD_DEP_DIR}/lib 
              ${KBLAS_BUILD_DEP_DIR}/lib64
              ${KBLAS_ROOT}/lib 
              ${KBLAS_ROOT}/lib64
        NO_DEFAULT_PATH
    )
    
    # If not found in specific paths, try system paths
    if(NOT KBLAS_LIBRARY)
        find_library(KBLAS_LIBRARY
            NAMES kblasgpu kblas
            PATHS ${KBLAS_BUILD_DEP_DIR}/lib 
                  ${KBLAS_BUILD_DEP_DIR}/lib64
                  ${KBLAS_ROOT}/lib 
                  ${KBLAS_ROOT}/lib64
        )
    endif()

    # Compose include/libraries variables to match pkg-config interface
    set(KBLAS_INCLUDE_DIRS ${KBLAS_INCLUDE_DIR})
    set(KBLAS_LIBRARIES ${KBLAS_LIBRARY})

    # Try to extract version from header file
    if(KBLAS_INCLUDE_DIR AND EXISTS "${KBLAS_INCLUDE_DIR}/kblas.h")
        file(STRINGS "${KBLAS_INCLUDE_DIR}/kblas.h" kblas_version_line REGEX "#define KBLAS_VERSION")
        if(kblas_version_line)
            string(REGEX REPLACE "#define KBLAS_VERSION \"([^\"]+)\"" "\\1" KBLAS_VERSION "${kblas_version_line}")
        endif()
    endif()

endif()

message(STATUS "KBLAS_INCLUDE_DIRS: ${KBLAS_INCLUDE_DIRS}")
message(STATUS "KBLAS_LIBRARIES: ${KBLAS_LIBRARIES}")
message(STATUS "KBLAS_VERSION: ${KBLAS_VERSION}")
message(STATUS "KBLAS_DEFINITIONS: ${KBLAS_DEFINITIONS}")

# Final check
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KBLAS
    REQUIRED_VARS KBLAS_INCLUDE_DIRS KBLAS_LIBRARIES
    VERSION_VAR KBLAS_VERSION
)

message(STATUS "KBLAS_FOUND: ${KBLAS_FOUND}")

# Imported target interface (optional, modern CMake usage)
if(KBLAS_FOUND AND NOT TARGET KBLAS::KBLAS)
    add_library(KBLAS::KBLAS INTERFACE IMPORTED)
    target_include_directories(KBLAS::KBLAS INTERFACE ${KBLAS_INCLUDE_DIRS})
    target_link_libraries(KBLAS::KBLAS INTERFACE ${KBLAS_LIBRARIES})
    if(KBLAS_DEFINITIONS)
        target_compile_options(KBLAS::KBLAS INTERFACE ${KBLAS_DEFINITIONS})
    endif()
endif()

# Hide internal variables from cache
mark_as_advanced(
    KBLAS_INCLUDE_DIR
    KBLAS_INCLUDE_DIRS
    KBLAS_LIBRARY
    KBLAS_LIBRARIES
    KBLAS_DEFINITIONS
)