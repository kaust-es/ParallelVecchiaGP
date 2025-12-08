# FindMAGMA.cmake
#
# This will define:
#   MAGMA_FOUND
#   MAGMA_INCLUDE_DIRS
#   MAGMA_LIBRARIES
#   MAGMA_VERSION
#   MAGMA_DEFINITIONS
#
# Also defines imported target:
#   MAGMA::MAGMA

# Try pkg-config first
find_package(PkgConfig QUIET)
pkg_check_modules(PC_MAGMA magma QUIET)

# Apply pkg-config results if found
if (PC_MAGMA_FOUND)
    set(MAGMA_DEFINITIONS ${PC_MAGMA_CFLAGS_OTHER})
    set(MAGMA_INCLUDE_DIRS ${PC_MAGMA_INCLUDEDIR})
    set(MAGMA_LIBRARY_DIRS ${PC_MAGMA_LIBDIR})
    set(MAGMA_LIBRARIES ${PC_MAGMA_LIBRARIES})
    set(MAGMA_VERSION ${PC_MAGMA_VERSION})
    # Filter out magma_sparse and CUDA libraries from MAGMA_LIBRARIES
    set(FILTERED_MAGMA_LIBS "")
    foreach(lib ${MAGMA_LIBRARIES})
        # Check if it's a CUDA library or magma_sparse (handle various formats: -lmagma_sparse, magma_sparse, or full path)
        if(NOT lib MATCHES "(cudart|cublas|cusparse|curand|cufft|cusolver|magma_sparse)")
            list(APPEND FILTERED_MAGMA_LIBS ${lib})
        endif()
    endforeach()
    set(MAGMA_LIBRARIES ${FILTERED_MAGMA_LIBS})
    message(STATUS "MAGMA FROM PKGConfig}")
    message(STATUS "PC_MAGMA_LIBDIR: ${PC_MAGMA_LIBDIR}")
    message(STATUS "PC_MAGMA_LIBRARY_DIRS: ${PC_MAGMA_LIBRARY_DIRS}")
endif()

# Fallback if pkg-config failed
if (NOT PC_MAGMA_FOUND)

    # Allow override by user
    set(MAGMA_ROOT
        $ENV{MAGMA_DIR}
        ${CMAKE_INSTALL_PREFIX}/MAGMA
        CACHE PATH "Root directory of MAGMA installation"
    )

    # Find headers manually
    find_path(MAGMA_INCLUDE_DIR
        NAMES magma.h
        PATHS ${MAGMA_ROOT}/include
    )

    # Find libraries manually
    find_library(MAGMA_LIBRARY
        NAMES magma
        PATHS ${MAGMA_ROOT}/lib ${MAGMA_ROOT}/lib64
    )

    find_library(MAGMA_CUDA_LIBRARY
        NAMES magma_cuda
        PATHS ${MAGMA_ROOT}/lib ${MAGMA_ROOT}/lib64
    )

    # Compose include/libraries variables to match pkg-config interface
    set(MAGMA_INCLUDE_DIRS ${MAGMA_INCLUDE_DIR})
    set(MAGMA_LIBRARIES ${MAGMA_LIBRARY})
    if(MAGMA_CUDA_LIBRARY)
        list(APPEND MAGMA_LIBRARIES ${MAGMA_CUDA_LIBRARY})
    endif()

    # Try to extract version
    if(MAGMA_INCLUDE_DIR)
        file(STRINGS "${MAGMA_INCLUDE_DIR}/magma.h" magma_version_line REGEX "#define MAGMA_VERSION_STRING")
        string(REGEX REPLACE "#define MAGMA_VERSION_STRING \"([^\"]+)\"" "\\1" MAGMA_VERSION "${magma_version_line}")
    endif()

endif()

message(STATUS "MAGMA_INCLUDE_DIRS: ${MAGMA_INCLUDE_DIRS}")
message(STATUS "MAGMA_LIBRARY_DIRS: ${MAGMA_LIBRARY_DIRS}")
message(STATUS "MAGMA_LIBRARIES: ${MAGMA_LIBRARIES}")
message(STATUS "MAGMA_VERSION: ${MAGMA_VERSION}")
message(STATUS "MAGMA_DEFINITIONS: ${MAGMA_DEFINITIONS}")

# Final check
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MAGMA
    REQUIRED_VARS MAGMA_INCLUDE_DIRS MAGMA_LIBRARIES
    VERSION_VAR MAGMA_VERSION
)

message(STATUS "MAGMA_FOUND: ${MAGMA_FOUND}")
# Imported target interface (optional, modern CMake usage)
if(MAGMA_FOUND AND NOT TARGET MAGMA::MAGMA)
    add_library(MAGMA::MAGMA INTERFACE IMPORTED)
    target_include_directories(MAGMA::MAGMA INTERFACE ${MAGMA_INCLUDE_DIRS})
    target_link_libraries(MAGMA::MAGMA INTERFACE ${MAGMA_LIBRARIES})
    if(MAGMA_DEFINITIONS)
        target_compile_options(MAGMA::MAGMA INTERFACE ${MAGMA_DEFINITIONS})
    endif()
endif()

# Hide internal variables from cache
mark_as_advanced(
    MAGMA_INCLUDE_DIR
    MAGMA_INCLUDE_DIRS
    MAGMA_LIBRARY
    MAGMA_CUDA_LIBRARY
    MAGMA_LIBRARIES
    MAGMA_DEFINITIONS
)