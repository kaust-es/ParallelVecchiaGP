
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief This file contains the CMake configuration for the Vecchia library.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author David Helmy
# @date 2025-09-29

# Add subdirectories for configurations, data-generators, data-units, and linear-algebra-solvers.
add_subdirectory(api)
add_subdirectory(configurations)
add_subdirectory(helpers)
add_subdirectory(cuda-kernels)
add_subdirectory(kernels)
add_subdirectory(data-units)
add_subdirectory(data-generators)
add_subdirectory(data-loader)
add_subdirectory(data-clustering)
add_subdirectory(conditioning-updater)
add_subdirectory(hardware)
add_subdirectory(estimators)
add_subdirectory(predictors)

if (USE_R)
    add_subdirectory(Rcpp-adapters)
endif ()

# Set the name of the library to be created.
set(LIB_NAME ${PROJECT_NAME})
# Create the library with the specified source files and linking libraries.
add_library(${LIB_NAME} SHARED ${SOURCES})

target_compile_definitions(${LIB_NAME} PUBLIC ${COMPILE_DEFINITIONS})
target_link_libraries(${LIB_NAME} ${LIBS})

# Set the version of the library.
set_target_properties(${LIB_NAME}
        PROPERTIES
        VERSION ${${PROJECT_NAME}_VERSION}
        )

# Set the installation directories for the library.
target_include_directories(${LIB_NAME}
        PUBLIC
        "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>"
        )
install(TARGETS ${LIB_NAME}
        EXPORT ${LIB_NAME}CoreConfig
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/VECCHIA/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/VECCHIA/lib
        )

install(EXPORT ${LIB_NAME}CoreConfig DESTINATION ${CMAKE_INSTALL_PREFIX}/VECCHIA/lib/cmake/)