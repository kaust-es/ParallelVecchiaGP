
# Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
# All rights reserved.
# VecchiaGP is a software package, provided by King Abdullah University of Science and Technology (KAUST).

# @file CMakeLists.txt
# @brief Defines an executables and links them with the VecchiaGP library and other libraries.
# @version 1.0.0
# @author Mahmoud ElKarargy
# @author Sohaila Khaled
# @author Qilong Pan
# @date 2025-09-29

# Define the target executable
add_executable(Example_Parallel_Vecchia_Estimation ${CMAKE_CURRENT_SOURCE_DIR}/ParallelVecchiaEstimation.cpp)
add_executable(Example_Parallel_Vecchia_Estimation_Prediction ${CMAKE_CURRENT_SOURCE_DIR}/ParallelVecchiaEstimationPrediction.cpp)

# Link the target executable with the project and any additional libraries
target_link_libraries(Example_Parallel_Vecchia_Estimation PUBLIC ${PROJECT_NAME}_INTERFACE)
target_link_libraries(Example_Parallel_Vecchia_Estimation_Prediction PUBLIC ${PROJECT_NAME}_INTERFACE)
