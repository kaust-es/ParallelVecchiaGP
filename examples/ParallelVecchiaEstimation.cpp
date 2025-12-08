
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file ParallelBlockVecchia.cpp
 * @brief This program tests the parallel block vecchia algorithm.
 * @details The program takes command line arguments to configure the data generation.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <api/VecchiaGP.hpp>
#include <hardware/VecchiaHardware.hpp>

using namespace vecchia::api;
using namespace vecchia::configurations;

/**
 * @brief Main entry point for the Data Generation & Parameter Estimation program.
 * @details This function either generates synthetic data using the VecchiaGB library, or reads an CSV file containing real data, and estimates parameters using MLE.
 * @param[in] argc The number of command line arguments.
 * @param[in] argv An array of command line argument strings.
 * @return An integer indicating the success or failure of the program. A return value of 0 indicates success, while any non-zero value indicates failure.
 *
 */
int main(int argc, char **argv) {

    // Create a new configurations object.
    Configurations configurations;
    // Initialize the arguments with the provided command line arguments
    configurations.InitializeArguments(argc, argv);
    
    // Initialize the Vecchia Hardware.
   auto hardware = VecchiaHardware(configurations.GetVecchiaType(), configurations.GetCoresNumber(),
                                    configurations.GetGPUsNumbers());

    // Load data by either read from file or create synthetic data.
    std::unique_ptr<VecchiaGBData<double>> data;
    VecchiaGP<double>::VecchiaLoadData(configurations, data);
    VecchiaGP<double>::VecchiaDataEstimation(configurations, data);
    
    return 0;
}   
