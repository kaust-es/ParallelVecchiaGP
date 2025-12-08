
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

    // Check if CSV files are provided for prediction-only mode
    // (when train/test locations and data paths are provided, but main data path is not)
    bool hasCSVFiles = !configurations.GetTrainLocationsPath().empty() && 
                       !configurations.GetTestLocationsPath().empty() &&
                       !configurations.GetTrainDataPath().empty() &&
                       !configurations.GetTestDataPath().empty();
    bool isPredictionOnlyMode = hasCSVFiles && configurations.GetDataPath().empty();
    
    // Load data by either read from file or create synthetic data.
    std::unique_ptr<VecchiaGBData<double>> data;
    
    // Always call VecchiaLoadData to ensure clustering happens if test locations are provided
    // This handles both normal mode and prediction-only mode
    // For Block Vecchia, test locations are clustered and stored in VecchiaGBData
    VecchiaGP<double>::VecchiaLoadData(configurations, data);
    
    // Run estimation only when we have proper training data (not in prediction-only mode)
    // In prediction-only mode, VecchiaLoadData may create synthetic data which isn't suitable for estimation
    // For Block Vecchia, skipping estimation doesn't affect the pre-clustered test locations stored in VecchiaGBData
    if (!isPredictionOnlyMode) {
        VecchiaGP<double>::VecchiaDataEstimation(configurations, data);
    }
    
    // Always run prediction (will use pre-clustered data from VecchiaLoadData if available)
    // For Block Vecchia, prediction uses the test clustering result stored during loadData
    VecchiaGP<double>::VecchiaPrediction(configurations, data, nullptr);

    return 0;
}   
