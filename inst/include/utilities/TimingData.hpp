// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file TimingData.hpp
 * @brief Structure to hold timing information for logging
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-01-29
 **/

#ifndef VECCHIAGP_TIMING_DATA_HPP
#define VECCHIAGP_TIMING_DATA_HPP

namespace vecchia {
namespace utilities {

/**
 * @struct TimingData
 * @brief Holds timing information for various stages of computation
 */
struct TimingData {
    // Preprocessing/Clustering timings
    double RAC_partitioning = 0.0;
    double centers_of_gravity_calculation = 0.0;
    double send_centers_of_gravity = 0.0;
    double reorder_centers = 0.0;
    double create_block_info = 0.0;
    double block_sending = 0.0;
    double nn_searching = 0.0;
    
    // GPU timings
    double gpu_copy = 0.0;
    double computation = 0.0;
    double gpu_total = 0.0;
    double cleanup_gpu = 0.0;
    
    // Overall timing
    double total = 0.0;
    
    // GFLOPS
    double total_gflops = 0.0;
    
    /**
     * @brief Reset all timing values to zero
     */
    void reset() {
        RAC_partitioning = 0.0;
        centers_of_gravity_calculation = 0.0;
        send_centers_of_gravity = 0.0;
        reorder_centers = 0.0;
        create_block_info = 0.0;
        block_sending = 0.0;
        nn_searching = 0.0;
        gpu_copy = 0.0;
        computation = 0.0;
        gpu_total = 0.0;
        cleanup_gpu = 0.0;
        total = 0.0;
        total_gflops = 0.0;
    }
};

} // namespace utilities
} // namespace vecchia

#endif // VECCHIAGP_TIMING_DATA_HPP

