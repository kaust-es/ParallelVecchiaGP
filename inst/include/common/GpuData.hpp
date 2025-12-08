// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file GpuData.hpp
 * @brief Contains the definition of the GpuData struct for GPU memory management
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#ifndef VECCHIAGP_GPUDATA_HPP
#define VECCHIAGP_GPUDATA_HPP

#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <magma_v2.h>
#endif

/**
 * @struct GpuData
 * @brief Structure to hold GPU memory pointers and metadata for Scaled Block Vecchia
 */
struct GpuData {
    // Host arrays of pointers
    double **h_locs_array;
    double **h_locs_neighbors_array;
    double **h_observations_array;
    double **h_observations_neighbors_array;
    double **h_cov_array;
    double **h_cross_cov_array;
    double **h_conditioning_cov_array;
    double **h_observations_neighbors_copy_array;
    double **h_observations_copy_array;
    double **h_mu_correction_array;
    double **h_cov_correction_array;
    
    // Device arrays of pointers
    double **d_locs_array;
    double **d_locs_neighbors_array;
    double **d_observations_points_array;
    double **d_observations_neighbors_array;
    double **d_cov_array;
    double **d_cross_cov_array;
    double **d_conditioning_cov_array;
    double **d_observations_neighbors_copy_array;
    double **d_observations_copy_array;
    double **d_mu_correction_array;
    double **d_cov_correction_array;
    
    // Device memory pointers
    double *d_locs_device;
    double *d_locs_neighbors_device;
    double *d_observations_device;
    double *d_observations_neighbors_device;
    double *d_cov_device;
    double *d_conditioning_cov_device;
    double *d_cross_cov_device;
    double *d_observations_neighbors_copy_device;
    double *d_observations_copy_device;
    double *d_mu_correction_device;
    double *d_cov_correction_device;
    double *d_range_device;
    
    // Leading dimensions
    std::vector<int> lda_locs;
    std::vector<int> lda_locs_neighbors;
    std::vector<int> ldda_locs;
    std::vector<int> ldda_neighbors;
    std::vector<int> ldda_cov;
    std::vector<int> ldda_cross_cov;
    std::vector<int> ldda_conditioning_cov;
    std::vector<int> h_const1;
    
    // Device leading dimensions
    int *d_ldda_locs;
    int *d_ldda_neighbors;
    int *d_ldda_cov;
    int *d_ldda_cross_cov;
    int *d_ldda_conditioning_cov;
    int *d_lda_locs;
    int *d_lda_locs_neighbors;
    int *d_const1;
    
    // MAGMA info
    magma_int_t *dinfo_magma;
    magma_int_t max_m, max_n1, max_n2;
    
    // Sizes
    size_t total_observations_points_size;
    size_t total_observations_neighbors_size;
    size_t total_locs_num_device;
    size_t total_locs_neighbors_num_device;
    int numPointsPerProcess;
    
    // Constructor
    GpuData() : 
        h_locs_array(nullptr), h_locs_neighbors_array(nullptr), h_observations_array(nullptr),
        h_observations_neighbors_array(nullptr), h_cov_array(nullptr), h_cross_cov_array(nullptr),
        h_conditioning_cov_array(nullptr), h_observations_neighbors_copy_array(nullptr),
        h_observations_copy_array(nullptr), h_mu_correction_array(nullptr), h_cov_correction_array(nullptr),
        d_locs_array(nullptr), d_locs_neighbors_array(nullptr), d_observations_points_array(nullptr),
        d_observations_neighbors_array(nullptr), d_cov_array(nullptr), d_cross_cov_array(nullptr),
        d_conditioning_cov_array(nullptr), d_observations_neighbors_copy_array(nullptr),
        d_observations_copy_array(nullptr), d_mu_correction_array(nullptr), d_cov_correction_array(nullptr),
        d_locs_device(nullptr), d_locs_neighbors_device(nullptr), d_observations_device(nullptr),
        d_observations_neighbors_device(nullptr), d_cov_device(nullptr), d_conditioning_cov_device(nullptr),
        d_cross_cov_device(nullptr), d_observations_neighbors_copy_device(nullptr),
        d_observations_copy_device(nullptr), d_mu_correction_device(nullptr), d_cov_correction_device(nullptr),
        d_range_device(nullptr), d_ldda_locs(nullptr), d_ldda_neighbors(nullptr), d_ldda_cov(nullptr),
        d_ldda_cross_cov(nullptr), d_ldda_conditioning_cov(nullptr), d_lda_locs(nullptr),
        d_lda_locs_neighbors(nullptr), d_const1(nullptr), dinfo_magma(nullptr),
        max_m(0), max_n1(0), max_n2(0), total_observations_points_size(0), total_observations_neighbors_size(0),
        total_locs_num_device(0), total_locs_neighbors_num_device(0), numPointsPerProcess(0) {}
};

#endif // VECCHIAGP_GPUDATA_HPP
