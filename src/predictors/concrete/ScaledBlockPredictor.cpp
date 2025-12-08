
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file ScaledBlockPredictor.cpp
 * @brief Implementation of Scaled Block Vecchia prediction
 * @details Implements distributed Scaled Block Vecchia prediction using GPU-accelerated computations
 *          with MAGMA vbatched operations for conditional mean and variance computation.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <magma_v2.h>

#include <predictors/concrete/ScaledBlockPredictor.hpp>
#include <estimators/concrete/ScaledBlockEstimator.hpp>
#include <hardware/VecchiaHardware.hpp>
#include <data-units/VecchiaGBData.hpp>
#include <data-units/BlockInfo.hpp>
#include <configurations/Configurations.hpp>
#include <kernels/Kernel.hpp>
#include <common/GpuData.hpp>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <magma_v2.h>
#include <cublas_v2.h>
#endif

using namespace vecchia::predictors;
using namespace vecchia::estimators;
using namespace vecchia::configurations;
using namespace vecchia::dataunits;
using namespace vecchia::common;

#ifdef USE_CUDA
extern GpuData copyDataToGPU(Configurations &aConfigurations, const std::vector<BlockInfo> &blockInfos, magma_queue_t queue);
extern void cleanupGpuMemory(GpuData &gpuData);
#endif

// Helper functions for error checking
#ifdef USE_CUDA
inline void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkMagmaError(magma_int_t error) {
    if (error != MAGMA_SUCCESS) {
        std::cerr << "MAGMA Error: " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

template <typename Real>
struct MagmaOps;

template <>
struct MagmaOps<double> {
    static inline void potrf_neighbors(magma_uplo_t uplo, const int* d_n, double** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_dpotrf_vbatched_max_nocheck(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, max_n, queue);
    }
    static inline void trsm_max(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                const int* d_m, const int* d_n,
                                double alpha,
                                double** d_A_array, const int* d_ldda_A,
                                double** d_B_array, const int* d_ldda_B,
                                magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_dtrsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
    static inline void gemm_max(magma_trans_t transA, magma_trans_t transB,
                                const int* d_m, const int* d_n, const int* d_k,
                                double alpha,
                                double const* const* d_A_array, const int* d_ldda_A,
                                double const* const* d_B_array, const int* d_ldda_B,
                                double beta,
                                double** d_C_array, const int* d_ldda_C,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* k_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_k));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magma_int_t* lddc_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_C));
        magmablas_dgemm_vbatched_max_nocheck(transA, transB, m_nc, n_nc, k_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, beta, d_C_array, lddc_nc, batchCount, max_m, max_n, max_k, queue);
    }
    static inline void potrf_final(magma_uplo_t uplo, const int* d_n, double** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_dpotrf_vbatched(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, queue);
    }
    static inline void trsm_final(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                  const int* d_m, const int* d_n, double alpha,
                                  double** d_A_array, const int* d_ldda_A,
                                  double** d_B_array, const int* d_ldda_B,
                                  magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_dtrsm_vbatched(side, uplo, transA, diag, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
};

template <>
struct MagmaOps<float> {
    static inline void potrf_neighbors(magma_uplo_t uplo, const int* d_n, float** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_spotrf_vbatched_max_nocheck(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, max_n, queue);
    }
    static inline void trsm_max(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                magma_int_t max_m, magma_int_t max_n,
                                const int* d_m, const int* d_n,
                                float alpha,
                                float** d_A_array, const int* d_ldda_A,
                                float** d_B_array, const int* d_ldda_B,
                                magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_strsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
    static inline void gemm_max(magma_trans_t transA, magma_trans_t transB,
                                const int* d_m, const int* d_n, const int* d_k,
                                float alpha,
                                float const* const* d_A_array, const int* d_ldda_A,
                                float const* const* d_B_array, const int* d_ldda_B,
                                float beta,
                                float** d_C_array, const int* d_ldda_C,
                                magma_int_t batchCount,
                                magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
                                magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* k_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_k));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magma_int_t* lddc_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_C));
        magmablas_sgemm_vbatched_max_nocheck(transA, transB, m_nc, n_nc, k_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, beta, d_C_array, lddc_nc, batchCount, max_m, max_n, max_k, queue);
    }
    static inline void potrf_final(magma_uplo_t uplo, const int* d_n, float** d_A_array, const int* d_ldda, magma_int_t* dinfo, magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* ldda_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda));
        magma_spotrf_vbatched(uplo, n_nc, d_A_array, ldda_nc, dinfo, batchCount, queue);
    }
    static inline void trsm_final(magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
                                  const int* d_m, const int* d_n, float alpha,
                                  float** d_A_array, const int* d_ldda_A,
                                  float** d_B_array, const int* d_ldda_B,
                                  magma_int_t batchCount, magma_queue_t queue) {
        magma_int_t* m_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_m));
        magma_int_t* n_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_n));
        magma_int_t* lddaA_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_A));
        magma_int_t* lddb_nc = const_cast<magma_int_t*>(reinterpret_cast<const magma_int_t*>(d_ldda_B));
        magmablas_strsm_vbatched(side, uplo, transA, diag, m_nc, n_nc, alpha, d_A_array, lddaA_nc, d_B_array, lddb_nc, batchCount, queue);
    }
};

#ifdef USE_CUDA
// Forward declaration for compute_covariance_vbatched from scaled_block_kernels.cu
extern void compute_covariance_vbatched(
    double **d_locs_A, int *d_lda_A, int inca, size_t total_A,
    double **d_locs_B, int *d_lda_B, int incb, size_t total_B,
    double **d_cov, int *d_ldda, int *d_n,
    size_t batchCount,
    int dim, const std::vector<double> &theta, double *d_range,
    bool add_nugget, cudaStream_t stream, Configurations &aConfigurations,
    int max_ldx1, int max_ldx2);


#endif

template <typename Real>
std::tuple<double, double, double> performPredictionOnGPU(const GpuData &gpuData, const std::vector<double> &theta, 
                                                          Configurations &aConfigurations, cudaStream_t stream, magma_queue_t queue)
{
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int gpu_id = VecchiaHardware::GetLocalGPUId();
    int dim = aConfigurations.GetDimensionSize();

    // Set the GPU
    checkCudaError(cudaSetDevice(gpu_id));
    
    size_t batchCount = gpuData.ldda_locs.size() - 1;
    magma_int_t *dinfo_magma = gpuData.dinfo_magma;
    int *d_ldda_locs = gpuData.d_ldda_locs;
    int *d_ldda_neighbors = gpuData.d_ldda_neighbors;
    int *d_ldda_cov = gpuData.d_ldda_cov;
    int *d_ldda_cross_cov = gpuData.d_ldda_cross_cov;
    int *d_ldda_conditioning_cov = gpuData.d_ldda_conditioning_cov;
    int *d_lda_locs = gpuData.d_lda_locs;
    int *d_lda_locs_neighbors = gpuData.d_lda_locs_neighbors;
    int *d_const1 = gpuData.d_const1;
    magma_int_t max_m = gpuData.max_m;
    magma_int_t max_n1 = gpuData.max_n1;
    magma_int_t max_n2 = gpuData.max_n2;
    int range_offset = 2;

    // copy the data from the device to the device
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_device, 
                                   gpuData.d_observations_neighbors_device, 
                                   gpuData.total_observations_neighbors_size, 
                                   cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_copy_device, 
                                   gpuData.d_observations_device, 
                                   gpuData.total_observations_points_size, 
                                   cudaMemcpyDeviceToDevice));
    {
        std::vector<Real> range_host(dim);
        for (int i=0;i<dim;++i) range_host[i] = static_cast<Real>(theta[range_offset + i]);
        checkCudaError(cudaMemcpy(gpuData.d_range_device, 
                                   range_host.data(), 
                                   dim * sizeof(Real), 
                                   cudaMemcpyHostToDevice));
    }

    // Use the data on the GPU for computation
    // 1. generate the covariance matrix, cross covariance matrix, conditioning covariance matrix
    // Pass pre-computed max dimensions to avoid expensive thrust::reduce!
    compute_covariance_vbatched(gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cov_array, gpuData.d_ldda_cov, gpuData.d_lda_locs,
                batchCount,
                dim, theta, gpuData.d_range_device, true, stream, aConfigurations,
                max_n1, max_n1);
    compute_covariance_vbatched(gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cross_cov_array, gpuData.d_ldda_cross_cov, gpuData.d_lda_locs,
                batchCount,
                dim, theta, gpuData.d_range_device, false, stream, aConfigurations,
                max_m, max_n1);
    compute_covariance_vbatched(gpuData.d_locs_neighbors_array,
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_conditioning_cov_array, gpuData.d_ldda_conditioning_cov, gpuData.d_lda_locs_neighbors,
                batchCount,
                dim, theta, gpuData.d_range_device, true, stream, aConfigurations,
                max_m, max_m);
    // Synchronize to make sure the kernel has finished
    checkCudaError(cudaStreamSynchronize(stream));
    
    // 2. perform the computation
    // 2.1 compute the correction term for mean and variance (i.e., Schur complement)
    MagmaOps<Real>::potrf_neighbors(MagmaLower, d_lda_locs_neighbors,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        dinfo_magma, batchCount, max_m, queue);
    // trsm
    MagmaOps<Real>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                        max_m, max_n1, 
                        d_lda_locs_neighbors, d_lda_locs,
                        (Real)1.0,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_cross_cov_array, d_ldda_cross_cov,
                        batchCount, queue);
    MagmaOps<Real>::trsm_max(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                        max_m, max_n2, 
                        d_lda_locs_neighbors, d_const1,
                        (Real)1.0,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                        batchCount, queue);
    // gemm
    MagmaOps<Real>::gemm_max(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_lda_locs, d_lda_locs_neighbors,
                             (Real)1, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                                gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             (Real)0, gpuData.d_cov_correction_array, d_ldda_cov,
                             batchCount, 
                             max_n1, max_n1, max_m, 
                             queue);
    MagmaOps<Real>::gemm_max(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_const1, d_lda_locs_neighbors,
                             (Real)1, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                                gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                             (Real)0, gpuData.d_mu_correction_array, d_ldda_locs,
                             batchCount, 
                             max_n1, max_n2, max_m,
                             queue);
    checkCudaError(cudaStreamSynchronize(stream));
    // 2.2 compute the conditional mean and variance
    for (size_t i = 0; i < batchCount; ++i){
        // compute conditional variance
        if constexpr (std::is_same<Real,double>::value) {
            magmablas_dgeadd(gpuData.lda_locs[i], gpuData.lda_locs[i],
                            -1.,
                            (double*)gpuData.h_cov_correction_array[i], gpuData.ldda_locs[i], 
                            (double*)gpuData.h_cov_array[i], gpuData.ldda_cov[i],
                            queue);
        } else {
            magmablas_sgeadd(gpuData.lda_locs[i], gpuData.lda_locs[i],
                            -1.f,
                            (float*)gpuData.h_cov_correction_array[i], gpuData.ldda_locs[i], 
                            (float*)gpuData.h_cov_array[i], gpuData.ldda_cov[i],
                            queue);
        }
        // compute conditional mean
        // copy h_mu_correction_array to h_observations_copy_array
        checkCudaError(cudaMemcpy(gpuData.h_observations_copy_array[i], 
                                  gpuData.h_mu_correction_array[i], 
                                  gpuData.lda_locs[i] * sizeof(Real), 
                                  cudaMemcpyDeviceToHost));
    }
    checkCudaError(cudaStreamSynchronize(stream));

    // New code starts here
    // 3. Copy mean and variance from GPU to CPU
    std::vector<Real> h_means(gpuData.numPointsPerProcess);
    std::vector<Real> h_variances(gpuData.numPointsPerProcess);
    std::vector<Real> true_observations(gpuData.numPointsPerProcess);
    // Copy true observations, accounting for padding
    int offset = 0;
    for (size_t i = 0; i < batchCount; ++i) {
        checkCudaError(cudaMemcpy(true_observations.data() + offset, 
                                  gpuData.h_observations_array[i], 
                                  gpuData.lda_locs[i] * sizeof(Real), 
                                  cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(h_means.data() + offset, 
                                  gpuData.h_observations_copy_array[i], 
                                  gpuData.lda_locs[i] * sizeof(Real), 
                                  cudaMemcpyDeviceToHost));
        //copy the diagonal of the covariance matrix
        for (size_t j = 0; j < gpuData.lda_locs[i]; ++j){
            Real tmp;
            checkCudaError(cudaMemcpy(&tmp, 
                                      &gpuData.h_cov_array[i][j * gpuData.ldda_cov[i] + j], 
                                      sizeof(Real), 
                                      cudaMemcpyDeviceToHost));
            h_variances[offset + j] = tmp;
        }
        offset += gpuData.lda_locs[i];
    }

    // 4. Perform sampling
    std::mt19937 gen(rank);
    std::vector<std::vector<double>> samples(gpuData.numPointsPerProcess, 
        std::vector<double>(1000));
    for (int i = 0; i < gpuData.numPointsPerProcess; ++i) {
        std::normal_distribution<double> d(h_means[i], std::sqrt(h_variances[i]));
        for (int j = 0; j < 1000; ++j) {
            samples[i][j] = d(gen);
        }
    }

    // 5. Calculate sample mean and sample variance
    std::vector<double> sample_means(gpuData.numPointsPerProcess);
    std::vector<double> sample_variances(gpuData.numPointsPerProcess);
    
    for (int i = 0; i < gpuData.numPointsPerProcess; ++i) {
        double sum = std::accumulate(samples[i].begin(), samples[i].end(), 0.0);
        sample_means[i] = sum / 1000;
        
        double sq_sum = std::inner_product(samples[i].begin(), samples[i].end(), samples[i].begin(), 0.0);
        sample_variances[i] = sq_sum / 1000 - sample_means[i] * sample_means[i];
    }
    
    // 6. Calculate MSPE, RMSPE and confidence interval coverage
    
    double local_mspe_sum = 0.0;
    double local_rmspe_sum = 0.0;
    int local_within_ci = 0;
    
    for (int i = 0; i < gpuData.numPointsPerProcess; ++i) {
        local_mspe_sum += std::pow(sample_means[i] - true_observations[i], 2);
        // Calculate percentage error for RMSPE
        if (std::abs(true_observations[i]) > 1e-5) {  // Avoid division by zero
            double rmspe = std::pow(100 * (sample_means[i] - true_observations[i]) / true_observations[i], 2);
            local_rmspe_sum += rmspe;
        }

        // Save prediction results to CSV file
        double ci_lower = sample_means[i] - 1.96 * std::sqrt(sample_variances[i]);
        double ci_upper = sample_means[i] + 1.96 * std::sqrt(sample_variances[i]);
        
        if (true_observations[i] >= ci_lower && true_observations[i] <= ci_upper) {
            local_within_ci++;
        }
    }
    
    // MPI Allreduce to sum up mspe, rmspe, within_ci, and point counts across all processes
    double global_mspe_sum = 0.0;
    double global_rmspe_sum = 0.0;
    int global_within_ci = 0;
    MPI_Allreduce(&local_mspe_sum, &global_mspe_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_rmspe_sum, &global_rmspe_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_within_ci, &global_within_ci, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Calculate final MSPE, RMSPE and CI coverage
    double mspe = std::round(global_mspe_sum / aConfigurations.GetTestPointsTotal() * 1e16) / 1e16;
    double rmspe = std::sqrt(global_rmspe_sum / aConfigurations.GetTestPointsTotal()); 
    double ci_coverage = static_cast<double>(global_within_ci) / aConfigurations.GetTestPointsTotal();

    // Print results
    if (rank == 0) {
        std::cout << "MSPE: " << mspe << std::endl;
        std::cout << "RMSPE: " << rmspe << "%" << std::endl;
        std::cout << "95% CI coverage: " << ci_coverage * 100 << "%" << std::endl;
        std::cout << "-------------------Prediction Done-----------------" << std::endl;
    }
    return std::make_tuple(mspe, rmspe, ci_coverage);
}

template<typename T>
void ScaledBlockPredictor<T>::InitMemory(Configurations &aConfigurations, 
                                         std::unique_ptr<VecchiaGBData<T>> &aData) {
    // For prediction, memory initialization is done in Predict method
    // This is similar to how ScaledBlockEstimator works
}

template<typename T>
T ScaledBlockPredictor<T>::Predict(Configurations &aConfigurations, 
                                   std::unique_ptr<VecchiaGBData<T>> &aData, 
                                   const double *apTheta) {

#ifdef USE_CUDA
    int rank = VecchiaHardware::GetMPIRank();
    int gpu_id = VecchiaHardware::GetLocalGPUId();
    
    // Static variables for GPU data and initialization state
    static GpuData gpuData;
    static bool gpu_initialized = false;
    static cudaStream_t stream = nullptr;
    static magma_queue_t queue = nullptr;

    double mspe = -1.0;
    double rmspe = -1.0;
    double ci_coverage = -1.0;
    
    // Initialize GPU data on first call
    if (!gpu_initialized) {
        // Get BlockInfo from VecchiaGBData
        auto& blockInfos_test = aData->GetBlockInfos_test();
        
        if (blockInfos_test.empty()) {
            if (rank == 0) {
                std::cerr << "ERROR: ScaledBlockPredictor::Predict() - BlockInfo_test data is empty!" << std::endl;
                std::cerr << "  Make sure clustering was performed before prediction." << std::endl;
            }
            return static_cast<T>(-15000.0);
        }
        
        if (rank == 0) {
            std::cout << "Initializing GPU for Scaled Block Vecchia Prediction with " << blockInfos_test.size() << " blocks" << std::endl;
        }
        
        // Set the GPU
        checkCudaError(cudaSetDevice(gpu_id));
        
        // Create CUDA stream
        checkCudaError(cudaStreamCreate(&stream));
        
        // Create MAGMA queue
        magma_queue_create(gpu_id, &queue);
        
        // Copy data to GPU
        gpuData = copyDataToGPU(aConfigurations, blockInfos_test, queue);
        
        gpu_initialized = true;
        
        if (rank == 0) {
            std::cout << "GPU initialization complete for prediction" << std::endl;
        }
    }
    
    // Get optimized theta values
    int dim = aConfigurations.GetDimensionSize();
    std::vector<double> optimized_theta(apTheta, apTheta + 2 + dim);
    
    // Perform prediction using the initialized GPU data
    std::tie(mspe, rmspe, ci_coverage) = performPredictionOnGPU<double>(gpuData, optimized_theta, aConfigurations, stream, queue);
    cleanupGpuMemory(gpuData);
    cudaStreamDestroy(stream);
    magma_queue_destroy(queue);

#else
    if (rank == 0) {
        std::cerr << "ERROR: ScaledBlockPredictor requires CUDA support!" << std::endl;
        std::cerr << "  Please recompile with USE_CUDA=ON" << std::endl;
    }
#endif

    return T(0.0);
}

// Explicit template instantiation - must be in the .cpp file where implementations are
// Note: VECCHIAGP_INSTANTIATE_CLASS macro in the header already instantiates double
// Only instantiate float if needed (the macro currently only instantiates double)
// template class vecchia::predictors::ScaledBlockPredictor<float>;

