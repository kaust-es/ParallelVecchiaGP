/**
 * @file scaled_block_kernels.cu
 * @brief GPU kernels for Scaled Block Vecchia approximation
 * @details Implements batched covariance generation, norm, and determinant computations
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
 *
 * Ported from ParallelScaledBlockVecchiaGP/src/gpu_kernels.cu
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <mpi.h>

#include <configurations/Configurations.hpp>
#include <common/Definitions.hpp>

using namespace vecchia::configurations;
using namespace vecchia::common;

#define BATCHCOUNT_MAX 65536
#define THREADS_PER_BLOCK 64
#define THREAD_X (16)
#define THREAD_Y (16)

// ============================================================================
// Covariance Kernel Implementations
// ============================================================================

// Matern 7/2 kernel (scaled, with device function for batched operations)
__device__ void Matern72_scaled_matcov_vbatched_kernel_device(
    const double* __restrict__ d_X1, int ldx1, int incx1, int stridex1,
    const double* __restrict__ d_X2, int ldx2, int incx2, int stridex2,
    double* __restrict__ d_C, int ldc, int n, int dim, 
    double sigma2, const double* __restrict__ range, 
    double nugget, bool nugget_tag,
    int gtx, int gty) {
    if (gtx < ldx1 && gty < ldx2 && gtx >= 0 && gty >= 0) {
        // Skip upper triangle for symmetric matrices (50% speedup!)
        if (d_X1 == d_X2 && gty > gtx) return;
        double dist_square = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = d_X1[gtx * incx1 + k * stridex1];
            double x2 = d_X2[gty * incx2 + k * stridex2];
            double diff = x1 - x2;
            // OPTIMIZATION: Use multiplication instead of division (range contains inv_range²)
            dist_square += diff * diff * range[k];
        }
        double scaled_distance = sqrt(dist_square);
        double a0 = 1.0;
        double a1 = 1.0;
        double a2 = 2.0 / 5.0;
        double a3 = 1.0 / 15.0;
        double item_poly = a0 + a1 * scaled_distance + a2 * scaled_distance * scaled_distance + a3 * scaled_distance * scaled_distance * scaled_distance;
        d_C[gtx + gty * ldc] = sigma2 * item_poly * exp( - scaled_distance );
    }
    // Add nugget
    if (gtx == gty && gtx < ldx1 && gty < ldx2 && nugget_tag) {
        d_C[gtx + gty * ldc] += nugget;
    }
}

// Batched Matern 7/2 kernel
__global__ void Matern72_scaled_matcov_vbatched_kernel(
    double** __restrict__ d_X1, const int* __restrict__ ldx1, int incx1, int stridex1,
    double** __restrict__ d_X2, const int* __restrict__ ldx2, int incx2, int stridex2,
    double** __restrict__ d_C, const int* __restrict__ ldc, const int* __restrict__ n, int dim, 
    const double sigma2, const double nugget, const double* __restrict__ range, bool nugget_tag) {
    
    const int batchid = blockIdx.z;
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gty = blockIdx.y * blockDim.y + threadIdx.y;

    Matern72_scaled_matcov_vbatched_kernel_device(
        d_X1[batchid], ldx1[batchid], incx1, stridex1, 
        d_X2[batchid], ldx2[batchid], incx2, stridex2, 
        d_C[batchid], ldc[batchid], n[batchid], dim, 
        sigma2, range, nugget, nugget_tag, gtx, gty);
}

// Batched Matern 7/2 wrapper
void Matern72_scaled_matcov_vbatched(
    double** __restrict__ d_X1, const int* __restrict__ ldx1, int incx1, int stridex1,
    double** __restrict__ d_X2, const int* __restrict__ ldx2, int incx2, int stridex2,
    double** __restrict__ d_C, const int* __restrict__ ldc, const int* __restrict__ n, int dim, const std::vector<double> &theta,
    const double* __restrict__ range, bool nugget_tag, 
    int max_ldx1, int max_ldx2,
    int batchCount, cudaStream_t stream) {
    
    dim3 blockDim(THREAD_X, THREAD_Y, 1);
    const int gridx = ((max_ldx1 + blockDim.x - 1) / blockDim.x);
    const int gridy = ((max_ldx2 + blockDim.y - 1) / blockDim.y);
    
    // theta[0]: variance, theta[1]: nugget, theta[2:]: range
    const int max_batch = 65535;
    for (int i = 0; i < batchCount; i += max_batch) {
        int gridz = std::min(max_batch, batchCount - i);
        dim3 gridDim(gridx, gridy, gridz);
        Matern72_scaled_matcov_vbatched_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_X1 + i, ldx1 + i, incx1, stridex1, 
            d_X2 + i, ldx2 + i, incx2, stridex2, 
            d_C + i, ldc + i, n + i, dim, 
            theta[0], theta[1], range, nugget_tag);
    }
}

// PowerExponential kernel
__global__ void PowerExp_matcov_scaled_kernel(
    const double* X1, int ldx1, int incx1, int stridex1,
    const double* X2, int ldx2, int incx2, int stridex2,
    double* C, int ldc, int n, int dim, 
    double sigma2, double smoothness, double nugget, 
    const double* range, bool nugget_tag) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ldx1 && j < ldx2 && i >= 0 && j >= 0) {
        double dist_square = 0;
        for (int k = 0; k < dim; k++) {
            double x1 = X1[i * incx1 + k * stridex1];
            double x2 = X2[j * incx2 + k * stridex2];
            dist_square += (x1 - x2) * (x1 - x2) / (range[k] * range[k]);
        }
        double scaled_distance = sqrt(dist_square);
        double power_distance = pow(scaled_distance, smoothness);
        C[i + j * ldc] = sigma2 * exp(-power_distance);
    }
    // Add nugget
    if (i == j && i < ldx1 && j < ldx2 && nugget_tag) {
        C[i + j * ldc] += nugget;
    }
}

/**
 * @brief Compute covariance matrices in batched mode
 * 
 * Main entry point for batched covariance generation. Dispatches to the appropriate
 * kernel based on the kernel type specified in the configuration.
 * 
 * Pass max_ldx1 and max_ldx2 as parameters instead of recomputing
 * them with thrust::reduce on every call (huge bottleneck for large batch counts)!
 */
void compute_covariance_vbatched(
    double **d_locs_A, int *d_lda_A, int inca, size_t total_A,
    double **d_locs_B, int *d_lda_B, int incb, size_t total_B,
    double **d_cov, int *d_ldda, int *d_n,
    size_t batchCount,
    int dim, const std::vector<double> &theta, double *d_range,
    bool add_nugget, cudaStream_t stream, Configurations &opts,
    int max_ldx1, int max_ldx2) {
    
    // Dispatch based on kernel type
    std::string kernel_type = opts.GetKernelType();
    
    if (kernel_type == "Matern72") {
        Matern72_scaled_matcov_vbatched(
            d_locs_A, d_lda_A, inca, total_A,
            d_locs_B, d_lda_B, incb, total_B,
            d_cov, d_ldda, d_n, dim, 
            theta, d_range, add_nugget,
            max_ldx1, max_ldx2, batchCount,
            stream);
    } else {
        // For other kernel types, we need to implement them similarly
        // For now, throw an error
        throw std::runtime_error("Unsupported kernel type for batched operations: " + kernel_type + 
                                ". Currently only Matern72 is implemented for scaled block Vecchia.");
    }
}

// ============================================================================
// Batched Norm and Log-Determinant Computations
// ============================================================================

/**
 * @brief GPU kernel for computing squared norm of vectors in batch
 */
__global__ void norm2_batch_kernel(const int* lda, const double* const* d_A_array, 
                                   const int* ldda, int batchCount, double* norm2_results) {
    int batch_id = blockIdx.x;
    if (batch_id >= batchCount) return;

    int n = lda[batch_id];
    const double* d_A = d_A_array[batch_id];

    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double thread_sum = 0.0;

    // Each thread handles one element if possible
    if (threadIdx.x < n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double val = d_A[i];
            thread_sum += val * val;
        }
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        norm2_results[batch_id] = shared_sum[0];
    }
}

/**
 * @brief Compute squared norm of multiple vectors in batch
 * 
 * Computes ||v||^2 for each vector in the batch and returns the sum.
 */
double norm2_batch(int *d_n, double **d_vec, int *d_ldda, 
                               size_t batchCount, cudaStream_t stream) {
    double* d_norm2_results;
    cudaMalloc(&d_norm2_results, std::min((int)batchCount, BATCHCOUNT_MAX) * sizeof(double));

    double total_norm2 = 0.0;
    int remaining = batchCount;
    int offset = 0;

    while (remaining > 0) {
        int current_batch = std::min(remaining, BATCHCOUNT_MAX);

        dim3 gridDim(current_batch);
        dim3 blockDim(THREADS_PER_BLOCK);

        norm2_batch_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_n + offset, 
            d_vec + offset, 
            d_ldda + offset, 
            current_batch, 
            d_norm2_results
        );

        // Use thrust to sum up the results on the GPU
        thrust::device_ptr<double> dev_ptr(d_norm2_results);
        double batch_norm2 = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + current_batch);
        total_norm2 += batch_norm2;

        remaining -= current_batch;
        offset += current_batch;
    }

    cudaFree(d_norm2_results);

    return total_norm2;
}

/**
 * @brief GPU kernel for computing log-determinant from Cholesky factors in batch
 */
__global__ void log_det_batch_kernel(const int* lda, const double* const* d_A_array, 
                                     const int* ldda, int batchCount, double* log_det_results) {
    int batch_id = blockIdx.x;
    if (batch_id >= batchCount) return;

    int n = lda[batch_id];
    int ldda_matrix = ldda[batch_id];
    const double* d_A = d_A_array[batch_id];

    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double thread_sum = 0.0;

    // Each thread handles one diagonal element if possible
    if (threadIdx.x < n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double val = d_A[i * ldda_matrix + i];
            thread_sum += 2 * log(val);
        }
    }

    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        log_det_results[batch_id] = shared_sum[0];
    }
}

/**
 * @brief Compute log-determinant of Cholesky factors in batch
 * 
 * Given Cholesky factorization L of matrices, computes sum of log-determinants.
 * For a positive definite matrix A = L*L^T, det(A) = det(L)^2, so:
 * log(det(A)) = 2 * sum(log(diag(L)))
 */
double log_det_batch(int *d_n, double **d_L, int *d_ldda, 
                                 size_t batchCount, cudaStream_t stream) {
    double* d_log_det_results;
    cudaMalloc(&d_log_det_results, std::min((int)batchCount, BATCHCOUNT_MAX) * sizeof(double));

    double total_log_det = 0.0;
    int remaining = batchCount;
    int offset = 0;

    while (remaining > 0) {
        int current_batch = std::min(remaining, BATCHCOUNT_MAX);

        dim3 gridDim(current_batch);
        dim3 blockDim(THREADS_PER_BLOCK);
         
        // Compute 2 * \sum_{i=1}^{#Bi} log(L_ii) for each batch
        log_det_batch_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_n + offset, 
            d_L + offset, 
            d_ldda + offset, 
            current_batch, 
            d_log_det_results
        );

        // Use thrust to sum up the results: \sum_{i=1}^{batchCount} log |\Sigma_i|
        thrust::device_ptr<double> dev_ptr(d_log_det_results);
        double batch_log_det = thrust::reduce(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + current_batch);
        total_log_det += batch_log_det;

        remaining -= current_batch;
        offset += current_batch;
    }

    cudaFree(d_log_det_results);

    return total_log_det;
}

