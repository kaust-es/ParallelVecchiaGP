
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file ParallelEstimator.cpp
 * @brief Implementation of Parallel Vecchia estimator (point-by-point processing).
 * @details This implements the Parallel Vecchia approximation where each location conditions
 * on its nearest neighbors, as opposed to Block Vecchia which processes blocks of locations.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2025-01-18
**/

#include <estimators/concrete/ParallelEstimator.hpp>
#include <hardware/VecchiaHardware.hpp>
#include <kernels/Kernel.hpp>
#include <common/PluginRegistry.hpp>
#include <helpers/CommunicatorMPI.hpp>
#include <utilities/Logger.hpp>
#include <utilities/ErrorHandler.hpp>
#include <cuda-kernels/gpukernels.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <kblas.h>

#include <cmath>
#include <omp.h>
#include <common/Definitions.hpp>

using namespace std;
using namespace vecchia::estimators;
using namespace vecchia::common;
using namespace vecchia::configurations;
using namespace vecchia::dataunits;

// Helper function to compute log-determinant from Cholesky factor on GPU
template<typename T>
void core_Xlogdet(T* L, int An, int ldda, T* logdet_result_h) {
    T* L_h = (T*)malloc(sizeof(T) * An * ldda);
    cudaMemcpy(L_h, L, sizeof(T) * An * ldda, cudaMemcpyDeviceToHost);
    *logdet_result_h = 0;
    for (int i = 0; i < An; i++) {
        if (L_h[i + i * ldda] > 0) {
            *logdet_result_h += log(L_h[i + i * ldda] * L_h[i + i * ldda]);
        }
    }
    free(L_h);
}

template<typename T>
void ParallelEstimator<T>::InitMemory(configurations::Configurations &aConfigurations, 
                                       std::unique_ptr<VecchiaGBData<T>> &aData) {
    
    LOGGER("--------Using the Parallel Vecchia Method--------")
    
    // Configuration
    int n = aConfigurations.GetProblemSize();
    int cs = aConfigurations.GetConditioningSize();
    int batchCount = aData->GetBatchCount();
    int ngpu = aConfigurations.GetGPUsNumbers();
    int align = 32;
    int bs = 1;
    int M = 1, N = 1;
    int Am = M, An = M;
    int Cm = M, Cn = N;
    int lda = Am;
    int ldc = Cm;
    int Acon = cs, Ccon = cs;
    int ldacon = cs;
    int ldccon = cs;
    
    // Calculate batch distribution across GPUs
    int* batchCount_gpu = nullptr;
    TESTING_MALLOC_PIN(batchCount_gpu, int, ngpu);
    
    if (ngpu > 1) {
        for (int g = 0; g < ngpu; g++) {
            if (g == (ngpu - 1)) {
                batchCount_gpu[g] = batchCount / ngpu + batchCount % ngpu;
            } else {
                batchCount_gpu[g] = batchCount / ngpu;
            }
        }
    } else {
        batchCount_gpu[0] = batchCount;
    }
    
    // Calculate aligned leading dimensions
    int ldda = ((lda + 31) / 32) * 32;
    int lddc = ((ldc + 31) / 32) * 32;
    int lddccon = ((ldccon + 31) / 32) * 32;
    int lddacon = lddccon;
    
    LOGGER("Batch count: " << batchCount)
    LOGGER("Conditioning size: " << cs)
    LOGGER("Number of GPUs: " << ngpu)
    LOGGER("Batch count per GPU[0]: " << batchCount_gpu[0])
    LOGGER("Aligned leading dimensions: lddacon=" << lddacon << ", lddccon=" << lddccon)
    
    // CPU Memory Allocation
    double* h_A = nullptr;
    double* h_C = nullptr;
    double* h_C_data = nullptr;

    TESTING_MALLOC_CPU(h_A, double, lda * An * batchCount);
    TESTING_MALLOC_CPU(h_C, double, ldc * n);
    TESTING_MALLOC_CPU(h_C_data, double, ldc * n);
    
    // Vecchia offset arrays
    double* h_A_cross = nullptr;
    double* h_A_offset_vector = nullptr;
    double* h_mu_offset_vector = nullptr;
    
    TESTING_MALLOC_CPU(h_A_cross, double, (long long)ldacon * An * batchCount);
    TESTING_MALLOC_CPU(h_A_offset_vector, double, batchCount);
    TESTING_MALLOC_CPU(h_mu_offset_vector, double, batchCount);
    
    // Retrieve conditioning observations (allocated by KnnConditioningUpdater)
    double* h_C_conditioning = aData->GetHostConditioningObs();
    
    // Declare arrays of pointers for multi-GPU
    double** dot_result_h = new double*[ngpu];
    double** logdet_result_h = new double*[ngpu];
    double** d_A_conditioning = new double*[ngpu];
    double** d_A_cross = new double*[ngpu];
    double** d_C_conditioning = new double*[ngpu];
    double** d_A_offset_vector = new double*[ngpu];
    double** d_mu_offset_vector = new double*[ngpu];
    double** d_C = new double*[ngpu];
    int** d_info = new int*[ngpu];
    double** locations_xx_d = new double*[ngpu];
    double** locations_yy_d = new double*[ngpu];
    double** locations_con_xx_d = new double*[ngpu];
    double** locations_con_yy_d = new double*[ngpu];
    
    // Per-GPU Memory Allocation Loop
    for (int g = 0; g < ngpu; g++)
    {
        check_error(cudaSetDevice(g));
        
        // CPU result arrays per GPU
        TESTING_MALLOC_CPU(dot_result_h[g], double, batchCount_gpu[g]);
        TESTING_MALLOC_CPU(logdet_result_h[g], double, batchCount_gpu[g]);
        
        // GPU device memory
        TESTING_MALLOC_DEV(d_A_conditioning[g], double, (long long)lddacon * Acon * batchCount_gpu[g]);
        TESTING_MALLOC_DEV(d_A_cross[g], double, (long long)lddacon * An * batchCount_gpu[g]);
        TESTING_MALLOC_DEV(d_C_conditioning[g], double, (long long)lddccon * Cn * batchCount_gpu[g]);
        TESTING_MALLOC_DEV(d_mu_offset_vector[g], double, batchCount_gpu[g]);
        TESTING_MALLOC_DEV(d_A_offset_vector[g], double, batchCount_gpu[g]);
        
        // Location arrays for covariance generation
        TESTING_MALLOC_DEV(locations_xx_d[g], double, batchCount_gpu[g]);
        TESTING_MALLOC_DEV(locations_yy_d[g], double, batchCount_gpu[g]);
        TESTING_MALLOC_DEV(locations_con_xx_d[g], double, cs * batchCount_gpu[g]);
        TESTING_MALLOC_DEV(locations_con_yy_d[g], double, cs * batchCount_gpu[g]);
        
        // Additional arrays
        TESTING_MALLOC_DEV(d_C[g], double, lddc * batchCount_gpu[g]);
        TESTING_MALLOC_DEV(d_info[g], int, ngpu);
        
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Store pointers in aData for use in Estimate
    aData->SetHostCrossCov(h_A_cross);
    
    // Store multi-GPU configuration
    aData->SetNumGPUs(ngpu);
    aData->SetBatchCountGPU(batchCount_gpu);
    
    // Store multi-GPU arrays
    aData->SetDotResultH(dot_result_h);
    aData->SetLogdetResultH(logdet_result_h);
    aData->SetDeviceC(d_C);
    aData->SetDeviceInfoArray(d_info);
    aData->SetDeviceCovarianceConditioningArray(d_A_conditioning);
    aData->SetDeviceCovarianceCrossArray(d_A_cross);
    aData->SetDeviceObservationsConditioningArray(d_C_conditioning);
    aData->SetDeviceCovarianceOffsetArray(d_A_offset_vector);
    aData->SetDeviceMuOffsetArray(d_mu_offset_vector);
    aData->SetLocationsXXD(locations_xx_d);
    aData->SetLocationsYYD(locations_yy_d);
    aData->SetLocationsConXXD(locations_con_xx_d);
    aData->SetLocationsConYYD(locations_con_yy_d);
    
    // Also store first GPU's arrays in single-pointer members
    aData->SetDeviceConditioningCov(d_A_conditioning[0]);
    aData->SetDeviceCrossCov(d_A_cross[0]);
    aData->SetDeviceConditioningObs(d_C_conditioning[0]);
    aData->SetDeviceCovOffset(d_A_offset_vector[0]);
    aData->SetDeviceMuOffset(d_mu_offset_vector[0]);
    aData->SetLogDetResults(logdet_result_h[0]);
    aData->SetNorm2Results(dot_result_h[0]);
    
    LOGGER("Allocated Memory:")
    LOGGER("  Conditioning Covariance (GPU): " << (long long)lddacon * Acon * batchCount_gpu[0] * sizeof(double) / (1024.0 * 1024.0) << " MB per GPU")
    LOGGER("  Cross Covariance (CPU): " << (long long)ldacon * An * batchCount * sizeof(double) / (1024.0 * 1024.0) << " MB")
    LOGGER("  Conditioning Observations (CPU): " << (long long)ldccon * Cn * batchCount * sizeof(double) / (1024.0 * 1024.0) << " MB")
    LOGGER("-------------Memory Allocate Done-------------")
}

template<typename T>
T ParallelEstimator<T>::Estimate(configurations::Configurations &aConfigurations, 
                                  std::unique_ptr<VecchiaGBData<T>> &aData,
                                  const double *localtheta) {
    
    int n = aConfigurations.GetProblemSize();
    int cs = aConfigurations.GetConditioningSize();
    int batchCount = aData->GetBatchCount();
    int bs = 1;  // block size (M = 1 for point-by-point)
    int M = 1, N = 1;
    int Cm = M, Cn = N;
    int ldc = Cm;
    int Acon = cs, Ccon = cs;
    int ldacon = cs;
    int ldccon = cs;
    int align = 32;
    int lddacon = ((ldacon + align - 1) / align) * align;
    int lddccon = ((ldccon + align - 1) / align) * align;
    int ngpu = aData->GetNumGPUs();
    
    // Get locations
    Locations<T>* main_locations = aData->GetLocations();
    Locations<T>* conditioning_locations = aData->GetConditioningLocations();
    double* h_C_data = aData->GetHostObservations();
    
    // Get memory pointers from aData - multi-GPU arrays
    double* h_C_conditioning = aData->GetHostConditioningObs();
    int* batchCount_gpu = aData->GetBatchCountGPU();
    
    // Multi-GPU arrays (arrays of pointers)
    double** dot_result_h = aData->GetDotResultH();
    double** logdet_result_h = aData->GetLogdetResultH();
    double** d_A_conditioning = aData->GetDeviceCovarianceConditioningArray();
    double** d_A_cross = aData->GetDeviceCovarianceCrossArray();
    double** d_C_conditioning = aData->GetDeviceObservationsConditioningArray();
    double** d_A_offset_vector = aData->GetDeviceCovarianceOffsetArray();
    double** d_mu_offset_vector = aData->GetDeviceMuOffsetArray();
    double** d_C = aData->GetDeviceC();
    int** d_info = aData->GetDeviceInfoArray();
    
    // Location arrays per GPU
    double** locations_xx_d = aData->GetLocationsXXD();
    double** locations_yy_d = aData->GetLocationsYYD();
    double** locations_con_xx_d = aData->GetLocationsConXXD();
    double** locations_con_yy_d = aData->GetLocationsConYYD();
    
    // Allocate temporary host arrays
    double* h_A_offset_vector = nullptr;
    double* h_mu_offset_vector = nullptr;
    TESTING_MALLOC_CPU(h_A_offset_vector, double, batchCount);
    TESTING_MALLOC_CPU(h_mu_offset_vector, double, batchCount);
    
    // Allocate temporary variance array (h_A) and observation copy (h_C)
    double* h_A = nullptr;
    double* h_C = nullptr;
    TESTING_MALLOC_CPU(h_A, double, batchCount);
    TESTING_MALLOC_CPU(h_C, double, n);
    
    // Save original pointer for later freeing
    double* h_C_original = h_C;
    
    // Copy observations and adjust pointer
    memcpy(h_C, h_C_data, sizeof(double) * n);
    h_C = h_C + (cs - 1);
    
    // Initialize variance array with sigma^2
    for (int i = 1; i < batchCount; i++) {
        h_A[i] = localtheta[0];
    }
    
    // Copy location data to GPU per GPU
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        
        int _sum = 0;
        for (int ig = 0; ig < g; ig++) {
            _sum += batchCount_gpu[ig];
        }
        
        // Copy main location coordinates (for current location being processed)
        check_cublas_error(cublasSetVectorAsync(
            batchCount_gpu[g], sizeof(double),
            main_locations->GetLocationX() + (cs - 1) + _sum, 1,
            locations_xx_d[g], 1,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        check_cublas_error(cublasSetVectorAsync(
            batchCount_gpu[g], sizeof(double),
            main_locations->GetLocationY() + (cs - 1) + _sum, 1,
            locations_yy_d[g], 1,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        
        // Copy conditioning location coordinates
        check_cublas_error(cublasSetVectorAsync(
            cs * batchCount_gpu[g], sizeof(double),
            conditioning_locations->GetLocationX() + _sum * cs, 1,
            locations_con_xx_d[g], 1,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        check_cublas_error(cublasSetVectorAsync(
            cs * batchCount_gpu[g], sizeof(double),
            conditioning_locations->GetLocationY() + _sum * cs, 1,
            locations_con_yy_d[g], 1,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Generate covariance matrices on GPU per GPU
    std::string kernel_name = aConfigurations.GetKernelName();
    int distance_metric = 0;
    
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        
        if (kernel_name.find("Matern") != std::string::npos || kernel_name.find("matern") != std::string::npos) {
            // Generate conditioning covariance
            cudaDcmg_matern135_2_strided(
                d_A_conditioning[g],
                cs, cs, lddacon, Acon,
                locations_con_xx_d[g], locations_con_yy_d[g],
                locations_con_xx_d[g], locations_con_yy_d[g],
                localtheta, distance_metric,
                batchCount_gpu[g],
                kblasGetStream(VecchiaHardware::GetKblasHandle(g)));
            
            // Generate cross-covariance
            cudaDcmg_matern135_2_strided(
                d_A_cross[g],
                Acon, bs, lddacon, Acon,
                locations_con_xx_d[g], locations_con_yy_d[g],
                locations_xx_d[g], locations_yy_d[g],
                localtheta, distance_metric,
                batchCount_gpu[g],
                kblasGetStream(VecchiaHardware::GetKblasHandle(g)));
        }
        
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Replace first cs elements of d_A_cross[0] with first cs observations
    check_error(cudaSetDevice(0));
    check_cublas_error(cublasSetVector(cs, sizeof(double),
                                       h_C_data, 1,
                                       d_A_cross[0], 1));
    check_error(cudaDeviceSynchronize());
    check_error(cudaGetLastError());
    
    // Copy observations to GPU per GPU
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        
        int _sum = 0;
        for (int ig = 0; ig < g; ig++) {
            _sum += Cm * Cn * batchCount_gpu[ig];
        }
        
        // Copy main observations h_C to d_C[g]
        check_cublas_error(cublasSetMatrixAsync(
            Cm, Cn * batchCount_gpu[g], sizeof(double),
            h_C + _sum, ldc,
            d_C[g], ldc,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Copy conditioning observations to GPU
    int z2count = 0;
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        
        check_cublas_error(cublasSetMatrixAsync(
            cs, Cn * batchCount_gpu[g], sizeof(double),
            h_C_conditioning + z2count, ldccon,
            d_C_conditioning[g], lddccon,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        
        z2count += cs * Cn * batchCount_gpu[g];
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Query workspace and allocate for KBLAS per GPU
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        kblasHandle_t kblas_handle_g = VecchiaHardware::GetKblasHandle(g);
        
        kblas_potrf_batch_strided_wsquery(kblas_handle_g, Acon, batchCount_gpu[g] * 3);
        kblas_trsm_batch_strided_wsquery(kblas_handle_g, 'L', lddccon, Cn, batchCount_gpu[g]);
        kblas_trsm_batch_strided_wsquery(kblas_handle_g, 'L', lddacon, bs, batchCount_gpu[g]);
        check_kblas_error(kblasAllocateWorkspace(kblas_handle_g));
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Batched Cholesky decomposition per GPU
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        kblasHandle_t kblas_handle_g = VecchiaHardware::GetKblasHandle(g);
        
        check_kblas_error(kblasDpotrf_batch_strided(kblas_handle_g,
                                                    'L', Acon,
                                                    d_A_conditioning[g], lddacon, Acon * lddacon,
                                                    batchCount_gpu[g],
                                                    d_info[g]));
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Batched triangular solves per GPU
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        kblasHandle_t kblas_handle_g = VecchiaHardware::GetKblasHandle(g);
        
        // Solve for conditioning observations
        check_kblas_error(kblasDtrsm_batch_strided(kblas_handle_g,
                                                   'L', 'L', 'N', 'N',
                                                   lddccon, Cn,
                                                   1.0,
                                                   d_A_conditioning[g], lddacon, Acon * lddacon,
                                                   d_C_conditioning[g], lddccon, Cn * lddccon,
                                                   batchCount_gpu[g]));
        
        // Solve for cross-covariance
        check_kblas_error(kblasDtrsm_batch_strided(kblas_handle_g,
                                                   'L', 'L', 'N', 'N',
                                                   lddacon, bs,
                                                   1.0,
                                                   d_A_conditioning[g], lddacon, Acon * lddacon,
                                                   d_A_cross[g], lddacon, bs * lddacon,
                                                   batchCount_gpu[g]));
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Compute quadratic terms per GPU
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        
        DgpuDotProducts_Strided(
            d_A_cross[g], d_A_cross[g],
            d_A_offset_vector[g],
            batchCount_gpu[g], cs, lddacon,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g)));
        
        DgpuDotProducts_Strided(
            d_A_cross[g], d_C_conditioning[g],
            d_mu_offset_vector[g],
            batchCount_gpu[g], cs, lddacon,
            kblasGetStream(VecchiaHardware::GetKblasHandle(g)));
        
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Copy results back to host per GPU
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        
        int _count = 0;
        for (int j = 0; j < g; j++) {
            _count += batchCount_gpu[j];
        }
        
        check_cublas_error(cublasGetVectorAsync(batchCount_gpu[g], sizeof(double),
                                                d_A_offset_vector[g], 1,
                                                h_A_offset_vector + _count, 1,
                                                kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        check_cublas_error(cublasGetVectorAsync(batchCount_gpu[g], sizeof(double),
                                                d_mu_offset_vector[g], 1,
                                                h_mu_offset_vector + _count, 1,
                                                kblasGetStream(VecchiaHardware::GetKblasHandle(g))));
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Synchronize all GPUs
    for (int g = 0; g < ngpu; g++) {
        check_error(cudaSetDevice(g));
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    
    // Compute log-likelihood
    double llk = 0.0;
    
    // First independent block - compute log-determinant from Cholesky factor
    core_Xlogdet<double>(d_A_conditioning[0], cs, lddacon, &(logdet_result_h[0][0]));
    dot_result_h[0][0] = h_mu_offset_vector[0];
    
    // Scalar vecchia approximation per GPU
    for (int g = 0; g < ngpu; g++) {
        if (g == 0) {
            for (int i = 1; i < batchCount_gpu[g]; i++) {
                h_C[i] -= h_mu_offset_vector[i];
                h_A[i] -= h_A_offset_vector[i];
                
                dot_result_h[g][i] = h_C[i] * h_C[i] / h_A[i];
                logdet_result_h[g][i] = log(h_A[i]);
            }
        } else {
            int _sum_batchcvec = 0;
            for (int j = 0; j < g; j++) {
                _sum_batchcvec += batchCount_gpu[j];
            }
            for (int i = 0; i < batchCount_gpu[g]; i++) {
                h_C[_sum_batchcvec + i] -= h_mu_offset_vector[_sum_batchcvec + i];
                h_A[_sum_batchcvec + i] -= h_A_offset_vector[_sum_batchcvec + i];
                
                dot_result_h[g][i] = h_C[_sum_batchcvec + i] * h_C[_sum_batchcvec + i] / h_A[_sum_batchcvec + i];
                logdet_result_h[g][i] = log(h_A[_sum_batchcvec + i]);
            }
        }
    }
    
    // Sum up log-likelihood per GPU
    for (int g = 0; g < ngpu; g++) {
        for (int k = 0; k < batchCount_gpu[g]; k++) {
            double llk_temp = 0;
            int _size_llh = 1;
            if (g == 0 && k == 0) {
                _size_llh = cs;
            }
            llk_temp = -(dot_result_h[g][k] + logdet_result_h[g][k] + _size_llh * log(2 * PI)) * 0.5;
            llk += llk_temp;
        }
    }
    
    // Cleanup temporary allocations
    TESTING_FREE_CPU(h_A);
    TESTING_FREE_CPU(h_C_original);  // Free using original pointer
    TESTING_FREE_CPU(h_A_offset_vector);
    TESTING_FREE_CPU(h_mu_offset_vector);
    
    return llk;
}
