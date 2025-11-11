
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// ExaGeoStat is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file ParallelBlockEstimator.cpp
 * @brief Implementation of linear algebra methods.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2025-09-29
**/

#include <estimators/concrete/ParallelBlockEstimator.hpp>
#include <kernels/Kernel.hpp>
#include <common/PluginRegistry.hpp>
#include <utilities/ErrorHandler.hpp>
#include <utilities/Flops.hpp>
#include <utilities/Logger.hpp>
#include <hardware/VecchiaHardware.hpp>

using namespace std;

using namespace vecchia::estimators;
using namespace vecchia::common;
using namespace vecchia::configurations;
using namespace vecchia::dataunits;


template<typename T>
void ParallelBlockEstimator<T>::InitMemory(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData) {
    
    int batchCount = aData->GetBatchCount();

    // Allocate memory for the arrays
    magma_int_t* host_ldda = aData->GetHostLDDA();
    magma_int_t* host_info_magma = aData->GetHostInfo();
    magma_int_t* host_const1 = aData->GetHostConst1();
    magma_int_t* device_batchNum = aData->GetDeviceBatchNum();
    magma_int_t* device_info_magma = aData->GetDeviceInfo();
    magma_int_t* device_const1 = aData->GetDeviceConst1();
    magma_int_t* device_ldda = aData->GetDeviceLDDA();
    magma_int_t* device_lda = aData->GetDeviceLDA();
    TESTING_CHECK(magma_imalloc_cpu(&host_ldda, batchCount));
    TESTING_CHECK(magma_imalloc_cpu(&host_info_magma, batchCount));
    TESTING_CHECK(magma_imalloc_cpu(&host_const1, batchCount));
    TESTING_CHECK(magma_imalloc(&device_batchNum, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&device_info_magma, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&device_const1, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&device_ldda, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&device_lda, batchCount + 1));

    for (int i = 0; i < batchCount; ++i)
    {
        host_const1[i] = 1;
        host_info_magma[i] = 0; // check the success of each batch 0: success 1: failure
    }

    // Assign the allocated memory to the data structure
    aData->SetHostInfo(host_info_magma);
    aData->SetHostConst1(host_const1);
    aData->SetDeviceBatchNum(device_batchNum);
    aData->SetDeviceInfo(device_info_magma);
    aData->SetDeviceConst1(device_const1);
    aData->SetDeviceLDDA(device_ldda);
    aData->SetDeviceLDA(device_lda);

    // batch sizes configuration and the memory allocations
    magma_int_t* host_lda = aData->GetHostLDA();
    int *batch_num = aData->GetBatchNum();
    host_lda = batch_num;
    
    // There is an issue where most of the API doesn't accept double precision
    long long total_size_cpu_covariance = 0, total_size_device_covariance = 0, total_size_cpu_observations = 0, total_size_device_observations = 0;
    real_Double_t gflops = 0;
    int align = 32;

    for (int k = 0; k < batchCount; k++)
    {
        host_ldda[k] = magma_roundup(batch_num[k], align); // multiple of 32 by default
        total_size_cpu_covariance += batch_num[k] * host_lda[k];
        total_size_device_covariance += batch_num[k] * host_ldda[k];
        total_size_cpu_observations += host_lda[k];
        total_size_device_observations += host_ldda[k];
        gflops += FLOPS_DPOTRF(batch_num[k]) / 1e9;
        gflops += FLOPS_DTRSM(MagmaLeft, batch_num[k], 1) / 1e9;
    }

    aData->SetTotalSizeDeviceObservations(total_size_device_observations);
    double *host_covariance = nullptr;
    double *device_covariance = nullptr;
    double *device_observations = nullptr;
    double *device_observations_copy = nullptr;
    TESTING_CHECK(magma_dmalloc_cpu(&host_covariance, total_size_cpu_covariance));
    TESTING_CHECK(magma_dmalloc(&device_covariance, total_size_device_covariance));
    TESTING_CHECK(magma_dmalloc(&device_observations, total_size_device_observations));
    TESTING_CHECK(magma_dmalloc(&device_observations_copy, total_size_device_observations));
    
    // Assign the allocated memory to the data structure
    aData->SetHostCovariance(host_covariance);
    aData->SetDeviceCovariance(device_covariance);
    aData->SetDeviceObservations(device_observations);
    aData->SetDeviceObservationsCopy(device_observations_copy);
    aData->SetHostLDA(host_lda);
    aData->SetHostLDDA(host_ldda);

    // the *_array (pointer to pointer)
    // is the same for each iteration in the log-likleihod
    double** host_covariance_array = nullptr;
    double** host_observations_array = nullptr;
    double** host_observations_array_copy = nullptr;
    double** device_covariance_array = nullptr;
    double** device_observations_array = nullptr;
    double** device_observations_array_copy = nullptr;
    
    TESTING_CHECK(magma_malloc_cpu((void **)&host_covariance_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc_cpu((void **)&host_observations_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc_cpu((void **)&host_observations_array_copy, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc((void **)&device_covariance_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc((void **)&device_observations_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc((void **)&device_observations_array_copy, batchCount * sizeof(double *)));
    
    // Set up array pointers
    host_covariance_array[0] = device_covariance;
    host_observations_array[0] = device_observations;
    host_observations_array_copy[0] = device_observations_copy;
    
    for (int i = 1; i < batchCount; i++)
    {
        host_covariance_array[i] = host_covariance_array[i - 1] + host_ldda[i - 1] * batch_num[i - 1];
        host_observations_array[i] = host_observations_array[i - 1] + host_ldda[i - 1];
        host_observations_array_copy[i] = host_observations_array_copy[i - 1] + host_ldda[i - 1];
    }
    
    // Copy observations to device
    double* host_observations_tmp = aData->GetHostObservations();
    double* device_observations_tmp = device_observations;

    // Get the queue from the hardware instance (static accessor)
    magma_queue_t queue = VecchiaHardware::GetQueue();
    
    for (int i = 0; i < batchCount; i++)
    {
        magma_dsetvector(batch_num[i],
                         host_observations_tmp, 1,
                         device_observations_tmp, 1,
                         queue);
        host_observations_tmp += host_lda[i];
        device_observations_tmp += host_ldda[i];
    }
    
    // Copy array pointers to device
    magma_setvector(batchCount, sizeof(double *), host_covariance_array, 1, device_covariance_array, 1, queue);
    magma_setvector(batchCount, sizeof(double *), host_observations_array, 1, device_observations_array, 1, queue);
    magma_setvector(batchCount, sizeof(double *), host_observations_array_copy, 1, device_observations_array_copy, 1, queue);
    magma_setvector(batchCount, sizeof(magma_int_t), batch_num, 1, device_batchNum, 1, queue);
    magma_setvector(batchCount, sizeof(magma_int_t), host_lda, 1, device_lda, 1, queue);
    magma_setvector(batchCount, sizeof(magma_int_t), host_ldda, 1, device_ldda, 1, queue);
    magma_setvector(batchCount, sizeof(magma_int_t), host_const1, 1, device_const1, 1, queue);
    
    // Assign array pointers to data structure
    aData->SetHostCovarianceArray(host_covariance_array);
    aData->SetDeviceCovarianceArray(device_covariance_array);
    aData->SetHostObservationsArray(host_observations_array);
    aData->SetDeviceObservationsArray(device_observations_array);
    aData->SetHostObservationsArrayCopy(host_observations_array_copy);
    aData->SetDeviceObservationsArrayCopy(device_observations_array_copy);
    
    // Handle Vecchia conditioning if enabled
    if (aConfigurations.GetConditioningSize() > 0)
    {
        int cs = aConfigurations.GetConditioningSize();
        
        // Allocate conditioning arrays
        magma_int_t* host_lda_conditioning = nullptr;
        magma_int_t* host_ldda_conditioning = nullptr;
        magma_int_t* device_lda_conditioning = nullptr;
        magma_int_t* device_ldda_conditioning = nullptr;
        
        TESTING_CHECK(magma_imalloc_cpu(&host_lda_conditioning, batchCount));
        TESTING_CHECK(magma_imalloc_cpu(&host_ldda_conditioning, batchCount));
        TESTING_CHECK(magma_imalloc(&device_lda_conditioning, batchCount + 1));
        TESTING_CHECK(magma_imalloc(&device_ldda_conditioning, batchCount + 1));
        
        for (int i = 0; i < batchCount; ++i)
            host_lda_conditioning[i] = cs;
        magma_setvector(batchCount, sizeof(magma_int_t), host_lda_conditioning, 1, device_lda_conditioning, 1, queue);
        
        for (int i = 0; i < batchCount; ++i)
            host_ldda_conditioning[i] = magma_roundup(cs, align);
        magma_setvector(batchCount, sizeof(magma_int_t), host_ldda_conditioning, 1, device_ldda_conditioning, 1, queue);
        
        // Allocate conditioning memory
        double* host_covariance_conditioning = nullptr;
        double* host_covariance_cross = nullptr;
        double* device_covariance_conditioning = nullptr;
        double* device_covariance_cross = nullptr;
        double* device_covariance_offset = nullptr;
        double* device_mu_offset = nullptr;
        double* device_observations_conditioning = nullptr;
        double* device_observations_conditioning_copy = nullptr;
        
        TESTING_CHECK(magma_dmalloc_cpu(&host_covariance_conditioning, cs * cs * batchCount));
        TESTING_CHECK(magma_dmalloc_cpu(&host_covariance_cross, cs * aConfigurations.GetProblemSize()));
        TESTING_CHECK(magma_dmalloc(&device_covariance_conditioning, host_ldda_conditioning[0] * cs * batchCount));
        TESTING_CHECK(magma_dmalloc(&device_covariance_cross, host_ldda_conditioning[0] * aConfigurations.GetProblemSize()));
        TESTING_CHECK(magma_dmalloc(&device_covariance_offset, total_size_device_covariance));
        TESTING_CHECK(magma_dmalloc(&device_mu_offset, total_size_device_observations));
        TESTING_CHECK(magma_dmalloc(&device_observations_conditioning, host_ldda_conditioning[0] * batchCount));
        TESTING_CHECK(magma_dmalloc(&device_observations_conditioning_copy, host_ldda_conditioning[0] * batchCount));
        
        // Allocate conditioning array pointers
        double** host_covariance_conditioning_array = nullptr;
        double** host_covariance_cross_array = nullptr;
        double** host_covariance_offset_array = nullptr;
        double** host_mu_offset_array = nullptr;
        double** host_observations_conditioning_array = nullptr;
        double** host_observations_conditioning_array_copy = nullptr;
        double** device_covariance_conditioning_array = nullptr;
        double** device_covariance_cross_array = nullptr;
        double** device_covariance_offset_array = nullptr;
        double** device_mu_offset_array = nullptr;
        double** device_observations_conditioning_array = nullptr;
        double** device_observations_conditioning_array_copy = nullptr;
        
        TESTING_CHECK(magma_malloc_cpu((void **)&host_covariance_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&host_covariance_cross_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&host_covariance_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&host_mu_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&host_observations_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&host_observations_conditioning_array_copy, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&device_covariance_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&device_covariance_cross_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&device_covariance_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&device_mu_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&device_observations_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&device_observations_conditioning_array_copy, batchCount * sizeof(double *)));
        
        // Set up conditioning array pointers
        host_covariance_conditioning_array[0] = device_covariance_conditioning;
        host_covariance_cross_array[0] = device_covariance_cross;
        host_covariance_offset_array[0] = device_covariance_offset;
        host_mu_offset_array[0] = device_mu_offset;
        host_observations_conditioning_array[0] = device_observations_conditioning;
        host_observations_conditioning_array_copy[0] = device_observations_conditioning_copy;
        
        for (int i = 1; i < batchCount; i++)
        {
            host_covariance_conditioning_array[i] = host_covariance_conditioning_array[i - 1] + host_ldda_conditioning[i - 1] * cs;
            host_covariance_cross_array[i] = host_covariance_cross_array[i - 1] + host_ldda_conditioning[i - 1] * batch_num[i - 1];
            host_covariance_offset_array[i] = host_covariance_offset_array[i - 1] + host_ldda[i - 1] * batch_num[i - 1];
            host_mu_offset_array[i] = host_mu_offset_array[i - 1] + host_ldda[i - 1];
            host_observations_conditioning_array[i] = host_observations_conditioning_array[i - 1] + host_ldda_conditioning[i - 1];
            host_observations_conditioning_array_copy[i] = host_observations_conditioning_array_copy[i - 1] + host_ldda_conditioning[i - 1];
        }
        
        // Copy conditioning observations to device
        double* host_observations_conditioning_tmp = aData->GetHostConditioningObs();
        double* device_observations_conditioning_tmp = device_observations_conditioning;
        
        for (int i = 0; i < batchCount; i++)
        {

            magma_dsetvector(cs,
                             host_observations_conditioning_tmp, 1,
                             device_observations_conditioning_tmp, 1,
                             queue);

            host_observations_conditioning_tmp += host_lda_conditioning[i];
            device_observations_conditioning_tmp += host_ldda_conditioning[i];
        }
        // Copy conditioning array pointers to device
        magma_setvector(batchCount, sizeof(double *), host_covariance_conditioning_array, 1, device_covariance_conditioning_array, 1, queue);
        magma_setvector(batchCount, sizeof(double *), host_covariance_cross_array, 1, device_covariance_cross_array, 1, queue);
        magma_setvector(batchCount, sizeof(double *), host_covariance_offset_array, 1, device_covariance_offset_array, 1, queue);
        magma_setvector(batchCount, sizeof(double *), host_mu_offset_array, 1, device_mu_offset_array, 1, queue);
        magma_setvector(batchCount, sizeof(double *), host_observations_conditioning_array, 1, device_observations_conditioning_array, 1, queue);
        magma_setvector(batchCount, sizeof(double *), host_observations_conditioning_array_copy, 1, device_observations_conditioning_array_copy, 1, queue);
        
        // Assign conditioning memory to data structure
        aData->SetHostConditioningCov(host_covariance_conditioning);
        aData->SetHostCrossCov(host_covariance_cross);
        aData->SetDeviceConditioningCov(device_covariance_conditioning);
        aData->SetDeviceCrossCov(device_covariance_cross);
        aData->SetDeviceCovOffset(device_covariance_offset);
        aData->SetDeviceMuOffset(device_mu_offset);
        aData->SetDeviceConditioningObs(device_observations_conditioning);
        aData->SetDeviceObservationsConditioningCopy(device_observations_conditioning_copy);
        
        aData->SetHostCovarianceConditioningArray(host_covariance_conditioning_array);
        aData->SetDeviceCovarianceConditioningArray(device_covariance_conditioning_array);
        aData->SetHostCovarianceCrossArray(host_covariance_cross_array);
        aData->SetDeviceCovarianceCrossArray(device_covariance_cross_array);
        aData->SetHostCovarianceOffsetArray(host_covariance_offset_array);
        aData->SetDeviceCovarianceOffsetArray(device_covariance_offset_array);
        aData->SetHostMuOffsetArray(host_mu_offset_array);
        aData->SetDeviceMuOffsetArray(device_mu_offset_array);
        aData->SetHostObservationsConditioningArray(host_observations_conditioning_array);
        aData->SetDeviceObservationsConditioningArray(device_observations_conditioning_array);
        aData->SetHostObservationsConditioningArrayCopy(host_observations_conditioning_array_copy);
        aData->SetDeviceObservationsConditioningArrayCopy(device_observations_conditioning_array_copy);
        
        aData->SetHostLDAConditioning(host_lda_conditioning);
        aData->SetHostLDDAConditioning(host_ldda_conditioning);
        aData->SetDeviceLDAConditioning(device_lda_conditioning);
        aData->SetDeviceLDDAConditioning(device_ldda_conditioning);
    }
    
    // Allocate intermediate results
    double* norm2_result_host = nullptr;
    double* logdet_result_host = nullptr;
    TESTING_CHECK(magma_dmalloc_cpu(&norm2_result_host, batchCount));
    TESTING_CHECK(magma_dmalloc_cpu(&logdet_result_host, batchCount));
    
    aData->SetNorm2Results(norm2_result_host);
    aData->SetLogDetResults(logdet_result_host);
    
    LOGGER("-------------Memory Allocate Done-------------")
}
template <typename T>
void core_Xlogdet(T *L, int An, int ldda, T *logdet_result_h)
{
  T *L_h = (T *)malloc(sizeof(T) * An * ldda);
  cudaMemcpy(L_h, L, sizeof(T) * An * ldda, cudaMemcpyDeviceToHost);
  *logdet_result_h = 0;
  for (int i = 0; i < An; i++)
  {
    if (L_h[i + i * ldda] > 0)
      *logdet_result_h += log(L_h[i + i * ldda] * L_h[i + i * ldda]);
  }
  free(L_h);
}

template<typename T>
T ParallelBlockEstimator<T>::Estimate(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData, const double *theta) {
    //-----------------------------------------------------------//
    //------------------Covariance matrix generation...-------------------//
    //-----------------------------------------------------------//

    // Generate covariance matrix for X using trend_model kernel
    // Register and create a kernel object
    kernels::Kernel<T> *pKernel = plugins::PluginRegistry<kernels::Kernel<T>>::Create(aConfigurations.GetKernelName(), aConfigurations.GetTimeSlot());

    // Use the FULL locations with offsets (kernel handles offset internally)
    // This matches the old code's behavior
#pragma omp parallel for
    for (size_t i = 0; i < aData->GetBatchCount(); i++)
    {   
        int offset = aData->GetBatchNumAccum()[i];
        
        pKernel->GenerateCovarianceMatrix(
            aData->GetHostCovariance() + aData->GetBatchNumSquareAccum()[i], 
            aData->GetBatchNum()[i], aData->GetBatchNum()[i], 
            offset, offset,  // Kernel uses offsets internally
            *aData->GetNewLocations(), *aData->GetNewLocations(), *aData->GetNewLocations(), 
            const_cast<double*>(theta), aConfigurations.GetDistanceMetric());
    }
    
    if(aConfigurations.GetConditioningSize() > 0){
        int cs = aConfigurations.GetConditioningSize();
        
#pragma omp parallel for
        for (size_t i = 0; i < aData->GetBatchCount(); i++){ 
            int offset_conditioning = i * cs;
            int offset_batch = aData->GetBatchNumAccum()[i];
            
            // Generate conditioning covariance: sigma_{22}
            pKernel->GenerateCovarianceMatrix(
                aData->GetHostConditioningCov() + i * cs * cs, 
                cs, cs, 
                offset_conditioning, offset_conditioning,
                *aData->GetConditioningLocations(), *aData->GetConditioningLocations(), *aData->GetConditioningLocations(), 
                const_cast<double*>(theta), aConfigurations.GetDistanceMetric());

            // Generate cross covariance: sigma_{12}
            pKernel->GenerateCovarianceMatrix(
                aData->GetHostCrossCov() + cs * offset_batch, 
                cs, aData->GetBatchNum()[i], 
                offset_conditioning, offset_batch,
                *aData->GetConditioningLocations(), *aData->GetNewLocations(), *aData->GetNewLocations(), 
                const_cast<double*>(theta), aConfigurations.GetDistanceMetric());

        }
    }
    
    //-----------------------------------------------------------//
    //------------------Memory set/get/...-----------------------//
    //-----------------------------------------------------------//
    double *host_Cov_tmp, *device_Cov_tmp;
    // Get the queue from the hardware instance (static accessor)
    magma_queue_t queue = VecchiaHardware::GetQueue();
    // copy the observations, which is overwritten for each iteration
    magma_dcopy(aData->GetHostLDDAConditioning()[0] * aData->GetBatchCount(), aData->GetDeviceConditioningObs(), 1, aData->GetDeviceObservationsConditioningCopy(), 1, queue);
    magma_dcopy(aData->GetTotalSizeDeviceObservations(), aData->GetDeviceObservations(), 1, aData->GetDeviceObservationsCopy(), 1, queue);
    
    host_Cov_tmp = aData->GetHostCovariance();
    device_Cov_tmp = aData->GetDeviceCovariance();
    for (int i = 0; i < aData->GetBatchCount(); i++)
    {
        magma_dsetmatrix(aData->GetBatchNum()[i], aData->GetBatchNum()[i],
                    host_Cov_tmp, aData->GetHostLDA()[i],
                    device_Cov_tmp, aData->GetHostLDDA()[i],
                    queue);
        host_Cov_tmp += aData->GetBatchNum()[i] * aData->GetHostLDA()[i];
        device_Cov_tmp += aData->GetBatchNum()[i] * aData->GetHostLDDA()[i];
    }
    magma_setvector(aData->GetBatchCount(), sizeof(int), aData->GetHostInfo(), 1, aData->GetDeviceInfo(), 1, queue);
    
    // Copy conditioning covariance and cross-covariance to device if conditioning is enabled
    if (aConfigurations.GetConditioningSize() > 0)
    {
        int cs = aConfigurations.GetConditioningSize();
        double* host_conditioning_cov_tmp = aData->GetHostConditioningCov();
        double* device_conditioning_cov_tmp = aData->GetDeviceConditioningCov();
        double* host_cross_cov_tmp = aData->GetHostCrossCov();
        double* device_cross_cov_tmp = aData->GetDeviceCrossCov();

        for (int i = 0; i < aData->GetBatchCount(); i++)
        {
            magma_dsetmatrix(cs, cs,
                        host_conditioning_cov_tmp, aData->GetHostLDAConditioning()[i],
                        device_conditioning_cov_tmp, aData->GetHostLDDAConditioning()[i],
                        queue);
            
            magma_dsetmatrix(cs, aData->GetBatchNum()[i],
                host_cross_cov_tmp, aData->GetHostLDAConditioning()[i],
                device_cross_cov_tmp, aData->GetHostLDDAConditioning()[i],
                queue);
            
            host_conditioning_cov_tmp += aData->GetHostLDAConditioning()[i] * cs;
            device_conditioning_cov_tmp += aData->GetHostLDDAConditioning()[i] * cs;
            host_cross_cov_tmp += aData->GetHostLDAConditioning()[i] * aData->GetBatchNum()[i];
            device_cross_cov_tmp += aData->GetHostLDDAConditioning()[i] * aData->GetBatchNum()[i];
            
        }



        //-----------------------------------------------------//
        //------------------Conditioning-----------------------//
        //-----------------------------------------------------//
        int info = magma_dpotrf_vbatched(
            MagmaLower, aData->GetDeviceLDAConditioning(),
            aData->GetDeviceCovarianceConditioningArray(), aData->GetDeviceLDDAConditioning(),
            aData->GetDeviceInfo(), aData->GetBatchCount(),
            queue);

        magmablas_dtrsm_vbatched(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
            aData->GetDeviceLDAConditioning(), aData->GetDeviceLDA(), 1.,
            aData->GetDeviceCovarianceConditioningArray(), aData->GetDeviceLDDAConditioning(),
            aData->GetDeviceCovarianceCrossArray(), aData->GetDeviceLDDAConditioning(),
            aData->GetBatchCount(), queue);

        magmablas_dtrsm_vbatched(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
            aData->GetDeviceLDAConditioning(), aData->GetDeviceConst1(), 1.,
            aData->GetDeviceCovarianceConditioningArray(), aData->GetDeviceLDDAConditioning(),
            aData->GetDeviceObservationsConditioningArrayCopy(), aData->GetDeviceLDDAConditioning(),
            aData->GetBatchCount(), queue);

        magmablas_dgemm_vbatched(MagmaTrans, MagmaNoTrans,
            aData->GetDeviceLDA(), aData->GetDeviceLDA(), aData->GetDeviceLDAConditioning(),
            1,
            aData->GetDeviceCovarianceCrossArray(), aData->GetDeviceLDDAConditioning(),
            aData->GetDeviceCovarianceCrossArray(), aData->GetDeviceLDDAConditioning(),
            0,
            aData->GetDeviceCovarianceOffsetArray(), aData->GetDeviceLDDA(),
            aData->GetBatchCount(),
            queue);

        // \Sigma_offset^T %*% z_offset
        // GEMV -> GEMM (GEMV is supposed to be better, but there is unknown issues with the API)
        magmablas_dgemm_vbatched(MagmaTrans, MagmaNoTrans,
            aData->GetDeviceLDA(), aData->GetDeviceConst1(), aData->GetDeviceLDAConditioning(),
            1,
            aData->GetDeviceCovarianceCrossArray(), aData->GetDeviceLDDAConditioning(),
            aData->GetDeviceObservationsConditioningArrayCopy(), aData->GetDeviceLDDAConditioning(),
            0,
            aData->GetDeviceMuOffsetArray(), aData->GetDeviceLDDA(),
            aData->GetBatchCount(),
            queue);

        for (size_t i = 1; i < aData->GetBatchCount(); ++i)
        {
            magmablas_dgeadd(aData->GetHostLDA()[i], aData->GetHostLDA()[i],
                -1.,
                aData->GetHostCovarianceOffsetArray()[i], aData->GetHostLDDA()[i], // d_ldda[i]
                aData->GetHostCovarianceArray()[i], aData->GetHostLDDA()[i],
                queue);

            magmablas_dgeadd(aData->GetHostLDA()[i], 1,
                -1,
                aData->GetHostMuOffsetArray()[i], aData->GetHostLDDA()[i],
                aData->GetHostObservationsArrayCopy()[i], aData->GetHostLDDA()[i],
                queue);
        }
    }

    //-----------------------------------------------------//
    //------------------independent blocks-----------------------//
    //-----------------------------------------------------//
    // intermidiate results
    double *logdet_result_h = aData->GetLogDetResults();
    double *norm2_result_h = aData->GetNorm2Results();
    int info = 0;        // debug for potrf
    double llk = 0;      // log-likelihood
    double _llk_tmp = 0; // debug for log-likelihood


    // cholesky
    info = magma_dpotrf_vbatched(
        MagmaLower, aData->GetDeviceBatchNum(),
        aData->GetDeviceCovarianceArray(), aData->GetDeviceLDDA(),
        aData->GetDeviceInfo(), aData->GetBatchCount(),
        queue);

    // Check for Cholesky failures
    magma_getvector(aData->GetBatchCount(), sizeof(int), aData->GetDeviceInfo(), 1, aData->GetHostInfo(), 1, queue);

    magmablas_dtrsm_vbatched(
        MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
        aData->GetDeviceBatchNum(), aData->GetDeviceConst1(), 1.,
        aData->GetDeviceCovarianceArray(), aData->GetDeviceLDDA(),
        aData->GetDeviceObservationsArrayCopy(), aData->GetDeviceLDDA(),
        aData->GetBatchCount(), queue);
    // printMatrixGPU(h_lda[1], 1, h_obs_array_copy[1], h_ldda[1], 1);
    for (int i = 0; i < aData->GetBatchCount(); ++i)
    {
        // determinant
        core_Xlogdet<T>(aData->GetHostCovarianceArray()[i], //
                        aData->GetBatchNum()[i], aData->GetHostLDDA()[i],
                        &(logdet_result_h[i]));
        // Dot scalar Z_new^T Z_new
        norm2_result_h[i] = magma_dnrm2(aData->GetHostLDA()[i],
                                        aData->GetHostObservationsArrayCopy()[i],
                                        1, queue);
    }


    for (int k = 0; k < aData->GetBatchCount(); k++)
    {
        _llk_tmp = -(norm2_result_h[k] * norm2_result_h[k] + logdet_result_h[k] + aData->GetBatchNum()[k] * log(2 * PI)) * 0.5;
        llk += _llk_tmp;
    }
    
    // Increment iteration counter and log results
    aData->SetMleIterations(aData->GetMleIterations() + 1);
    
    LOGGER("Iteration " + std::to_string(aData->GetMleIterations()) + 
           " - Model Parameters (Variance, Range, Smoothness): (" +
           std::to_string(theta[0]) + ", " +
           std::to_string(theta[1]) + ", " +
           std::to_string(theta[2]) + ") -> Loglik: " +
           std::to_string(llk))

    return llk;
}

