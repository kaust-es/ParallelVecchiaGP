#ifndef LLH_XVECCHIA_BATCH_H
#define LLH_XVECCHIA_BATCH_H

#include <omp.h>
#include <iostream>
#include <cstdio> // For printf

#include "gpukernels.h"

template <class T>

T llh_Xvecchia_batch(unsigned n, const T *localtheta, T *grad, void *f_data)
{
    T llk = 0;
    llh_data *data = static_cast<llh_data *>(f_data);
    double gflops_batch_potrf = 0.0, gflops_batch_trsm = 0.0, gflops_quadratic = 0.0;
    double indep_time = 0.0, dcmg_time = 0.0, vecchia_time_batch_potrf = 0.0, vecchia_time_batch_trsm = 0.0, vecchia_time_quadratic = 0.0;
    double time_copy = 0.0, time_copy_hd = 0.0, time_copy_dh = 0.0, vecchia_time_total = 0.0;
    double alpha_1 = 1.;
    double beta_n1 = -1.;
    double beta_0 = 0.;
    int omp_threads = data->omp_threads;
    omp_set_num_threads(omp_threads);

    // printf("[info] Starting Covariance Generation. \n");
    struct timespec start_dcmg, end_dcmg;
    clock_gettime(CLOCK_MONOTONIC, &start_dcmg);

    // note that we need copy the first block oberservation;
    // then h_C needs to point the location after the first block size;
    memcpy(data->h_C, data->h_C_data, sizeof(T) * data->num_loc);
    data->h_C = data->h_C + data->cs - 1;

    // covariance matrix generation
    // the first h_A is only for complement;
    // here is the scalar variance
    // where localtheta[0] is the variance and
    // localtheta[3] represent the nugget
    for (int i = 1; i < data->batchCount; i++)
    {
        if (data->kernel == 3 && localtheta[3] > 0)
        {
            // powexp nugget
            data->h_A[i] = localtheta[0] + localtheta[3];
        }
        else
        {
            data->h_A[i] = localtheta[0];
        }
    }

    ///*************** Covariance matrix generation on GPU *****************//
    ///*************** Covariance and cross variance  *****************//
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        if (data->kernel == 1)
        {
            // matern kernel with fixed nu = 0.5 1.5 2.5
            cudaDcmg_matern135_2_strided(
                data->d_A_conditioning[g],
                data->cs, data->cs, data->lddacon, data->Acon,
                // int m0, int n0,
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                localtheta, data->distance_metric,
                data->batchCount_gpu[g],
                kblasGetStream(*(data->kblas_handle[g])));
            cudaDcmg_matern135_2_strided(
                data->d_A_cross[g],
                data->Acon, data->bs, data->lddacon, data->Acon,
                // int m0, int n0,
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                data->locations_xx_d[g],
                data->locations_yy_d[g],
                localtheta, data->distance_metric,
                data->batchCount_gpu[g],
                kblasGetStream(*(data->kblas_handle[g])));
        }
        else if (data->kernel == 2)
        {
            // exponential power kernel
            cudaDcmg_powexp_strided(
                data->d_A_conditioning[g],
                data->cs, data->cs, data->lddacon, data->Acon,
                // int m0, int n0,
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                localtheta, data->distance_metric,
                data->batchCount_gpu[g],
                kblasGetStream(*(data->kblas_handle[g])));
            cudaDcmg_powexp_strided(
                data->d_A_cross[g],
                data->Acon, data->bs, data->lddacon, data->Acon,
                // int m0, int n0,
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                data->locations_xx_d[g],
                data->locations_yy_d[g],
                localtheta, data->distance_metric,
                data->batchCount_gpu[g],
                kblasGetStream(*(data->kblas_handle[g])));
        }
        else if (data->kernel == 3)
        {
            // exponential power kernel with nugget
            cudaDcmg_powexp_nugget_strided(
                data->d_A_conditioning[g],
                data->cs, data->cs, data->lddacon, data->Acon,
                // int m0, int n0,
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                localtheta, data->distance_metric,
                data->batchCount_gpu[g],
                kblasGetStream(*(data->kblas_handle[g])));
            cudaDcmg_powexp_nugget_strided(
                data->d_A_cross[g],
                data->Acon, data->bs, data->lddacon, data->Acon,
                // int m0, int n0,
                data->locations_con_xx_d[g],
                data->locations_con_yy_d[g],
                data->locations_xx_d[g],
                data->locations_yy_d[g],
                localtheta, data->distance_metric,
                data->batchCount_gpu[g],
                kblasGetStream(*(data->kblas_handle[g])));
        }
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    // exit(0);
    // the first cs data->d_A_cross has to be treated carefully (replace with h_C_conditioning)
    // because we need its quadratic term
    check_error(cudaSetDevice(data->devices[0]));
    check_cublas_error(cublasSetVector(data->cs, sizeof(T),
                                       data->h_C_data, 1,
                                       data->d_A_cross[0], 1));
    check_error(cudaDeviceSynchronize());
    check_error(cudaGetLastError());
    // memcpy(data->h_A_cross, data->h_C_data, sizeof(T) * data->cs);

    ///*****************************************************************************//

    clock_gettime(CLOCK_MONOTONIC, &end_dcmg);
    dcmg_time = end_dcmg.tv_sec - start_dcmg.tv_sec + (end_dcmg.tv_nsec - start_dcmg.tv_nsec) / 1e9;
    check_error(cudaGetLastError());
    // exit(0);

    // timing
    struct timespec start_copy_hd, end_copy_hd;
    clock_gettime(CLOCK_MONOTONIC, &start_copy_hd);
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        int _sum = 0;
        for (int ig = 0; ig < g; ig++)
        {
            _sum += data->Cm * data->Cn * data->batchCount_gpu[ig];
        }
        check_cublas_error(cublasSetMatrixAsync(
            data->Cm, data->Cn * data->batchCount_gpu[g], sizeof(T),
            data->h_C + _sum, data->ldc,
            data->d_C[g], data->lddc,
            kblasGetStream(*(data->kblas_handle[g]))));
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    long long batch22count = 0;
    int batch21count = 0;
    int z2count = 0;

    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        check_cublas_error(cublasSetMatrixAsync(data->cs, data->Cn * data->batchCount_gpu[g], sizeof(T),
                                                data->h_C_conditioning + z2count,
                                                data->ldccon,
                                                data->d_C_conditioning[g],
                                                data->lddccon,
                                                kblasGetStream(*(data->kblas_handle[g]))));
        z2count += data->cs * data->Cn * data->batchCount_gpu[g];
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_copy_hd);
    time_copy_hd = end_copy_hd.tv_sec - start_copy_hd.tv_sec + (end_copy_hd.tv_nsec - start_copy_hd.tv_nsec) / 1e9;

    // conditioning part 1.2, wsquery for potrf and trsm
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        if (data->strided)
        {
            // *3 because of unstable of wsquery
            kblas_potrf_batch_strided_wsquery(*(data->kblas_handle[g]), data->Acon, data->batchCount_gpu[g] * 3);
            kblas_trsm_batch_strided_wsquery(*(data->kblas_handle[g]), 'L', data->lddccon, data->Cn, data->batchCount_gpu[g]);
            kblas_trsm_batch_strided_wsquery(*(data->kblas_handle[g]), 'L', data->lddccon, data->Cn, data->batchCount_gpu[g]);
        }
        check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }

    gflops_batch_potrf = data->batchCount * FLOPS_POTRF<T>(data->cs) / 1e9;
    gflops_batch_trsm = 2 * data->batchCount * FLOPS_TRSM<T>('L', data->lddacon, data->An) / 1e9;
    gflops_quadratic = 2 * data->batchCount * FLOPS_DOTPRODUCT<T>(data->cs) / 1e9;
    // gflops_quadratic = 2*FLOPS_GEMM_v1<T>(data->batchCount, data->batchCount, data->cs) / 1e9;

    /*----------------------------*/
    /* correction terms */
    /*----------------------------*/

    struct timespec start_batch_potrf, end_batch_potrf;
    clock_gettime(CLOCK_MONOTONIC, &start_batch_potrf);
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        /*
        cholesky decomposition
        */
        // for (int i = 0; i < data->batchCount_gpu[g]; i++) // data->batchCount_gpu[g]
        // {
        //     printf("%dth", i);
        //     printMatrixGPU(data->Acon, data->Acon, data->d_A_conditioning[g] + i * data->Acon * data->lddacon, data->lddacon);
        // }
        // printf("[info] Starting Cholesky decomposition. \n");
        if (data->strided)
        {
            check_kblas_error(kblasXpotrf_batch_strided(*(data->kblas_handle[g]),
                                                        'L', data->Acon,
                                                        data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon,
                                                        data->batchCount_gpu[g],
                                                        data->d_info[g]));
        }
        // printf("[info] Finished Cholesky decomposition. \n");
        // check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_batch_potrf);
    vecchia_time_batch_potrf = end_batch_potrf.tv_sec - start_batch_potrf.tv_sec + (end_batch_potrf.tv_nsec - start_batch_potrf.tv_nsec) / 1e9;

    /*
    triangular solution: L \Sigma_offset <- \Sigma_old && L z_offset <- z_old
    */
    struct timespec start_batch_trsm, end_batch_trsm;
    clock_gettime(CLOCK_MONOTONIC, &start_batch_trsm);
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        // printf("[info] Starting triangular solver. \n");
        if (data->strided)
        {
            // for (int i = 0; i < data->batchCount_gpu[g]; i++)
            // {
            //     // printf("%dth", i);
            //     printVecGPU(data->Acon, 1, data->d_A_cross[g] + i * data->lddacon, data->lddacon, i);
            // }
            // for (int i = 0; i < data->batchCount_gpu[g]; i++)
            // {
            //     printf("%dth", i);
            //     printMatrixGPU(data->Acon, data->Acon, data->d_A_conditioning[g] + i * data->Acon * data->lddacon, data->lddacon);
            // }
            // fprintf(stderr, "%d \n", data->batchCount_gpu[g]);
            check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                       'L', 'L', 'N', data->diag,
                                                       data->lddccon, data->Cn,
                                                       1.,
                                                       data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                       data->d_C_conditioning[g], data->lddccon, data->Cn * data->lddccon,
                                                       data->batchCount_gpu[g]));
            check_kblas_error(kblasXtrsm_batch_strided(*(data->kblas_handle[g]),
                                                       'L', 'L', 'N', data->diag,
                                                       data->lddacon, data->An,
                                                       1.,
                                                       data->d_A_conditioning[g], data->lddacon, data->Acon * data->lddacon, // A <- L
                                                       data->d_A_cross[g], data->lddacon, data->An * data->lddacon,
                                                       data->batchCount_gpu[g]));
        }
        // check_kblas_error(kblasAllocateWorkspace(*(data->kblas_handle[g])));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_batch_trsm);
    vecchia_time_batch_trsm = end_batch_trsm.tv_sec - start_batch_trsm.tv_sec + (end_batch_trsm.tv_nsec - start_batch_trsm.tv_nsec) / 1e9;

    /*----------------------------*/
    /* quadratic term calculations*/
    /*----------------------------*/
    struct timespec start_qua, end_qua;
    clock_gettime(CLOCK_MONOTONIC, &start_qua);
    for (int g = 0; g < data->ngpu; g++)
    {
        /*
        GEMM and GEMV: \Sigma_offset^T %*% \Sigma_offset and \Sigma_offset^T %*% z_offset
        */
        // start_timing(curStream);
        check_error(cudaSetDevice(data->devices[g]));
        if (data->strided)
        {
            // Launch the kernel
            // fprintf(stderr, "--------------gpu: %d ------------\n", g);
            // printVecGPU(data->Acon, data->Cn, data->d_A_cross[0], data->lddacon, 2);
            // printVecGPU(data->Acon, data->Cn, data->d_C_conditioning[0], data->lddacon, 2);
            DgpuDotProducts_Strided(
                data->d_A_cross[g], data->d_A_cross[g],
                data->d_A_offset_vector[g],
                data->batchCount_gpu[g],
                data->cs, data->lddacon,
                kblasGetStream(*(data->kblas_handle[g])));
            DgpuDotProducts_Strided(
                data->d_A_cross[g], data->d_C_conditioning[g],
                data->d_mu_offset_vector[g],
                data->batchCount_gpu[g],
                data->cs, data->lddacon,
                kblasGetStream(*(data->kblas_handle[g])));
            // printVecGPUv1(data->batchCount_gpu[g], data->d_A_offset_vector[g]);
        }
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_qua);
    vecchia_time_quadratic = end_qua.tv_sec - start_qua.tv_sec + (end_qua.tv_nsec - start_qua.tv_nsec) / 1e9;
    // copy
    struct timespec start_copy_dh, end_copy_dh;
    clock_gettime(CLOCK_MONOTONIC, &start_copy_dh);
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        int _count = 0;
        for (int j = 0; j < g; j++)
            _count += data->batchCount_gpu[j];
        // copy the mu' and sigma' from gpu to host
        check_cublas_error(cublasGetVectorAsync(data->batchCount_gpu[g], sizeof(T),
                                                data->d_A_offset_vector[g], 1,
                                                data->h_A_offset_vector + _count, 1,
                                                kblasGetStream(*(data->kblas_handle[g]))));
        check_cublas_error(cublasGetVectorAsync(data->batchCount_gpu[g], sizeof(T),
                                                data->d_mu_offset_vector[g], 1,
                                                data->h_mu_offset_vector + _count, 1,
                                                kblasGetStream(*(data->kblas_handle[g]))));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }
    clock_gettime(CLOCK_MONOTONIC, &end_copy_dh);
    time_copy_dh += end_copy_dh.tv_sec - start_copy_dh.tv_sec + (end_copy_dh.tv_nsec - start_copy_dh.tv_nsec) / 1e9;

    // synchronize the gpu
    for (int g = 0; g < data->ngpu; g++)
    {
        check_error(cudaSetDevice(data->devices[g]));
        check_error(cudaDeviceSynchronize()); // TODO sync with streams instead
        check_error(cudaGetLastError());
    }

    /*
    Independent computing
    */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // first independent block likelihood
    core_Xlogdet<T>(data->d_A_conditioning[0],
                    data->cs, data->lddacon,
                    &(data->logdet_result_h[0][0]));
    // data->dot_result_h[0][0] = data->h_mu_offset_matrix[0];
    data->dot_result_h[0][0] = data->h_mu_offset_vector[0];
    // scalar vecchia approximation
    // int _sum_batchcmat = 0;
    for (int g = 0; g < data->ngpu; g++)
    {
        if (g == 0)
        {
            for (int i = 1; i < data->batchCount_gpu[g]; i++)
            {
                // correction
                data->h_C[i] -= data->h_mu_offset_vector[i];
                data->h_A[i] -= data->h_A_offset_vector[i];
                // llhi calulation
                data->dot_result_h[g][i] = data->h_C[i] * data->h_C[i] / data->h_A[i];
                data->logdet_result_h[g][i] = log(data->h_A[i]);
                // fprintf(stderr, "The %d, %lf \n", i, data->h_A_offset_vector[i]);
            }
        }
        else
        {
            int _sum_batchcvec = 0;
            for (int j = 0; j < g; j++)
            {
                _sum_batchcvec += data->batchCount_gpu[j];
            }
            for (int i = 0; i < data->batchCount_gpu[g]; i++)
            {
                data->h_C[_sum_batchcvec + i] -= data->h_mu_offset_vector[_sum_batchcvec + i];
                // the first is no meaning
                data->h_A[_sum_batchcvec + i] -= data->h_A_offset_vector[_sum_batchcvec + i];
                // llhi calulation
                data->dot_result_h[g][i] = data->h_C[_sum_batchcvec + i] * data->h_C[_sum_batchcvec + i] / data->h_A[_sum_batchcvec + i];
                data->logdet_result_h[g][i] = log(data->h_A[_sum_batchcvec + i]);
            }
        }
    }
    // printf("-----------------------------------------\n");
    for (int g = 0; g < data->ngpu; g++)
    {
        // printf("----------------%dth GPU---------------\n", g);
        for (int k = 0; k < data->batchCount_gpu[g]; k++)
        {
            T llk_temp = 0;
            int _size_llh = 1;
            if (g == 0 && k == 0)
            {
                _size_llh = data->cs;
            }
            llk_temp = -(data->dot_result_h[g][k] + data->logdet_result_h[g][k] + _size_llh * log(2 * PI)) * 0.5;
            llk += llk_temp;
            // printf("%dth log determinant is % lf\n", k, data->logdet_result_h[g][k]);
            // printf("%dth dot product is % lf\n", k, data->dot_result_h[g][k]);
            // printf("%dth pi is % lf\n", k, _size_llh * log(2 * PI));
            // printf("%dth log likelihood is % lf\n", k, llk_temp);
            // printf("-------------------------------------\n");
        }
    }
    // recover the h_C
    data->h_C = data->h_C - (data->cs - 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    indep_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    vecchia_time_total = vecchia_time_batch_potrf + vecchia_time_batch_trsm + vecchia_time_quadratic;
    time_copy = time_copy_dh + time_copy_hd;
    if (data->perf == 1)
    {
        std::string _file_path = "./log/perf_locs_" + std::to_string(data->num_loc) + "_" + "cs_" + std::to_string(data->cs) + "_" + "seed_" + std::to_string(data->seed);
        const char *file_path = _file_path.c_str();
        FILE *_file = fopen(file_path, "w");
        if (_file == NULL)
        {
            perror("Error opening _file");
            return 1; // Handle error as appropriate
        }
        fprintf(_file, "===================================Execution time (s)=======================================\n");
        fprintf(_file, "   Time total      Time dcmg      Copy(CPU->GPU)     BatchPOTRF    BatchTRSM   Quadratic      Independent\n");
        fprintf(_file, "   %8.6lf        %8.6lf        %8.6lf        %8.6lf      %8.6lf      %8.6lf       %8.6lf\n",
                dcmg_time + indep_time + vecchia_time_total + time_copy,
                dcmg_time,
                time_copy_hd,
                vecchia_time_batch_potrf,
                vecchia_time_batch_trsm,
                vecchia_time_quadratic,
                indep_time);
        fprintf(_file, "=============================Computing performance (Gflops/s)===================================\n");
        fprintf(_file, "     Vecchia         BatchPOTRF         BatchTRSM       Quadratic\n");
        fprintf(_file, "   %8.2lf           %8.2lf        %8.2lf        %8.2lf  \n",
                (gflops_batch_potrf + gflops_batch_trsm + gflops_quadratic) / (vecchia_time_batch_potrf + vecchia_time_batch_trsm + vecchia_time_quadratic),
                gflops_batch_potrf / vecchia_time_batch_potrf,
                gflops_batch_trsm / vecchia_time_batch_trsm,
                gflops_quadratic / vecchia_time_quadratic);
        // fprintf(_file, "%lf ==============\n", llk);
        fprintf(_file, "==========================================================================================\n");
        fclose(_file);
    }
    if (data->perf != 1)
    {
        if (data->kernel == 1 || data->kernel == 2)
        {
            printf("%dth Model Parameters (Variance, range, smoothness): (%1.8lf, %1.8lf, %1.8lf) -> Loglik: %.18lf \n",
                   data->iterations, localtheta[0], localtheta[1], localtheta[2], llk);
        }
        else if (data->kernel == 3)
        {
            printf("%dth Model Parameters (Variance, range, smoothness, nugget): (%1.8lf, %1.8lf, %1.8lf, %1.8lf) -> Loglik: %.18lf \n",
                   data->iterations, localtheta[0], localtheta[1], localtheta[2], localtheta[3], llk);
        }
        else if (data->kernel == 4)
        {
            printf("%dth Model Parameters (Variance1, Variance2, range, smoothness1, smoothness2, beta): (%lf, %lf, %lf, %lf, %lf, %lf) -> Loglik: %lf \n",
                   data->iterations, localtheta[0], localtheta[1], localtheta[2],
                   localtheta[3], localtheta[4], localtheta[5], llk);
        }
    }
    data->iterations += 1;
    // printf("-----------------------------------------------------------------------------\n");
    return llk;
}

#endif