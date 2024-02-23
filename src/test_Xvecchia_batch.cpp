/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file testing/batch_triangular/test_Xtrsm_batch.cpp

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
// #include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <ctime>
#include <nlopt.hpp>
#include <vector>
// #include <gsl/gsl_errno.h>
#include <typeinfo>

#if ((defined PREC_c) || (defined PREC_z)) && (defined USE_MKL)
// TODO need to handle MKL types properly
#undef USE_MKL
#endif

// KBLAS helper
#include "testing_helper.h"
#include "Xblas_core.ch"

// flops
#include "flops.h"

// Used for llh
#include "ckernel.h"
#include "llg.h"
// used for vecchia
extern "C"
{
#include "vecchia_helper_c.h"
}
// used for nearest neighbor
#include "nearest_neighbor.h"
// this is not formal statement and clarification, only for convenience
#include "utils.h"
#include "llh_Xvecchia_batch.h"

template <class T>
int test_Xvecchia_batch(kblas_opts &opts, T alpha)
{
    llh_data data;

    // preconfig
    bool strided = opts.strided;
    int ngpu = opts.ngpu;
    int nonUniform = opts.nonUniform;

    size_t batchCount;

    // BLAS language,
    // A* stands for the covariance matrix
    // C* stands for the observations
    int M, N;
    int Am, An, Cm, Cn;
    int lda, ldc, ldda, lddc;
    int ISEED[4] = {0, 0, 0, 1};
    // vecchia language
    // bs: block size; cs: conditioning size
    int bs, cs;

    kblasHandle_t kblas_handle[ngpu];

    T *h_A, *h_C;
    T *h_C_data;
    T *d_C[ngpu];
    T *dot_result_h[ngpu];
    T *logdet_result_h[ngpu];
    size_t batchCount_gpu[ngpu];
    //  potrf used
    int *d_info[ngpu];
    location *locations;
    location *locations_con;

    // // no nugget
    std::vector<T> localtheta_initial;
    std::vector<T> ub;
    T *grad; // used for future gradient based optimization, please do not comment it

    // vecchia offset
    // T *h_A_conditioning;
    T *h_A_cross, *h_C_conditioning;
    T *h_A_offset_vector, *h_mu_offset_vector;
    T *d_A_conditioning[ngpu], *d_A_cross[ngpu], *d_C_conditioning[ngpu];
    int ldacon, ldccon, Acon, Ccon;
    int lddacon, lddccon;
    // used for the store the memory of offsets for mu and
    // or, you could say that this is correction term
    T *d_A_offset_vector[ngpu], *d_mu_offset_vector[ngpu];
    // covariance matrix generation on GPU
    double *locations_xx_d[ngpu], *locations_yy_d[ngpu];
    double *locations_con_xx_d[ngpu], *locations_con_yy_d[ngpu];

    // time
    double whole_time = 0;

    // if (ngpu > 1)
    //     opts.check = 0;
    if (nonUniform)
        strided = 0;

    // USING
    cudaError_t err;
    for (int g = 0; g < ngpu; g++)
    {
        err = cudaSetDevice(opts.devices[g]);
        kblasCreate(&kblas_handle[g]);
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }

    // this is used for future development for the block vecchia
    M = N = 1;
    data.M = M;
    data.N = N;

    fflush(stdout);

    // first to create a folder to save the log information
    createLogFile(opts);

    // Vecchia config
    lda = Am = M;
    An = M;
    ldc = Cm = M;
    Cn = N;
    bs = M;

    if (cs > opts.num_loc)
    {
        fprintf(stderr, "Warning: your conditioning size is larger than the number of location in total!\n");
        cs = ldacon = ldccon = Acon = Ccon = opts.num_loc;
    }
    else
    {
        cs = ldacon = ldccon = Acon = Ccon = opts.vecchia_cs;
    }

    // the batchCount is choosen to the largest
    batchCount = opts.num_loc - cs + 1;
    fprintf(stderr, "Your totol batch count: %d, and the GPUs tobe used %d.\n", batchCount, ngpu);
    if (batchCount % ngpu != 0)
    {
        fprintf(stderr, "Warning: your data is not assigned to each gpu equally.\n");
    }

    TESTING_MALLOC_PIN(batchCount_gpu, int, ngpu);
    if (ngpu > 1)
    {
        for (int g = 0; g < ngpu; g++)
        {
            if (g == (ngpu - 1))
            {
                // last one contain the rests
                batchCount_gpu[g] = batchCount / ngpu + batchCount % ngpu;
            }
            else
            {
                // the rest has the same
                batchCount_gpu[g] = batchCount / ngpu;
            }
            // fprintf(stderr, "batchCount_gpu[g]: %d \n", batchCount_gpu[g]);
        }
    }
    else
    {
        batchCount_gpu[0] = batchCount / ngpu;
    }

    // Vecchia config for strided access
    ldda = ((lda + 31) / 32) * 32;
    lddc = ((ldc + 31) / 32) * 32;
    lddccon = ((ldccon + 31) / 32) * 32;
    lddacon = lddccon;

    // batched log-likelihood
    TESTING_MALLOC_CPU(h_A, T, lda * An * batchCount);
    // h_C_data: original data; h_C overwritten in each iteration
    TESTING_MALLOC_CPU(h_C, T, ldc * opts.num_loc);
    TESTING_MALLOC_CPU(h_C_data, T, ldc * opts.num_loc);
    if (opts.vecchia)
    {
        // used for vecchia offset
        // TESTING_MALLOC_CPU(h_A_conditioning, T,  static_cast<long long> (ldacon) * Acon * batchCount);
        TESTING_MALLOC_CPU(h_A_cross, T, ldacon * An * batchCount);
        TESTING_MALLOC_CPU(h_C_conditioning, T, ldccon * Cn * batchCount);
        // extra memory for mu
        TESTING_MALLOC_CPU(h_A_offset_vector, T, batchCount);
        TESTING_MALLOC_CPU(h_mu_offset_vector, T, batchCount);
    }

    for (int g = 0; g < ngpu; g++)
    {
        check_error(cudaSetDevice(opts.devices[g]));
        TESTING_MALLOC_CPU(dot_result_h[g], T, batchCount_gpu[g]);
        TESTING_MALLOC_CPU(logdet_result_h[g], T, batchCount_gpu[g]);
        if (opts.vecchia)
        {
            // used for vecchia offset
            TESTING_MALLOC_DEV(d_A_conditioning[g], T, static_cast<long long>(lddacon) * Acon * batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_A_cross[g], T, static_cast<long long>(lddacon) * An * batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_C_conditioning[g], T, lddccon * Cn * batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_mu_offset_vector[g], T, batchCount_gpu[g]);
            TESTING_MALLOC_DEV(d_A_offset_vector[g], T, batchCount_gpu[g]);
            // covariance matrix generation on GPU
            TESTING_MALLOC_DEV(locations_xx_d[g], double, batchCount_gpu[g]);
            TESTING_MALLOC_DEV(locations_yy_d[g], double, batchCount_gpu[g]);
            TESTING_MALLOC_DEV(locations_con_xx_d[g], double, cs *batchCount_gpu[g]);
            TESTING_MALLOC_DEV(locations_con_yy_d[g], double, cs *batchCount_gpu[g]);
        }
        TESTING_MALLOC_DEV(d_C[g], T, lddc * batchCount_gpu[g]);
        TESTING_MALLOC_DEV(d_info[g], int, ngpu);
        check_error(cudaDeviceSynchronize());
        check_error(cudaGetLastError());
    }
    // exit(0);
    // fprintf(stderr, "%d %d %d", lddacon, Acon,  batchCount_gpu[0]);

    /*
    Dataset: defined by yourself
    */
    // Uniform random generation for locations / read locations from disk
    // random generate with seed as 1
    if (opts.perf == 1)
    {
        // fprintf(stderr, "%d \n", opts.seed);
        locations = GenerateXYLoc(opts.num_loc, opts.seed);
        // for(int i = 0; i < opts.num_loc; i++) h_C_data[i] = (T) rand()/(T)RAND_MAX;
        for (int i = 0; i < opts.num_loc; i++)
            h_C_data[i] = 0.0;
        // printLocations(opts.num_loc, locations);
        // printVectorCPU(opts.num_loc, h_C, 1, 1);
        // for(int i = 0; i < Cm * batchCount; i++) printf("%ith %lf \n",i, h_C[i]);
        // // the optimization initial values
        if (opts.kernel == 1 || opts.kernel == 2)
        {
            localtheta_initial = {opts.sigma, opts.beta, opts.nu};
        }
        else if (opts.kernel == 3)
        {
            localtheta_initial = {opts.sigma, opts.beta, opts.nu, opts.nugget};
        }
        for (int i = 0; i < opts.num_params; i++)
            ub.push_back(2);
        // data.distance_metric = 0;
        // for(int i = 0; i < 30; i++) printf("%ith (%lf, %lf, %lf) \n", i, locations->x[i], locations->y[i], h_C_data[i]);
        // exit(0);
        // fprintf(stderr, "%lf, %lf ,%lf \n", localtheta_initial[0], localtheta_initial[1], localtheta_initial[2]);
    }
    else
    {
        // real dataset wind speed (umcomment it if used)
        std::string xy_path;
        std::string z_path;
        if (opts.xy_path.empty())
        {
            if (opts.num_loc == 250000)
            {
                // xy_path = "./soil_moist/meta_train_0.125";
                // z_path = "./soil_moist/observation_train_0.125";
                xy_path = "./wind/meta_train_250000";
                z_path = "./wind/observation_train_250000";
            }
            data.distance_metric = 1;
        }
        else
        {
            // Convert char* to std::string
            xy_path = opts.xy_path;
            z_path = opts.obs_path;
        }

        for (int i = 0; i < opts.num_params; i++)
            ub.push_back(2);
        locations = loadXYcsv(xy_path, opts.num_loc);
        loadObscsv<T>(z_path, opts.num_loc, h_C_data);
        if (opts.kernel == 1)
        {
            localtheta_initial = {opts.lower_bound, opts.lower_bound, opts.nu};
            // localtheta_initial = {1.5, 0.1, 2.5};
        }
        else if (opts.kernel == 3)
        {
            // power exponential with nugget effect
            localtheta_initial = {opts.lower_bound, opts.lower_bound, opts.lower_bound, opts.lower_bound};
        }
        else
        {
            // 2: power expoenential
            localtheta_initial = {opts.lower_bound, opts.lower_bound, opts.lower_bound};
        }
    }

    // Ordering for locations and observations
    if (opts.perf == 1)
    {
        if (opts.randomordering == 1)
        {
            fprintf(stderr, "You were using the Random ordering. \n");
            random_reordering(opts.num_loc, locations, h_C_data);
        }
        else
        {
            // for synthetic data in exageostat, morton is default
            fprintf(stderr, "You were using the Morton ordering. \n");
        }
    }
    else
    {
        // real dataset ordering
        if (opts.randomordering == 1)
        {
            fprintf(stderr, "You were using the Random ordering. \n");
            random_reordering(opts.num_loc, locations, h_C_data);
        }
        else if (opts.mortonordering == 1)
        {
            zsort_reordering(opts.num_loc, locations, h_C_data);
            fprintf(stderr, "You were using the Morton ordering. \n");
        }
    }
    /*
    Locations preparation
    */
    // printLocations(opts.num_loc, locations);
    // printLocations(batchCount * lda, locations);
    memcpy(h_C, h_C_data, sizeof(T) * opts.num_loc);
    if (opts.vecchia)
    {
        /************* Nearest neighbor searching ****************/
        locations_con = (location *)malloc(sizeof(location));
        locations_con->x = (T *)malloc(batchCount * cs / opts.p * sizeof(double));
        locations_con->y = (T *)malloc(batchCount * cs / opts.p * sizeof(double));
        locations_con->z = NULL;
        data.locations_con = locations_con;
        // copy for the first independent block
        // h_C_data: quadratic calculation for mu
        // locations_con->x/y: covariance matrix generation
        memcpy(h_C_conditioning, h_C_data, sizeof(T) * cs);
        memcpy(locations_con->x, locations->x, sizeof(T) * cs);
        memcpy(locations_con->y, locations->y, sizeof(T) * cs);
        // to match the nearest neighbor searching algorithm
        locations_con->x += cs;
        locations_con->y += cs;
        if (opts.knn)
        {
#pragma omp parallel for
            for (int i = 0; i < (batchCount - 1); i++)
            {
                // how many previous points you would like to include in your nearest neighbor searching
                // int con_loc = std::max(i * bs - 10000 * bs, 0);
                int con_loc = 0;
                findNearestPoints(
                    h_C_conditioning, h_C_data, locations_con, locations,
                    con_loc, cs + i * bs,
                    cs + (i + 1) * bs, cs, i, data.distance_metric);
                // printLocations(10, locations);
                // printLocations(cs * (i+1), locations_con);
                // if (i==0) exit(0);
                // fprintf(stderr, "asdasda\n");
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < (batchCount - 1); i++)
            {
                memcpy(locations_con->x + i * cs, locations->x + i * bs, sizeof(T) * cs);
                memcpy(locations_con->y + i * cs, locations->y + i * bs, sizeof(T) * cs);
                memcpy(h_C_conditioning + (i + 1) * cs, h_C + i * bs, sizeof(T) * cs);
            }
        }
        // point back to the starting;
        locations_con->x -= cs;
        locations_con->y -= cs;
        // printLocations(opts.num_loc, locations);
        // printLocations(cs * batchCount, locations_con);

        /************* Copy location from CPU to GPU ****************/
        for (int g = 0; g < ngpu; g++)
        {
            check_error(cudaSetDevice(opts.devices[g]));
            int _sum = 0;
            for (int ig = 0; ig < g; ig++)
            {
                _sum += batchCount_gpu[ig];
            }
            // conditioning locations
            check_cublas_error(cublasSetVectorAsync(
                cs * batchCount_gpu[g], sizeof(double),
                locations_con->x + _sum * cs, 1,
                locations_con_xx_d[g], 1,
                kblasGetStream(kblas_handle[g])));
            check_cublas_error(cublasSetVectorAsync(
                cs * batchCount_gpu[g], sizeof(double),
                locations_con->y + _sum * cs, 1,
                locations_con_yy_d[g], 1,
                kblasGetStream(kblas_handle[g])));
            // condition locations,
            check_cublas_error(cublasSetVectorAsync(
                batchCount_gpu[g], sizeof(double),
                locations->x + (cs - 1) + _sum, 1,
                locations_xx_d[g], 1,
                kblasGetStream(kblas_handle[g])));
            check_cublas_error(cublasSetVectorAsync(
                batchCount_gpu[g], sizeof(double),
                locations->y + (cs - 1) + _sum, 1,
                locations_yy_d[g], 1,
                kblasGetStream(kblas_handle[g])));
            check_error(cudaDeviceSynchronize());
            check_error(cudaGetLastError());
        }
        /**********************************************************/
    }

    // prepare these for llh_Xvecchia_batch
    data.M = M;
    data.N = N;
    data.strided = opts.strided;
    data.ngpu = opts.ngpu;
    data.nonUniform = opts.nonUniform;
    data.Am = Am;
    data.An = An;
    data.Cm = Cm;
    data.Cn = Cn;
    data.lda = lda;
    data.ldc = ldc;
    data.ldda = ldda;
    data.lddc = lddc;
    data.ldacon = ldacon;
    data.ldccon = ldccon;
    data.Acon = Acon;
    data.Ccon = Ccon;
    data.lddacon = lddacon;
    data.lddccon = lddccon;
    data.batchCount = batchCount;
    data.bs = bs;
    data.cs = cs;
    data.h_A = h_A;
    data.h_C = h_C;
    data.h_C_data = h_C_data;
    data.seed = opts.seed;
    // // no nugget
    data.locations = locations;
    // vecchia offset
    data.h_A_cross = h_A_cross;
    data.h_C_conditioning = h_C_conditioning;
    data.h_A_offset_vector = h_A_offset_vector;
    data.h_mu_offset_vector = h_mu_offset_vector;

    // opts
    data.vecchia = opts.vecchia;
    data.iterations = 0;
    data.omp_threads = opts.omp_numthreads;

    data.num_loc = opts.num_loc;
    // kernel related
    data.kernel = opts.kernel;
    data.num_params = opts.num_params;
    data.vecchia_time_total = 0; // used for accumulatet the time on vecchia
    data.p = opts.p;             // bivariate = 2 or univariate = 1
    data.perf = opts.perf;

    for (int g = 0; g < ngpu; g++)
    {
        data.kblas_handle[g] = &(kblas_handle[g]);

        data.d_C[g] = d_C[g];
        data.dot_result_h[g] = dot_result_h[g];
        data.logdet_result_h[g] = logdet_result_h[g];
        data.d_info[g] = d_info[g];
        data.d_A_conditioning[g] = d_A_conditioning[g];
        data.d_A_cross[g] = d_A_cross[g];
        data.d_C_conditioning[g] = d_C_conditioning[g];
        data.d_A_offset_vector[g] = d_A_offset_vector[g];
        data.d_mu_offset_vector[g] = d_mu_offset_vector[g];
        data.devices[g] = opts.devices[g];
        data.batchCount_gpu[g] = batchCount_gpu[g];
        // covariance matrix generation on GPU
        data.locations_xx_d[g] = locations_xx_d[g];
        data.locations_yy_d[g] = locations_yy_d[g];
        data.locations_con_xx_d[g] = locations_con_xx_d[g];
        data.locations_con_yy_d[g] = locations_con_yy_d[g];
    }

    struct timespec start_whole, end_whole;
    clock_gettime(CLOCK_MONOTONIC, &start_whole);

    // Set up the optimization problem
    nlopt::opt opt(nlopt::LN_BOBYQA, opts.num_params); // Use the BOBYQA algorithm in 2 dimensions
    std::vector<T> lb(opts.num_params, opts.lower_bound);
    if (opts.kernel == 4)
    {                   // bivariate matern kernel
        ub.back() = 1.; // beta should be constrained somehow
    }
    else if (opts.kernel == 1)
    {
        // matern kernel with fixed nu
        ub.back() = opts.nu;
        lb.back() = opts.nu;
    }
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_ftol_rel(opts.tol);
    opt.set_maxeval(opts.maxiter);
    // opt.set_maxeval(1);
    opt.set_max_objective(llh_Xvecchia_batch, &data); // Pass a pointer to the data structure
    // Optimize the log likelihood
    T maxf;
    try
    {
        // Cautious for future develop
        // Perform the optimization
        nlopt::result result = opt.optimize(localtheta_initial, maxf);
    }
    catch (const std::exception &e)
    {
        // Handle any other exceptions that may occur during optimization
        std::cerr << "Exception caught: " << e.what() << std::endl;
        // ...
    }

    double max_llh = opt.last_optimum_value();
    int num_iterations = opt.get_numevals();

    clock_gettime(CLOCK_MONOTONIC, &end_whole);
    whole_time = end_whole.tv_sec - start_whole.tv_sec + (end_whole.tv_nsec - start_whole.tv_nsec) / 1e9;
    saveLogFileSum<T>(num_iterations, localtheta_initial, max_llh, whole_time, opts);
    // int num_evals = 0;
    // num_evals = opt.get_numevals();
    printf("Done! \n");

    // vecchia
    // cudaFreeHost(h_A_conditioning);
    cudaFreeHost(h_A_cross);
    cudaFreeHost(h_C_conditioning);
    cudaFreeHost(locations->x);
    cudaFreeHost(locations->y);
    if (opts.vecchia)
    {
        cudaFreeHost(locations_con->x);
        cudaFreeHost(locations_con->y);
    }
    for (int g = 0; g < ngpu; g++)
    {
        check_error(cudaFree(d_A_conditioning[g]));
        check_error(cudaFree(d_A_cross[g]));
        check_error(cudaFree(d_C_conditioning[g]));
        check_error(cudaFree(d_A_offset_vector[g]));
        check_error(cudaFree(d_mu_offset_vector[g]));
        check_error(cudaFree(locations_con_xx_d[g]));
        check_error(cudaFree(locations_con_yy_d[g]));
    }
    // independent
    cudaFreeHost(h_A);
    cudaFreeHost(h_C);
    for (int g = 0; g < ngpu; g++)
    {
        check_error(cudaSetDevice(opts.devices[g]));
        check_error(cudaFree(d_C[g]));
    }

    for (int g = 0; g < ngpu; g++)
    {
        free(dot_result_h[g]);
        free(logdet_result_h[g]);
    }

    for (int g = 0; g < ngpu; g++)
    {
        kblasDestroy(&kblas_handle[g]);
    }
    return 0;
}

//==============================================================================================
int main(int argc, char **argv)
{

    kblas_opts opts;
    parse_opts(argc, argv, &opts);

    // #if defined PREC_d
    //     check_error(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    // #endif

    // #if (defined PREC_s) || (defined PREC_d)
    // TYPE alpha = 1.;
    // #elif defined PREC_c
    //     TYPE alpha = make_cuFloatComplex(1.2, -0.6);
    // #elif defined PREC_z
    //     TYPE alpha = make_cuDoubleComplex(1.2, -0.6);
    // #endif
    double alpha = 1.;
    // gsl_set_error_handler_off();
    return test_Xvecchia_batch<double>(opts, alpha);
}
