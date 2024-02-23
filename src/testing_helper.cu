/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file testing/testing_helper.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <thrust/logical.h>

#include <sys/time.h>
#include <stdarg.h>

#include <cmath>

#include "testing_helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Error helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const char *cublasGetErrorString(cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "success";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "not initialized";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "out of memory";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "invalid value";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "architecture mismatch";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "memory mapping error";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "execution failed";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "internal error";
  default:
    return "unknown error code";
  }
}

extern "C" void gpuCublasAssert(cublasStatus_t code, const char *file, int line)
{
  if (code != CUBLAS_STATUS_SUCCESS)
  {
    printf("gpuCublasAssert: %s %s %d\n", cublasGetErrorString(code), file, line);
    exit(-1);
  }
}

extern "C" void gpuKblasAssert(int code, const char *file, int line)
{
  if (code != 1) // TODO replace by KBlas_Success
  {
    printf("gpuKblasAssert: %s %s %d\n", kblasGetErrorString(code), file, line);
    exit(-1);
  }
}
////////////////////////////////////////////////////////////
// Command line parser
////////////////////////////////////////////////////////////

// Function to display help text
void displayHelp()
{
  std::cout << "Usage: test_dvecchia_batch [options]\n"
            << "Options:\n"
            << "  --help                Display this help message and exit\n"
            << "  --ikernel             The parameters in kernel, sigma^2:range:smooth, e.g., 1.5:0.1:0.5\n"
            << "  --kernel              The name of kernels, such as matern kernel, e.g., univariate_matern_stationary_no_nugget\n"
            << "  --kernel_init         The initial values of parameters in kernel, sigma^2:range:smooth, e.g., 1.5:0.1:0.5\n"
            << "  --vecchia_cs          [int] The conditioning size in Vecchia method, e.g., 1500,\n"
            << "  --num_loc             [int] The number of locations, e.g., 20000,\n"
            << "  --knn                 nearest neighbors searching, default to use.\n"
            << "  --perf                Only calculate the one iteraion of classic Vecchia.\n"
            << "  --seed                [int] random generation for locations and observations.\n"
            << "  --xy_path             [string] locations path.\n"
            << "  --obs_path            [string] observations path.\n"
            << "  --tol                 [int] tolerance of BOBYQA, 5 -> 1e-5.\n"
            << "  --omp_threads         [int] number openmp threads, default 40.\n"
            // Add more options as necessary
            << std::endl;
}

void kblas_assert(int condition, const char *msg, ...)
{
  if (!condition)
  {
    printf("Assert failed: ");
    va_list va;
    va_start(va, msg);
    vprintf(msg, va);
    printf("\n");
    exit(1);
  }
}

extern "C" int parse_opts(int argc, char **argv, kblas_opts *opts)
{
  // fill in default values
  for (int d = 0; d < MAX_NGPUS; d++)
    opts->devices[d] = d;

  opts->nstream = 1;
  opts->ngpu = 1;
  opts->tolerance = 0.;
  opts->time = 0;
  opts->nonUniform = 0; // TBD
  // opts->batchCount = 4;
  opts->strided = 1; // TBD

  // local theta for kernel in GPs
  opts->sigma = 0.1;
  opts->beta = 0.1;
  opts->nu = 0.1;
  opts->nugget = 0.0;
  // bivariate
  opts->sigma1 = 0.1;
  opts->sigma2 = 0.1;
  opts->alpha = 0.1;
  opts->nu1 = 0.1;
  opts->nu2 = 0.1;
  opts->beta = 0.1;

  // performance test
  opts->perf = 0;

  // vecchia conditioning
  opts->vecchia = 1;
  opts->vecchia_cs = 0;

  // optimization setting
  opts->tol = 1e-5;
  opts->maxiter = 1000;
  opts->lower_bound = 0.001;
  opts->upper_bound = 3;

  // openmp
  opts->omp_numthreads = 40;

  // extra config
  opts->kernel = 1;
  opts->num_params = 3;
  opts->num_loc = 40000;

  // bivariate
  opts->p = 1; // univaraite

  // k nearest neighbors
  opts->knn = 0;

  // random ordering
  opts->randomordering = 0;
  opts->mortonordering = 1;

  // irregular locations generation
  opts->seed = 0;

  int ndevices;
  cudaGetDeviceCount(&ndevices);
  int info;
  int ntest = 1;
  for (int i = 1; i < argc; ++i)
  {
    // ----- scalar arguments
    if (strcmp("--dev", argv[i]) == 0 && i + 1 < argc)
    {
      int n;
      info = sscanf(argv[++i], "%d", &n);
      if (info == 1)
      {
        char inp[512];
        char *pch;
        int ngpus = 0;
        strcpy(inp, argv[i]);
        pch = strtok(inp, ",");
        do
        {
          info = sscanf(pch, "%d", &n);
          if (ngpus >= MAX_NGPUS)
          {
            printf("warning: selected number exceeds KBLAS max number of GPUs, ngpus=%d.\n", ngpus);
            break;
          }
          if (ngpus >= ndevices)
          {
            printf("warning: max number of available devices reached, ngpus=%d.\n", ngpus);
            break;
          }
          if (n >= ndevices || n < 0)
          {
            printf("error: device %d is invalid; ensure dev in [0,%d].\n", n, ndevices - 1);
            break;
          }
          opts->devices[ngpus++] = n;
          pch = strtok(NULL, ",");
        } while (pch != NULL);
        opts->ngpu = ngpus;
      }
      else
      {
        fprintf(stderr, "error: --dev %s is invalid; ensure you have comma seperated list of integers.\n",
                argv[i]);
        exit(1);
      }
      kblas_assert(opts->ngpu > 0 && opts->ngpu <= ndevices,
                   "error: --dev %s is invalid; ensure dev in [0,%d].\n", argv[i], ndevices - 1);
    }
    else if (strcmp("--ngpu", argv[i]) == 0 && i + 1 < argc)
    {
      opts->ngpu = atoi(argv[++i]);
      kblas_assert(opts->ngpu <= MAX_NGPUS,
                   "error: --ngpu %s exceeds MAX_NGPUS, %d.\n", argv[i], MAX_NGPUS);
      kblas_assert(opts->ngpu <= ndevices,
                   "error: --ngpu %s exceeds number of CUDA devices, %d.\n", argv[i], ndevices);
      kblas_assert(opts->ngpu > 0,
                   "error: --ngpu %s is invalid; ensure ngpu > 0.\n", argv[i]);
    }
    else if (strcmp("--nstream", argv[i]) == 0 && i + 1 < argc)
    {
      opts->nstream = atoi(argv[++i]);
      kblas_assert(opts->nstream > 0,
                   "error: --nstream %s is invalid; ensure nstream > 0.\n", argv[i]);
    }
    else if (strcmp("--omp_threads", argv[i]) == 0 && i + 1 < argc)
    {
      opts->omp_numthreads = atoi(argv[++i]);
      kblas_assert(opts->omp_numthreads >= 1,
                   "error: --omp_numthreads %s is invalid; ensure omp_numthreads >= 1.\n", argv[i]);
    }
    else if (strcmp("--strided", argv[i]) == 0 || strcmp("-s", argv[i]) == 0)
    {
      opts->strided = 1;
    }
    // used for performance test
    else if (strcmp("--perf", argv[i]) == 0)
    {
      opts->perf = 1;
      opts->maxiter = 1;
    }
    // else if ( strcmp("--vecchia", argv[i]) == 0 ) {
    //    opts->vecchia  = 1;
    //   }
    else if ((strcmp("--vecchia_cs", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      int num;
      info = sscanf(argv[i], "%d", &num);
      if (info == 1 && num > 0)
      {
        opts->vecchia_cs = num;
        opts->vecchia = 1;
        // }else if(info == 1 && num == 0){
        //   opts->vecchia_cs = 0;
        //   opts->vecchia = 0;
      }
      else
      {
        fprintf(stderr, "error: --vecchia_cs %s is invalid; ensure only one number and 0 < vecchia_cs <= N.\n", argv[i]);
        exit(1);
      }
    }
    // real dataset input
    else if (strcmp(argv[i], "--xy_path") == 0 && i + 1 < argc)
    {
      i++;
      opts->xy_path = argv[i]; // The next argument is the path
      std::cout << "xy_path: " << opts->xy_path << std::endl;
    }
    // real dataset input
    else if (strcmp(argv[i], "--obs_path") == 0 && i + 1 < argc)
    {
      i++;
      opts->obs_path = argv[i]; // The next argument is the path
      std::cout << "obs_path: " << opts->obs_path << std::endl;
    }
    // used for optimization
    else if ((strcmp("--maxiter", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      int maxiter;
      info = sscanf(argv[i], "%d", &maxiter);
      if (info == 1 && maxiter > 0)
      {
        opts->maxiter = maxiter;
      }
      else
      {
        fprintf(stderr, "error: --maxiter %s is invalid; ensure maxiter > 0 and be integer.\n", argv[i]);
        exit(1);
      }
    }
    else if ((strcmp("--tol", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      int tol;
      info = sscanf(argv[i], "%d", &tol);
      if (info == 1 && tol > 0)
      {
        opts->tol = pow(10, -tol);
      }
      else
      {
        fprintf(stderr, "error: --tol %s is invalid; ensure tol > 0.\n", argv[i]);
        exit(1);
      }
    }
    else if ((strcmp("--lower_bound", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      double lower_bound;
      info = sscanf(argv[i], "%lf", &lower_bound);
      if (info == 1 && lower_bound > 0)
      {
        opts->lower_bound = lower_bound;
      }
      else
      {
        fprintf(stderr, "error: --lower_bound %s is invalid; ensure lower_bound > 0.\n", argv[i]);
        exit(1);
      }
    }
    else if ((strcmp("--upper_bound", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      double upper_bound;
      info = sscanf(argv[i], "%lf", &upper_bound);
      if (info == 1 && upper_bound < 100)
      {
        opts->upper_bound = upper_bound;
      }
      else
      {
        fprintf(stderr, "error: --upper_bound %s is invalid; ensure upper_bound < 100. (Or you fix 100 in opts file)\n", argv[i]);
        exit(1);
      }
    }
    // --- extra config
    else if ((strcmp("--kernel", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      char *kernel_str = argv[i];

      if (strcmp(kernel_str, "univariate_matern_stationary_no_nugget") == 0)
      {
        fprintf(stderr, "You are using the Matern Kernel 1/2, 3/2, 5/2, (sigma^2, range, smooth)!\n");
        opts->kernel = 1;     // You can change this value as needed
        opts->num_params = 3; // Set appropriate values for the 'matern' kernel
        opts->p = 1;          // You can modify this as per the requirement for 'matern'
      }
      else if (strcmp(kernel_str, "univariate_powexp_stationary_no_nugget") == 0)
      {
        fprintf(stderr, "You are using the Power exponential Kernel (sigma^2, range, smooth)!\n");
        opts->kernel = 2;     // Change as per your requirement for 'powexp'
        opts->num_params = 3; // Set appropriate values for the 'powexp' kernel
        opts->p = 1;          // Modify as needed for 'powexp'
      }
      else if (strcmp(kernel_str, "univariate_powexp_nugget_stationary") == 0)
      {
        fprintf(stderr, "You are using the Power exponential Kernel with nugget (sigma^2, range, smooth, nugget)!\n");
        opts->kernel = 3;     //
        opts->num_params = 4; //
        opts->p = 1;          // Modify as needed for 'powexp'
      }
      else
      {
        fprintf(stderr, "Unsupported kernel type: %s\n", kernel_str);
        exit(1);
      }
    }
    else if ((strcmp("--num_loc", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      int num_loc;
      info = sscanf(argv[i], "%d", &num_loc);
      opts->num_loc = num_loc;
    }
    // k nearest neighbors
    else if (strcmp("--knn", argv[i]) == 0)
    {
      opts->knn = 1;
    }
    // ordering
    else if (strcmp("--randomordering", argv[i]) == 0)
    {
      opts->randomordering = 1;
      opts->mortonordering = 0;
    }
    // iiregular locations generation seeds
    else if ((strcmp("--seed", argv[i]) == 0) && i + 1 < argc)
    {
      i++;
      int seed;
      info = sscanf(argv[i], "%d", &seed);
      opts->seed = seed;
    }
    // ture parameters
    else if (strcmp("--ikernel", argv[i]) == 0 && i + 1 < argc)
    {
      i++;
      double a1 = -1, a2 = -1, a3 = -1, a4 = -1; // Initialize with default values indicating 'unknown'
      char s1[10], s2[10], s3[10], s4[10];       // Arrays to hold the string representations

      // Parse the input into string buffers
      int info = sscanf(argv[i], "%9[^:]:%9[^:]:%9[^:]:%9[^:]", s1, s2, s3, s4);

      if (info < 3 && info > 4)
      {
        printf("Other kernels have been developing on the way!");
        exit(0);
      }

      // Check and convert each value
      if (strcmp(s1, "?") != 0)
        a1 = atof(s1);
      if (strcmp(s2, "?") != 0)
        a2 = atof(s2);
      if (strcmp(s3, "?") != 0)
        a3 = atof(s3);
      if (info == 4)
      {
        if (strcmp(s4, "?") != 0)
          a4 = atof(s4);
      }

      // Assign values to opts if they are not unknown
      if (a1 != -1)
        opts->sigma = a1;
      if (a2 != -1)
        opts->beta = a2;
      if (a3 != -1)
        opts->nu = a3;
      if (info == 4)
      {
        if (a4 != -1)
          opts->nugget = a4;
      }
    }
    // ----- usage
    else if (strcmp("-h", argv[i]) == 0 || strcmp("--help", argv[i]) == 0)
    {
      displayHelp();
      exit(0);
    }
    else
    {
      fprintf(stderr, "error: unrecognized option %s\n", argv[i]);
      exit(1);
    }
  }
  kblas_assert(ntest <= MAX_NTEST, "error: tests exceeded max allowed tests!\n");
  opts->ntest = ntest;

  // set device
  cudaError_t ed = cudaSetDevice(opts->devices[0]);
  if (ed != cudaSuccess)
  {
    printf("Error setting device : %s \n", cudaGetErrorString(ed));
    exit(-1);
  }

  return 1;
} // end parse_opts
