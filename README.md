## Batched Vecchia Algorithm 

The batched Vecchia approximation is used to accelerate the Vecchia approximation of Gaussian process in GPU; and its accuracy (in terms of KL divergence) is reached around ***$10^{-1}$ to $10^{-3}$ with only 60 neighbors***. And this size can easily be extend to ***1 million locations*** within single 32Gib V100. The following is the performance,


Our computational harness is built using gcc version 10.2.0 (12.2.0) and CUDA version 11.4 (11.8). It was linked with Intel MKL 2022.2.1, KBLAS-GPU, MAGMA 2.6.0, and NLopt v2.7.1 optimization libraries. Our paper is linked as 

#### 1. Installation Guidence

Installation of MAGMA 2.6.0 and NLopt v2.7.1, please refer

  - MAGMA 2.6.0 and KBLAS-GPU
  - 1. Download https://icl.utk.edu/magma/downloads/ (Guidance) 
  - 2. Install guide, pleaes use intel MKL https://icl.utk.edu/projectsfiles/magma/doxygen/installing.html
  - 3. For example, `bash installExample.sh`

  - Nlopt v2.7.1 (easy guidance) https://nlopt.readthedocs.io/en/latest/NLopt_Installation/

#### 2. Usage 

Here are two examples illustrating how to use the block Vecchia (afterwards your results are stored in the `./log` fie)

0. Helper 
`./bin/test_dvecchia_batch --help (-h)`

1. Performance test, such as monitor the time or calculate the intermediate results in KL divergence,
`./bin/test_dvecchia_batch --ikernel 1.5:0.1:0.5 --kernel univariate_matern_stationary_no_nugget --num_loc 20000 --perf --vecchia_cs 300 --vecchia_bc 1500 --knn --seed 0`

2. Real dataset.
`./bin/test_dvecchia_batch --ikernel ?:?:? --kernel univariate_matern_stationary_no_nugget --num_loc 250000   --vecchia_cs 90 --knn --xy_path replace_your_location_path --obs_path replace your_observation_path`
(optional)
`--kernel_init 0.1:0.1:0.1  --tol 9 --omp_threads 40`

