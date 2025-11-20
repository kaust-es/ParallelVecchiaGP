# VecchiaGP User Guide

Complete guide for installation, usage, and examples for VecchiaGP.

---

## Table of Contents

1. [Installation Guide](#installation-guide)
   - [C++ Installation](#c-installation)
   - [R Package Installation](#r-package-installation)
   - [Troubleshooting](#troubleshooting)
2. [Examples](#examples)
   - [Parallel Vecchia](#parallel-vecchia)
   - [Block Vecchia](#block-vecchia)
   - [Scaled Block Vecchia](#scaled-block-vecchia)
3. [Argument Reference](#argument-reference)
   - [Core Arguments](#core-arguments)
   - [Vecchia Type Specific Arguments](#vecchia-type-specific-arguments)
   - [Advanced Arguments](#advanced-arguments)

---

## Installation Guide

### Prerequisites

Before installing VecchiaGP, ensure you have:

- **CMake** (version 3.20 or higher)
- **CUDA Toolkit** (for GPU acceleration)
- **gcc/g++** compilers (C++11 or higher)
- **R** (version 3.6.0 or higher) - only if using R interface
- **MPI** (required, for distributed computing)

### C++ Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/kaust-es/ParallelVecchiaGP.git
cd ParallelVecchiaGP
```

#### Step 2: Configure the Build

Run the configure script with the `-e` flag to enable examples:

```bash
./configure -e
```

**Configure Options:**
- `-e`: Enable building examples
- `-r`: Enable R support
- `-t`: Enable building tests
- `-v`: Verbose output
- `-w`: Show warnings
- `-h`: Show help message



**Note:** The configure script automatically downloads and builds all dependencies (MAGMA, KBLAS, BLASPP, LAPACK, GSL, NLOPT) in `installdir/_deps/`.

#### Step 3: Build the Project

```bash
./clean_build.sh
```

This compiles the C++ code and builds example executables in `bin/examples/`.

#### Step 4: Set Environment Variables

Add the following to your `~/.bashrc` or `~/.bash_profile`:

```bash
export PKG_CONFIG_PATH=$PWD/installdir/_deps/MAGMA/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$PWD/installdir/_deps/KBLAS/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$PWD/installdir/_deps/BLASPP/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$PWD/installdir/_deps/LAPACK/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$PWD/installdir/_deps/GSL/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=$PWD/installdir/_deps/NLOPT/lib/pkgconfig:$PWD/installdir/_deps/NLOPT/lib64/pkgconfig:$PKG_CONFIG_PATH
```

Then reload your shell:

```bash
source ~/.bashrc
```

#### Step 5: Verify Installation

Test the installation by running an example:

```bash
./bin/examples/Example_Parallel_Vecchia_Estimation --N=2000 --iTheta=1.5:0.1:0.5 --kernel=univariate_matern_stationary --performance --conditioning_size=300 --knn --block_size=1 --seed=0 --max_mle_iterations=1 --ncores=40 --VecchiaType=parallel --permutation=random --gpus=1
```

### R Package Installation

#### Step 1: Install R Dependencies

```R
install.packages("Rcpp")
install.packages("assertthat")
```

#### Step 2: Install VecchiaGP R Package

From the project root directory:

```bash
./configure -m -r
R CMD INSTALL . 
```

#### Step 3: Load the Package

```R
library(Vecchia)
```

### Troubleshooting

#### Issue: CMake Not Found

**Solution:**
```bash
sudo apt install cmake
# Or install from source if needed
```

#### Issue: CUDA Not Detected

**Solution:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### Issue: Libtool Not Found

**Solution:**
```bash
sudo apt install libtool libtool-bin
```

#### Issue: R Package Installation Fails

**Solution:**
```bash
sudo apt install r-base-dev
```

---

## Examples

### Parallel Vecchia

#### Example 1: Estimation Only (R)

```R
library(Vecchia)

# Step 1: Load or generate data
data_result <- load_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    initial_theta = c(1.5, 0.1, 0.5),  # variance, range, smoothness
    problem_size = 2000,
    seed = 0,
    block_size = 1,
    dimension = "2D",
    conditioning_size = 300,
    ncores = 40,
    permutation = "random",
    gpus = 1
)

# Step 2: Estimate parameters
model_result <- model_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    data = data_result,
    initial_theta = c(1.5, 0.1, 0.5),
    lb = c(0.01, 0.01, 0.01),  # lower bounds
    ub = c(3.0, 3.0, 3.0),     # upper bounds
    tol = 6,                    # tolerance = 1e-6
    mle_itr = 100,              # max MLE iterations
    conditioning_size = 300,
    ncores = 40,
    permutation = "random",
    gpus = 1
)

cat("Log-likelihood:", model_result$log_likelihood, "\n")
cat("Estimated theta:", model_result$estimated_theta, "\n")
```

#### Example 2: Estimation Only (C++)

```bash
./bin/examples/Example_Parallel_Vecchia_Estimation \
    --N=2000 \
    --iTheta=1.5:0.1:0.5 \
    --kernel=univariate_matern_stationary \
    --VecchiaType=parallel \
    --seed=0 \
    --conditioning_size=300 \
    --block_size=1 \
    --ncores=40 \
    --permutation=random \
    --gpus=1 \
    --max_mle_iterations=100 \
    --tolerance=6
```

#### Example 3: Full Workflow (Estimation + Prediction) (R)

```R
library(Vecchia)

# Step 1: Load data
data_result <- load_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    initial_theta = c(1.5, 0.1, 0.5),
    problem_size = 2000,
    seed = 0,
    block_size = 1,
    dimension = "2D",
    conditioning_size = 300,
    ncores = 40,
    permutation = "random",
    gpus = 1
)

# Step 2: Estimate parameters
model_result <- model_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    data = data_result,
    initial_theta = c(1.5, 0.1, 0.5),
    lb = c(0.01, 0.01, 0.01),
    ub = c(3.0, 3.0, 3.0),
    tol = 6,
    mle_itr = 100,
    conditioning_size = 300,
    ncores = 40,
    permutation = "random",
    gpus = 1
)

# Step 3: Predict at new locations
predicted_values <- predict_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    estimated_theta = model_result$estimated_theta,
    train_data = data_result,
    conditioning_size = 300,
    ncores = 40,
    permutation = "random",
    gpus = 1
)

cat("Number of predictions:", length(predicted_values), "\n")
```

### Block Vecchia

#### Example 1: Estimation Only (R)

```R
library(Vecchia)

# Step 1: Load data
data_result <- load_data(
    vecchia_type = "block",
    kernel = "univariate_matern_stationary",
    initial_theta = c(1.5, 0.1, 0.5),
    problem_size = 2000,
    seed = 0,
    block_size = 200,  # Number of clusters
    dimension = "2D",
    conditioning_size = 300,
    ncores = 40,
    permutation = "random",
    gpus = 0
)

# Step 2: Estimate parameters
model_result <- model_data(
    vecchia_type = "block",
    kernel = "univariate_matern_stationary",
    data = data_result,
    initial_theta = c(1.5, 0.1, 0.5),
    lb = c(0.01, 0.01, 0.01),
    ub = c(3.0, 3.0, 3.0),
    tol = 6,
    mle_itr = 100,
    block_size = 200,
    dimension = "2D",
    conditioning_size = 300,
    ncores = 40,
    permutation = "random",
    gpus = 0
)

cat("Log-likelihood:", model_result$log_likelihood, "\n")
cat("Estimated theta:", model_result$estimated_theta, "\n")
```

#### Example 2: Prediction Only (Using External Files) (R)

```R
library(Vecchia)

# Define file paths
train_locs_file <- "./simu_ds/prediction_data_0.014290_2.500000/LOC_1_train.csv"
test_locs_file <- "./simu_ds/prediction_data_0.014290_2.500000/LOC_1_test.csv"
train_data_file <- "./simu_ds/prediction_data_0.014290_2.500000/Z1_1_train.csv"
test_data_file <- "./simu_ds/prediction_data_0.014290_2.500000/Z1_1_test.csv"

# Step 1: Initialize hardware (minimal data loading)
data_result <- load_data(
    vecchia_type = "block",
    kernel = "univariate_matern_stationary",
    initial_theta = c(1.5, 0.1, 0.5),
    problem_size = 50,
    seed = 1,
    block_size = 200,
    dimension = "2D",
    conditioning_size = 300,
    ncores = 20,
    permutation = "random",
    gpus = 0
)

# Step 2: Predict using external files
prediction_theta <- c(1.5, 0.014290, 2.500000)  # variance, range, smoothness

predicted_values <- predict_data(
    vecchia_type = "block",
    kernel = "univariate_matern_stationary",
    estimated_theta = prediction_theta,
    block_size = 200,
    dimension = "2D",
    train_data = train_data_file,
    test_data = test_data_file,
    train_locs = train_locs_file,
    test_locs = test_locs_file,
    conditioning_size = 300,
    ncores = 20,
    permutation = "random",
    seed = 1,
    problem_size = 50,
    gpus = 0
)

cat("Number of predictions:", length(predicted_values), "\n")
```

#### Example 3: estimation (C++)

```bash
./bin/examples/Example_Parallel_Vecchia_Estimation --N=2000 --iTheta=1.5:0.1:0.5 --kernel=univariate_matern_stationary --performance --conditioning_size=300 --knn --block_size=1 --seed=0 --max_mle_iterations=1 --ncores=40 --VecchiaType=block --permutation=random --gpus=0
```

#### Example 4: prediction (C++)

```bash
./bin/examples/Example_Parallel_Vecchia_Estimation_Prediction  --train_locs=$train_locs --test_locs=$test_locs --train_data=$train_data --test_data=$test_data --N=50 --itheta=${sigma_str},${beta_str},${nu_str} --kernel=UnivariateMaternStationary --conditioning_size=$m --knn --block_size=$k --seed=$i --max_mle_iterations=1 --cores=20 --vecchiaType=block --permutation=random --gpus=0 --dim=3 --distance_metric=eg --scale_factor=1.0 --conditional_sim=1000 --kmeans_max_iter=50
```

### Scaled Block Vecchia

#### Example 1: Full Workflow (Estimation + Prediction) (R)

```R
library(Vecchia)

# Step 1: Load data for 8D Scaled Block Vecchia
data_result <- load_data(
    vecchia_type = "scaled_block",
    kernel = "univariate_matern_stationary",
    initial_theta = c(1.0, 0.001),  # variance, nugget (for Matern72)
    problem_size = 1000,
    seed = 0,
    block_size = 100,
    dimension = "8",  # 8-dimensional data
    distance_scale = c(0.05, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0),  # Per-dimension scaling
    nn_multiplier = 500,
    conditioning_size = 200,
    ncores = 20,
    permutation = "morton",
    kernel_type = "Matern72",  # Matern72 kernel
    gpus = 0
)

# Step 2: Estimate parameters
# Note: For Scaled Block, theta includes [variance, nugget, distance_scale[0], ..., distance_scale[7]]
model_result <- model_data(
    vecchia_type = "scaled_block",
    kernel = "univariate_matern_stationary",
    data = data_result,
    initial_theta = c(1.0, 0.001),  # Only variance and nugget (distance_scale auto-appended)
    lb = c(0.001, 1e-6, 5e-5, 5e-5, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001),  # 10 values
    ub = c(10, 0.01, 0.5, 0.5, 1, 10, 10, 10, 10, 10),  # 10 values
    tol = 6,
    mle_itr = 50,
    block_size = 100,
    dimension = "8",
    distance_scale = c(0.05, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0),
    nn_multiplier = 500,
    conditioning_size = 200,
    ncores = 20,
    permutation = "morton",
    kernel_type = "Matern72",
    gpus = 0
)

cat("Log-likelihood:", model_result$log_likelihood, "\n")
cat("Estimated theta (10 values):", model_result$estimated_theta, "\n")

# Step 3: Predict at new locations
predicted_values <- predict_data(
    vecchia_type = "scaled_block",
    kernel = "univariate_matern_stationary",
    estimated_theta = model_result$estimated_theta,
    block_size = 100,
    dimension = "8",
    train_data = data_result,
    distance_scale = c(0.05, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0),
    nn_multiplier = 500,
    conditioning_size = 200,
    ncores = 20,
    permutation = "morton",
    kernel_type = "Matern72",
    gpus = 0
)

cat("Number of predictions:", length(predicted_values), "\n")
```

#### Example 2: Full Workflow with MPI (C++)

```bash
mpirun -n 2 ./bin/examples/Example_Parallel_Vecchia_Estimation_Prediction --N=1000 --block_size=100 --distance_scale=0.05:0.05:0.1:1.0:1.0:1.0:1.0:1.0 --nn_multiplier=500 --iTheta=1.0:0.001 --kernel=univariate_matern_stationary --seed=0 --max_mle_iterations=1 --ncores=40 --VecchiaType=scaled_block --permutation=morton --gpus=0 --dim=8 --ncores=20 --kernel_type=Matern72 --conditioning_size=200 --tolerance=6
```

**Key Points for Scaled Block Vecchia:**
- Use `--dim=8` (or other integer) instead of `--dimension=2D`
- Use `--kernel_type=Matern72` (or Matern12, Matern32, Matern52, PowerExponential)
- Provide `--distance_scale` as comma/colon-separated values matching dimension
- `--iTheta` format: `variance:nugget` for Matern kernels, `variance:smoothness:nugget` for PowerExponential
- Use `--m_test` for test/prediction conditioning size (different from `--conditioning_size` for training)

---

## Argument Reference

### Core Arguments

#### `--N` or `--n`
- **Type:** Integer
- **Description:** Number of data points/locations
- **Example:** `--N=2000`
- **Default:** 0 (must be specified)

#### `--VecchiaType` or `--vecchiaType`
- **Type:** String
- **Values:** `parallel`, `block`, `scaled_block`
- **Description:** Type of Vecchia approximation to use
- **Example:** `--VecchiaType=parallel`

#### `--kernel`
- **Type:** String
- **Values:** `univariate_matern_stationary`, `Matern`
- **Description:** Covariance kernel type
- **Example:** `--kernel=univariate_matern_stationary`

#### `--iTheta` or `--itheta` or `--initial_theta`
- **Type:** Colon or comma-separated doubles
- **Description:** Initial parameter values
- **Format:**
  - **Parallel/Block Vecchia:** `variance:range:smoothness` (e.g., `1.5:0.1:0.5`)
  - **Scaled Block Vecchia (Matern):** `variance:nugget` (e.g., `1.0:0.001`)
  - **Scaled Block Vecchia (PowerExponential):** `variance:smoothness:nugget` (e.g., `1.0:0.5:0.001`)
- **Example:** `--iTheta=1.5:0.1:0.5`

#### `--seed`
- **Type:** Integer
- **Description:** Random seed for reproducibility
- **Example:** `--seed=42`
- **Default:** Current time (non-reproducible)

#### `--block_size`
- **Type:** Integer
- **Description:** 
  - **Parallel Vecchia:** Usually 1
  - **Block Vecchia:** Number of clusters (e.g., 200)
  - **Scaled Block Vecchia:** Number of blocks (e.g., 10000)
- **Example:** `--block_size=200`

#### `--conditioning_size`
- **Type:** Integer
- **Description:** Number of conditioning points for training/estimation
- **Example:** `--conditioning_size=300`
- **Default:** 100

#### `--m_test` or `--test_conditioning_size`
- **Type:** Integer
- **Description:** Number of conditioning points for test/prediction (Scaled Block only)
- **Example:** `--m_test=200`
- **Default:** 120

#### `--ncores` or `--cores`
- **Type:** Integer
- **Description:** Number of OpenMP threads
- **Example:** `--ncores=40`
- **Default:** 40

#### `--gpus`
- **Type:** Integer
- **Description:** Number of GPUs to use
- **Example:** `--gpus=1`
- **Default:** 0

#### `--max_mle_iterations`
- **Type:** Integer
- **Description:** Maximum number of MLE optimization iterations
- **Example:** `--max_mle_iterations=100`
- **Default:** 1

#### `--tolerance`
- **Type:** Integer
- **Description:** Tolerance exponent for optimization (e.g., 6 means 1e-6)
- **Example:** `--tolerance=6`
- **Default:** 5 (1e-5)

#### `--permutation`
- **Type:** String
- **Values:** `random`, `morton`, `kdtree`, `hilbert`, `mmd`
- **Description:** Spatial ordering/permutation method
- **Example:** `--permutation=morton`
- **Default:** `random`

### Vecchia Type Specific Arguments

#### For Parallel/Block Vecchia

##### `--dimension`
- **Type:** String
- **Values:** `2D`, `3D`, `ST`
- **Description:** Spatial dimension for Parallel/Block Vecchia
- **Example:** `--dimension=2D`

#### For Scaled Block Vecchia

##### `--dim`
- **Type:** Integer
- **Description:** Integer dimension for Scaled Block Vecchia (e.g., 8 for 8D)
- **Example:** `--dim=8`

##### `--kernel_type`
- **Type:** String
- **Values:** `Matern12`, `Matern32`, `Matern52`, `Matern72`, `PowerExponential`
- **Description:** Specific kernel type for Scaled Block Vecchia
- **Example:** `--kernel_type=Matern72`
- **Default:** `Matern72`
- **Note:** Determines parameter structure:
  - **Matern kernels:** `[variance, nugget] + [distance_scale per dimension]`
  - **PowerExponential:** `[variance, smoothness, nugget] + [distance_scale per dimension]`

##### `--distance_scale`
- **Type:** Colon or comma-separated doubles
- **Description:** Per-dimension distance scaling factors
- **Format:** Must match `--dim` (e.g., for `--dim=8`, provide 8 values)
- **Example:** `--distance_scale=0.05,0.05,0.1,1.0,1.0,1.0,1.0,1.0`
- **Default:** All 1.0 (uniform scaling)

##### `--nn_multiplier`
- **Type:** Integer
- **Description:** Multiplier for coarse-to-fine nearest neighbor search
- **Example:** `--nn_multiplier=500`
- **Default:** 400

##### `--clustering_method`
- **Type:** String
- **Values:** `random`, `kmeans++`
- **Description:** Clustering method for block formation
- **Example:** `--clustering_method=kmeans++`
- **Default:** `random`

##### `--kmeans_max_iter`
- **Type:** Integer
- **Description:** Maximum iterations for k-means clustering
- **Example:** `--kmeans_max_iter=50`
- **Default:** 50

##### `--num_total_points_test`
- **Type:** Integer
- **Description:** Total number of test points for prediction
- **Example:** `--num_total_points_test=1000`
- **Default:** 0

##### `--num_total_blocks_test`
- **Type:** Integer
- **Description:** Total number of test blocks for prediction
- **Example:** `--num_total_blocks_test=100`
- **Default:** 0

### Advanced Arguments

#### `--lower_bounds` or `--lowerBounds`
- **Type:** Colon or comma-separated doubles
- **Description:** Lower bounds for MLE optimization
- **Format:** Must match number of parameters
- **Example:** `--lower_bounds=0.01:0.01:0.01`
- **Default:** `0.01:0.01:0.01` (for 3-parameter kernels)

#### `--upper_bounds` or `--upperBounds`
- **Type:** Colon or comma-separated doubles
- **Description:** Upper bounds for MLE optimization
- **Format:** Must match number of parameters
- **Example:** `--upper_bounds=3.0:3.0:3.0`
- **Default:** `3.0:3.0:3.0` (for 3-parameter kernels)

#### `--data_path` or `--dataPath`
- **Type:** String (file path)
- **Description:** Path to CSV file with locations
- **Example:** `--data_path=./data/locations.csv`

#### `--observations_path` or `--observationsPath`
- **Type:** String (file path)
- **Description:** Path to CSV file with observations
- **Example:** `--observations_path=./data/observations.csv`

#### `--train_locs` or `--trainLocationsPath`
- **Type:** String (file path)
- **Description:** Path to CSV file with training locations (for prediction)
- **Example:** `--train_locs=./data/train_locs.csv`

#### `--test_locs` or `--testLocationsPath`
- **Type:** String (file path)
- **Description:** Path to CSV file with test locations (for prediction)
- **Example:** `--test_locs=./data/test_locs.csv`

#### `--train_data` or `--trainDataPath`
- **Type:** String (file path)
- **Description:** Path to CSV file with training data (for prediction)
- **Example:** `--train_data=./data/train_data.csv`

#### `--test_data` or `--testDataPath`
- **Type:** String (file path)
- **Description:** Path to CSV file with test data (for prediction)
- **Example:** `--test_data=./data/test_data.csv`

#### `--partition`
- **Type:** String
- **Values:** `linear`, `none`
- **Description:** Partition method for distributed Scaled Block Vecchia
- **Example:** `--partition=linear`
- **Default:** `linear`

#### `--distance_metric`
- **Type:** String
- **Values:** `euclidean` (eg), `great_circle` (gcd)
- **Description:** Distance metric for spatial calculations
- **Example:** `--distance_metric=euclidean`
- **Default:** `euclidean`

#### `--precision`
- **Type:** String
- **Values:** `single`, `double`, `mixed`
- **Description:** Numerical precision for computations
- **Example:** `--precision=double`
- **Default:** `double`

#### `--verbose`
- **Type:** String
- **Values:** `quiet`, `standard`, `detailed`
- **Description:** Verbosity level for output
- **Example:** `--verbose=detailed`
- **Default:** `standard`

#### `--knn`
- **Type:** Flag (no value)
- **Description:** Enable k-nearest neighbor search (default: enabled)
- **Example:** `--knn`

#### `--performance`
- **Type:** Flag (no value)
- **Description:** Performance mode (single iteration, observations=0)
- **Example:** `--performance`

#### `--help`
- **Type:** Flag (no value)
- **Description:** Display help message with all available arguments
- **Example:** `--help`

---

## R Interface Arguments

### `load_data()` Function

**Required:**
- `vecchia_type`: `"parallel"`, `"block"`, or `"scaled_block"`
- `kernel`: `"univariate_matern_stationary"` or `"Matern"`
- `initial_theta`: Numeric vector of initial parameters

**Optional:**
- `problem_size`: Integer (default: 2000)
- `seed`: Integer (default: 123)
- `block_size`: Integer (default: 200)
- `dimension`: String `"2D"`, `"3D"`, or integer as string `"8"` for scaled_block
- `data_path`: String (empty for synthetic data)
- `distance_scale`: Numeric vector (for scaled_block)
- `nn_multiplier`: Integer (for scaled_block, default: 400)
- `conditioning_size`: Integer (default: 100)
- `ncores`: Integer (default: 40)
- `permutation`: String (default: `"random"`)
- `kernel_type`: String (for scaled_block, default: `"Matern72"`)
- `gpus`: Integer (default: 0)

### `model_data()` Function

**Required:**
- `vecchia_type`: String
- `kernel`: String
- `lb`: Numeric vector (lower bounds)
- `ub`: Numeric vector (upper bounds)

**Optional:**
- `data`: List from `load_data()` or NULL
- `matrix`: Numeric vector (pre-computed covariance matrix)
- `x`, `y`: Numeric vectors (coordinates, if data is NULL)
- `initial_theta`: Numeric vector
- `tol`: Integer (tolerance exponent, default: 6)
- `mle_itr`: Integer (max iterations, default: 100)
- All other arguments same as `load_data()`

### `predict_data()` Function

**Required:**
- `vecchia_type`: String
- `kernel`: String
- `estimated_theta`: Numeric vector

**Optional:**
- `train_data`: List from `load_data()` or file path string
- `test_data`: File path string or NULL
- `train_locs`: File path string
- `test_locs`: File path string
- All other arguments same as `load_data()`

---

## Notes

1. **Parameter Format:** Use colon (`:`) or comma (`,`) as separators for multi-value arguments (e.g., `--iTheta=1.5:0.1:0.5` or `--iTheta=1.5,0.1,0.5`)

2. **Scaled Block Vecchia:** Requires different parameter structure:
   - Use `--dim=8` (integer) instead of `--dimension=2D`
   - Provide `--distance_scale` matching dimension
   - Use `--kernel_type` to specify kernel variant
   - `--iTheta` format depends on kernel type

3. **MPI Usage:** For distributed Scaled Block Vecchia, use `mpirun -n <num_processes>` before the executable

4. **Data Reuse:** In R, pass `data_result` from `load_data()` to `model_data()` and `predict_data()` to avoid reloading data

5. **File Formats:** CSV files should have one value per line (no headers)

---


**Authors:** Qilong Pan, Sohayla Khaled, Mahmoud ElKarargy, Sameh Abdulah (Maintainer)  
**Contact:** sameh.abdulah@kaust.edu.sa  
**License:** BSD 3-Clause License  
**Copyright:** (c) 2017-2025 King Abdullah University of Science and Technology


