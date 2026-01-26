# VecchiaGP: Parallel Gaussian Process Kernels for Large-Scale Geospatial Analysis

The **VecchiaGP** project develops scalable, high-performance implementations of the Vecchia approximation method for Gaussian Processes (GPs), targeting large-scale geospatial data analysis on modern computing architectures.

Gaussian Processes are widely used in environmental and climate modeling, but their computational complexity traditionally scales cubically with the number of data points, making them impractical for large datasets. To address this, the Vecchia approximation reduces the complexity by approximating the joint GP distribution as a product of conditional distributions, enabling much more efficient computations.

By leveraging parallel GPU implementations, the VecchiaGP framework significantly accelerates processing by performing batched matrix computations, making it possible to handle larger datasets with improved efficiency and accuracy.

---

## Table of Contents

- [Scientific Background](#scientific-background)
- [Project Purpose](#project-purpose)
- [Features](#features)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [C++ Installation](#c-installation)
  - [R Package Installation](#r-package-installation)
  - [Common Installation Errors](#common-installation-errors-and-solutions)
- [Usage](#usage)
  - [R Interface](#r-interface)
  - [C++ Examples](#c-examples)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Configuration Options](#configuration-options)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## Scientific Background

- **Scalability**: Traditional GP models become computationally infeasible for large datasets due to the cubic scaling of matrix operations. The Vecchia approximation reduces this complexity, providing a scalable alternative suitable for exascale systems.

- **Parallelism**: Implementing Vecchia approximations in a parallel computing environment — particularly using GPUs — allows multiple conditional distributions to be computed simultaneously, improving throughput and reducing overall computation time.

- **R Integration**: By exposing the GPU kernels to R through Rcpp, the project ensures that scientific users can access high-performance routines directly within R, a language widely used for statistical computing and data analysis.

---

## Project Purpose

The primary objective of this project is to demonstrate the use of state-of-the-art statistical algorithms, such as the Vecchia approximation, implemented on high-performance computing (HPC) hardware. This enables routine use of large geospatial datasets and statistical models by environmental scientists, climate researchers, and the broader scientific community.

The approach focuses not on introducing new datasets or algorithms, but on operationalizing existing state-of-the-art methods within a scalable, efficient, and user-accessible software framework.

---

## Features

- **Multiple Vecchia Types**: 
  - `block` (PARALLEL_BLOCK_VECCHIA_GP) - Available by default
  - `parallel` (PARALLEL_VECCHIA_GP) - Requires KBLAS (use `./configure -e -k`, CUDA 11.x only)
  - `scaled_block` (PARALLEL_SCALED_BLOCK_VECCHIA_GP) - Available by default
- **GPU Acceleration**: CUDA-based parallel implementations for high-performance computing
- **Multiple Kernels**: Supports various covariance kernels including:
  - Univariate Matern Stationary
  - Matern kernels with different smoothness parameters (Matern32, Matern52, Matern72)
- **Flexible Data Handling**: 
  - Synthetic data generation
  - External data loading from CSV files
  - Support for 2D, 3D, and high-dimensional (8D+) spatial data
- **Parameter Estimation**: Maximum Likelihood Estimation (MLE) with configurable optimization bounds
- **Prediction**: Efficient prediction at new locations using estimated parameters
- **Spatial Ordering**: Multiple permutation strategies (Morton, Random) for neighbor selection
- **MPI Support**: Required distributed computing support
- **R Integration**: Full R interface via Rcpp for seamless integration with R workflows

---

## Installation

### Requirements

To build and run this software, you will need:

1. **CMake** (version 3.20 or higher) - Required for building the project
2. **wget** - For downloading dependencies
3. **curl** - For downloading dependencies
4. **gcc** and **g++** compilers (version 10.2.0 or higher, includes OpenMP support)
5. **autoconf** and **automake** - For building dependencies
6. **libtool** - For building dependencies
7. **CUDA Toolkit** (version 11.4 or later) - For GPU acceleration
8. **Intel MKL** (version 2022.2.1 or later) - Provides optimized BLAS/LAPACK
9. **MPI** (e.g., OpenMPI) - Required for distributed computing
10. **R** (version 3.6.0 or higher) - Only if you plan on using the R functionality
11. **Rcpp** (>= 1.0.9) - R package for C++ integration
12. **Git** (>= 2.0.0) - For cloning the repository

**Note:** The following dependencies are **automatically downloaded and built** by the configure script:
- MAGMA (Matrix Algebra on GPU, version 2.7.0 or later)
- GSL (GNU Scientific Library, version 2.6 or later)
- NLopt (Nonlinear optimization library, version 2.7.1 or later)
- KBLAS, BLASPP, LAPACK

### C++ Installation

To install the `Vecchia` project locally (C++ version), run the following commands in your terminal:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kaust-es/ParallelVecchiaGP.git
   cd ParallelVecchiaGP
   ```

2. **Run the configure script** to prepare the build environment:
   ```bash
   ./configure -e
   ```
   
   **What does the configure script do?**
   - Checks for required dependencies (CMake, CUDA, GCC, etc.)
   - Sets up CMake build configuration with appropriate flags
   - Downloads and builds required libraries (GSL, MAGMA, NLopt, etc.) into a local `installdir/_deps/` directory
   - Configures install paths to avoid requiring root/sudo permissions
   
   **Common options:**
   - `-e`: Enable building examples (recommended for testing)
   - `-k`: Enable KBLAS support (for PARALLEL_VECCHIA_GP method, requires CUDA 11.x)
   - `-h`: Show all available options
   - `-r`: Enable R support
   - `-v`: Enable verbose output
   
   **Note:** By default, KBLAS is disabled. Use `-k` flag to enable PARALLEL_VECCHIA_GP method.
   
   This step is **not required** when using R installation.
   **Note:** MPI is required and must be installed on your system before running configure.

3. **Build the project** (use the `-h` flag for help):
   ```bash
   ./clean_build.sh
   ```
   This step is **not required** when using R installation.

4. **Export dependency paths** to your `.bashrc` file:
   ```bash
   export PKG_CONFIG_PATH=$PWD/installdir/_deps/MAGMA/lib/pkgconfig:$PKG_CONFIG_PATH
   export PKG_CONFIG_PATH=$PWD/installdir/_deps/KBLAS/lib/pkgconfig:$PKG_CONFIG_PATH
   export PKG_CONFIG_PATH=$PWD/installdir/_deps/BLASPP/lib/pkgconfig:$PKG_CONFIG_PATH
   export PKG_CONFIG_PATH=$PWD/installdir/_deps/LAPACK/lib/pkgconfig:$PKG_CONFIG_PATH
   export PKG_CONFIG_PATH=$PWD/installdir/_deps/GSL/lib/pkgconfig:$PKG_CONFIG_PATH
   export PKG_CONFIG_PATH=$PWD/installdir/_deps/NLOPT/lib/pkgconfig:$PWD/installdir/_deps/NLOPT/lib64/pkgconfig:$PKG_CONFIG_PATH
   ```

   Then reload your shell configuration:
   ```bash
   source ~/.bashrc
   ```

Now, you can use the pkg-config executable to collect compiler and linker flags for VecchiaGP.

### R Package Installation

1. **Install required R packages**:
   ```R
   install.packages("Rcpp")
   install.packages("assertthat")
   ```

2. **Install the Vecchia R package**:
   ```bash
   R CMD INSTALL . --configure-args="-r"
   ```

   Make sure your current directory is the VecchiaGP project root.

3. **Load the package in R**:
   ```R
   library(Vecchia)
   ```

---

## Common Installation Errors and Solutions

### 1. Missing CMake

The installation requires **CMake** version 3.20 or higher. Ensure it is installed on your system before proceeding.

**Solution**: The configure script can automatically build CMake locally if it's not available or if the version is too old. Alternatively, you can install it manually from your package manager or from source.

### 2. Missing Libtool

If you encounter the following error:
```
./autogen.sh: line 17: libtool: command not found
./autogen.sh: line 20: glibtool: command not found
```

**Solution**: Install Libtool locally:
```sh
wget http://ftpmirror.gnu.org/libtool/libtool-2.4.7.tar.gz
tar -xvzf libtool-2.4.7.tar.gz
cd libtool-2.4.7
./configure --prefix=$HOME/local
make
make install
```

Then update your environment variables:
```sh
export PATH=$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

After this, restart your terminal or run `source ~/.bashrc` to apply the changes.

### 3. CUDA Not Found

If CUDA is not detected during configuration:

**Solution**: Ensure CUDA is properly installed and the `CUDA_HOME` environment variable is set:
```sh
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 4. R Package Installation Issues

If R package installation fails, ensure all R dependencies are installed and R development tools are available (r-base-dev).

---

## Usage

### R Interface

The R interface provides three main functions: `load_data()`, `model_data()`, and `predict_data()`.

#### Basic Workflow

```R
library(Vecchia)

# Step 1: Load or generate data
data_result <- load_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    initial_theta = c(1.5, 0.1, 0.5),  # variance, range, smoothness
    problem_size = 2000,
    seed = 0,
    block_count = 1,  # Number of blocks (1 for parallel Vecchia)
    dimension = "2D",
    conditioning_size = 300,
    ncores = 40,
    permutation = "random"
)

# Step 2: Estimate parameters
model_result <- model_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    data = data_result,
    initial_theta = c(1.5, 0.1, 0.5),
    lb = c(0.01, 0.01, 0.01),  # lower bounds
    ub = c(3.0, 3.0, 3.0),     # upper bounds
    tol = 6,                    # tolerance (1e-6)
    mle_itr = 100,              # max MLE iterations
    conditioning_size = 300,
    ncores = 40,
    permutation = "random"
)

# Step 3: Predict at new locations
predicted_values <- predict_data(
    vecchia_type = "parallel",
    kernel = "univariate_matern_stationary",
    estimated_theta = model_result$estimated_theta,
    train_data = data_result,
    conditioning_size = 300,
    ncores = 40,
    permutation = "random"
)
```

#### Using External Data Files

```R
# Load data from CSV files
data_result <- load_data(
    vecchia_type = "block",
    kernel = "univariate_matern_stationary",
    initial_theta = c(1.5, 0.1, 0.5),
    data_path = "./data/my_data.csv",
    dimension = "2D",
    block_count = 200,  # Number of blocks
    conditioning_size = 300
)

# Prediction with external test data
predicted_values <- predict_data(
    vecchia_type = "block",
    kernel = "univariate_matern_stationary",
    estimated_theta = c(1.5, 0.014290, 2.500000),
    train_data = "./data/train_data.csv",
    test_data = "./data/test_data.csv",
    train_locs = "./data/train_locs.csv",
    test_locs = "./data/test_locs.csv",
    block_count = 200,  # Number of blocks
    conditioning_size = 300,
    ncores = 20
)
```

### C++ Examples

C++ examples are available in the `examples/` directory. To build and run:

```bash
# Build examples (if not already built)
cd bin
cmake --build . --target Example_Parallel_Vecchia_Estimation

# Run example (from bin directory)
./examples/Example_Parallel_Vecchia_Estimation --N=20000 --iTheta=1.5:0.1:0.5 --kernel=univariate_matern_stationary --performance --conditioning_size=300 --knn --block_count=1 --seed=0 --max_mle_iterations=1 --ncores=40 --VecchiaType=parallel --permutation=random --gpus=1

```

Or from the project root:
```bash
./bin/examples/Example_Parallel_Vecchia_Estimation [options...]
```

See the `examples/` directory for more C++ examples:
- `ParallelVecchiaEstimation.cpp` - Parallel Vecchia estimation
- `ParallelVecchiaEstimationPrediction.cpp` - Estimation and prediction
- R examples in `examples/*.R` - R interface examples

---

## API Documentation

### R Functions

#### `load_data()`

Load or generate data for Vecchia approximation.

**Parameters:**
- `vecchia_type` (string): Type of Vecchia approximation - `"block"`, `"parallel"`, or `"scaled_block"`
- `kernel` (string): Covariance kernel type - `"univariate_matern_stationary"` or `"Matern"`
- `initial_theta` (numeric vector): Initial parameter values [variance, range, smoothness] or [variance, nugget, ...]
- `distance_matrix` (string): Distance metric - `"euclidean"` (default)
- `problem_size` (integer): Number of data points (default: 2000)
- `seed` (integer): Random seed for reproducibility (default: 123)
- `block_count` (integer): Number of blocks for block Vecchia (default: 200).
- `dimension` (string): Spatial dimension - `"2D"`, `"3D"`, or `"8"` (for scaled_block)
- `data_path` (string): Path to CSV file with external data (empty string for synthetic data)
- `distance_scale` (numeric vector, optional): Distance scaling factors for scaled_block
- `nn_multiplier` (integer, optional): Multiplier for nearest neighbors
- `conditioning_size` (integer, optional): Number of conditioning points
- `ncores` (integer, optional): Number of CPU cores
- `permutation` (string, optional): Permutation strategy - `"morton"` or `"random"`
- `kernel_type` (string, optional): Kernel type - `"Matern32"`, `"Matern52"`, `"Matern72"`

**Returns:** List containing:
- `m`: Measurements/observations
- `x`: X coordinates
- `y`: Y coordinates
- Internal data structures for use with `model_data()` and `predict_data()`

#### `model_data()`

Estimate parameters using Maximum Likelihood Estimation (MLE).

**Parameters:**
- `vecchia_type` (string): Type of Vecchia approximation
- `kernel` (string): Covariance kernel type
- `distance_matrix` (string): Distance metric
- `lb` (numeric vector): Lower bounds for optimization
- `ub` (numeric vector): Upper bounds for optimization
- `tol` (numeric): Tolerance exponent (e.g., 6 means 1e-6)
- `mle_itr` (integer): Maximum MLE iterations (default: 100)
- `block_count` (integer): Number of blocks
- `dimension` (string): Spatial dimension
- `data` (list or NULL): Data result from `load_data()` or NULL
- `matrix` (numeric vector, optional): Pre-computed covariance matrix
- `x`, `y` (numeric vectors, optional): Coordinates (if data is NULL)
- `initial_theta` (numeric vector, optional): Initial parameter values
- `distance_scale` (numeric vector, optional): Distance scaling factors
- `nn_multiplier` (integer, optional): Nearest neighbor multiplier
- `conditioning_size` (integer, optional): Number of conditioning points
- `ncores` (integer, optional): Number of CPU cores
- `permutation` (string, optional): Permutation strategy
- `kernel_type` (string, optional): Kernel type
- `seed` (integer, optional): Random seed

**Returns:** List containing:
- `log_likelihood`: Optimal log-likelihood value
- `estimated_theta`: Estimated parameter vector

#### `predict_data()`

Perform prediction at new locations.

**Parameters:**
- `vecchia_type` (string): Type of Vecchia approximation
- `kernel` (string): Covariance kernel type
- `distance_matrix` (string): Distance metric
- `estimated_theta` (numeric vector): Estimated parameters from `model_data()`
- `block_count` (integer): Number of blocks
- `dimension` (string): Spatial dimension
- `train_data` (list or string): Training data from `load_data()` or file path
- `test_data` (string or NULL): Test data file path or NULL
- `train_locs` (string, optional): Training locations file path
- `test_locs` (string, optional): Test locations file path
- `distance_scale` (numeric vector, optional): Distance scaling factors
- `nn_multiplier` (integer, optional): Nearest neighbor multiplier
- `conditioning_size` (integer, optional): Number of conditioning points
- `ncores` (integer, optional): Number of CPU cores
- `permutation` (string, optional): Permutation strategy
- `kernel_type` (string, optional): Kernel type
- `seed` (integer, optional): Random seed
- `problem_size` (integer, optional): Problem size

**Returns:** Numeric vector of predicted values

---

## Project Structure

```
VecchiaGP/
├── bin/                    # Build directory
├── cmake/                  # CMake modules and find scripts
├── examples/               # Example scripts and C++ programs
│   ├── *.R                 # R example scripts
│   └── *.cpp               # C++ example programs
├── inst/                   # Installation files
│   └── include/            # Header files
├── installdir/             # Installed dependencies
├── prerequisites/          # Prerequisites and libraries
├── R/                      # R package files
├── src/                    # Source code
│   ├── api/                # Main API
│   ├── conditioning-updater/
│   ├── configurations/     # Configuration management
│   ├── cuda-kernels/       # CUDA kernel implementations
│   ├── data-clustering/    # Data clustering algorithms
│   ├── data-generators/    # Synthetic data generation
│   ├── data-loader/        # Data loading utilities
│   ├── data-units/         # Data structures
│   ├── estimators/         # Parameter estimation
│   ├── hardware/           # Hardware abstraction
│   ├── helpers/            # Helper functions
│   ├── kernels/            # Covariance kernels
│   ├── predictors/         # Prediction algorithms
│   └── Rcpp-adapters/      # R interface adapters
├── tests/                  # Test suite
├── CMakeLists.txt          # Main CMake configuration
├── configure               # Configuration script
├── clean_build.sh          # Build script
└── README.md               # This file
```

---

## Dependencies

### System Dependencies (Pre-installed Required)

These must be installed on your system before running configure:

- **GCC/G++** (10.2.0 or later) - Provides C/C++ compiler and OpenMP support
- **Intel MKL** (2022.2.1 or later) - Provides optimized BLAS/LAPACK implementations
- **CUDA Toolkit** (11.4 or later) - For GPU acceleration
- **MPI** (e.g., OpenMPI) - For distributed computing
- **CMake** (3.20 or higher) - Build system
- **Git** - Version control (for cloning repository)

### Automatically Built Dependencies

The configure script automatically downloads and builds these locally (no root access needed):

- **MAGMA** (2.7.0+) - Matrix Algebra on GPU and Multicore Architectures
- **KBLAS** - Kernel BLAS library for GPU (only when using `-k` flag, CUDA 11.x required)
- **BLASPP** - C++ API for BLAS
- **LAPACK** - Linear Algebra Package
- **GSL** (2.6+) - GNU Scientific Library
- **NLOPT** (2.7.1+) - Nonlinear optimization library
- **GFortran** - GNU Fortran compiler (if not available)

**Note:** KBLAS is only downloaded and built when you use `./configure -e -k`. It is required for the PARALLEL_VECCHIA_GP method but is incompatible with CUDA 12+.

All automatically-built dependencies are installed locally in `installdir/_deps/` during the configure step.

---

## Configuration Options

### CMake Options

When building with CMake, you can configure:

- `BUILD_TESTS` (OFF): Build test suite
- `BUILD_EXAMPLES` (ON): Build example programs
- `USE_R` (OFF): Enable R and Rcpp integration
- `USE_KBLAS` (OFF): Enable KBLAS for PARALLEL_VECCHIA_GP method (CUDA 11.x only)

Example:
```bash
cmake -DBUILD_TESTS=ON .. -DUSE_KBLAS=ON ..
```

### Configure Script Options

The `configure` script is a convenience wrapper around CMake that:
- Automatically detects and validates system dependencies
- Downloads and builds required libraries locally (no root access needed)
- Sets up proper installation paths in `./installdir/_deps/`
- Generates the build configuration in the `bin/` directory

Run `./configure -h` to see all available options.

Common options:
- `-e`: Enable building examples (recommended for testing)
- `-k`: Enable KBLAS support (required for PARALLEL_VECCHIA_GP, works with CUDA 11.x only)
- `-r`: Enable R support
- `-t`: Enable building tests
- `-v`: Enable verbose output
- `-w`: Enable showing warnings
- `-i [path]`: Specify custom installation path (default: `./installdir/_deps/`)

**Important:** KBLAS is disabled by default. If you need the PARALLEL_VECCHIA_GP method:
- With CUDA 11.x: Use `./configure -e -k` to enable KBLAS
- With CUDA 12.0+: KBLAS is incompatible - use PARALLEL_BLOCK_VECCHIA_GP or PARALLEL_SCALED_BLOCK_VECCHIA_GP instead

**Note:** The configure script handles all dependency management automatically. You do not need to manually install GSL, MAGMA, NLOPT, or other mathematical libraries - they will be downloaded and built locally.

---

## Examples

### Example 1: Parallel Vecchia Estimation

See `examples/parallel_estimation_example.R` for a complete example of:
- Loading/generating data
- Parameter estimation with MLE
- Using parallel Vecchia approximation

### Example 2: Scaled Block Vecchia (Full Workflow)

See `examples/scaled_block_full_example.R` for:
- 8-dimensional scaled block Vecchia
- Complete estimation and prediction workflow
- Using Morton ordering

### Example 3: Block Prediction Only

See `examples/block_prediction_example.R` for:
- Prediction-only mode using external files
- Loading train/test data from CSV files
- Using pre-estimated parameters

### Example 4: C++ Command-Line Usage

See `examples/ParallelVecchiaEstimation.cpp` and `examples/ParallelVecchiaEstimationPrediction.cpp` for C++ program examples.

---

## Contributing

This project is developed by the STSDS group at King Abdullah University of Science and Technology (KAUST).

**Authors:**
- Qilong Pan
- Sohayla Khaled
- Mahmoud ElKarargy
- Sameh Abdulah (Maintainer)

**Contact:**
- Maintainer: Sameh Abdulah <sameh.abdulah@kaust.edu.sa>
- GitHub: https://github.com/kaust-es/ParallelVecchiaGP/
- Issues: https://github.com/kaust-es/ParallelVecchiaGP/issues

---

## License

[BSD 3-Clause License](LICENSE)

Copyright (c) 2017-2025 King Abdullah University of Science and Technology, All rights reserved.

VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).
