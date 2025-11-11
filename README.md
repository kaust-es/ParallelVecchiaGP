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

- **Multiple Vecchia Types**: Supports `block`, `parallel`, and `scaled_block` Vecchia approximations
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
- **MPI Support**: Optional distributed computing support for scaled block Vecchia
- **R Integration**: Full R interface via Rcpp for seamless integration with R workflows

---

## Installation

### Requirements

To build and run this software, you will need:

1. **CMake** (version 3.20 or higher) - Required for building the project
2. **wget** - For downloading dependencies
3. **curl** - For downloading dependencies
4. **gcc** and **g++** compilers (C++11 or higher)
5. **autoconf** and **automake** - For building dependencies
6. **libtool** - For building dependencies
7. **CUDA Toolkit** - For GPU acceleration
8. **R** (version 3.6.0 or higher) - Only if you plan on using the R functionality
9. **Rcpp** (>= 1.0.9) - R package for C++ integration
10. **Git** (>= 2.0.0) - For cloning the repository

### C++ Installation

To install the `Vecchia` project locally (C++ version), run the following commands in your terminal:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kaust-es/ParallelVecchiaGP.git
   cd ParallelVecchiaGP
   ```

2. **Run the configure script** (use the `-h` flag for help to see supported options):
   ```bash
   ./configure -e -m
   ```
   This step is **not required** when using R installation.

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

**Solution**: Install CMake:
```sh
sudo apt install cmake
```

Or install from source if needed (the configure script can do this automatically).

### 2. Missing Libtool

If you encounter the following error:
```
./autogen.sh: line 17: libtool: command not found
./autogen.sh: line 20: glibtool: command not found
```

**Solution**: Install Libtool:
```sh
sudo apt install libtool libtool-bin
```

Alternatively, install Libtool locally:
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

If R package installation fails:

**Solution**: Ensure all R dependencies are installed and R development tools are available:
```sh
sudo apt install r-base-dev
```

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
    block_size = 1,
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
    block_size = 200,
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
    block_size = 200,
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
./examples/Example_Parallel_Vecchia_Estimation \
    --N=2000 \
    --iTheta=1.5:0.1:0.5 \
    --kernel=univariate_matern_stationary \
    --conditioning_size=300 \
    --knn \
    --block_size=1 \
    --seed=0 \
    --max_mle_iterations=100 \
    --ncores=40 \
    --VecchiaType=parallel \
    --permutation=random \
    --gpus=1
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
- `block_size` (integer): Block size for block Vecchia (default: 200)
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
- `block_size` (integer): Block size
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
- `block_size` (integer): Block size
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

The project automatically downloads and builds the following dependencies:

- **MAGMA**: Matrix Algebra on GPU and Multicore Architectures
- **KBLAS**: Kernel BLAS library for GPU
- **BLASPP**: C++ API for BLAS
- **LAPACK**: Linear Algebra Package
- **GSL**: GNU Scientific Library
- **NLOPT**: Nonlinear optimization library
- **GFortran**: GNU Fortran compiler
- **CUDA**: NVIDIA CUDA Toolkit (for GPU acceleration)

All dependencies are installed locally in `installdir/_deps/` during the configure step.

---

## Configuration Options

### CMake Options

When building with CMake, you can configure:

- `USE_MPI` (OFF): Enable MPI for distributed Scaled Block Vecchia
- `BUILD_TESTS` (OFF): Build test suite
- `BUILD_EXAMPLES` (ON): Build example programs
- `USE_R` (OFF): Enable R and Rcpp integration

Example:
```bash
cmake -DUSE_MPI=ON -DBUILD_TESTS=ON ..
```

### Configure Script Options

Run `./configure -h` to see all available options.

Common options:
- `-e`: Enable examples
- `-r`: Enable R support
- `-m`: Enable MPI support
- `-t`: Enable building tests
- `-v`: Enable verbose output
- `-w`: Enable showing warnings
- `-i [path]`: Specify installation path (default: `./installdir/_deps/`)

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
