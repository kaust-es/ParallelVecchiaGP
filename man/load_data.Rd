
% Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
% All rights reserved.
% Vecchia is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

% @file load_data.Rd
% @brief roxygen2 documentation for the R Interface of R_VecchiaLoadData function.
% @version 1.0.0
% @author Generated for R wrapper
% @date 2025-01-01

\name{load_data}
\alias{load_data}
\title{Load Data function}

\usage{
load_data(vecchia_type = "block", kernel = "univariate_matern_stationary",
initial_theta = c(1.0, 0.5, 0.1), distance_matrix = "euclidean",
problem_size = 2000, seed = 123, block_size = 200,
dimension = "2D", data_path = "", distance_scale = NULL,
nn_multiplier = NULL, conditioning_size = NULL, ncores = NULL,
permutation = NULL, kernel_type = NULL)
}

\arguments{
  \item{vecchia_type}{A string specifies the Vecchia approximation type, either "parallel" or "block" or "scaled_block". Default is "block".}
  \item{kernel}{A string specifies the kernel to use. Available kernels include:
    \itemize{
      \item "BivariateMaternFlexible"
      \item "BivariateMaternParsimonious"
      \item "BivariateSpacetimeMaternStationary"
      \item "TrivariateMaternParsimonious"
      \item "UnivariateExpNonGaussian"
      \item "UnivariateMaternDbeta"
      \item "UnivariateMaternDdbetaBeta"
      \item "UnivariateMaternDdbetaNu"
      \item "UnivariateMaternDdnuNu"
      \item "UnivariateMaternDdsigmaSquare"
      \item "UnivariateMaternDdsigmaSquareBeta"
      \item "UnivariateMaternDdsigmaSquareNu"
      \item "UnivariateMaternDnu"
      \item "UnivariateMaternDsigmaSquare"
      \item "UnivariateMaternNonGaussian"
      \item "UnivariateMaternNuggetsStationary"
      \item "UnivariateMaternStationary" (or "univariate_matern_stationary")
      \item "UnivariatePowExpStationary"
      \item "UnivariateSpacetimeMaternStationary"
    }
  }
  \item{initial_theta}{A numeric vector represents the initial parameter values for the kernel.}
  \item{distance_matrix}{A string specifies the distance metric, either "euclidean" or "great_circle". Default is "euclidean".}
  \item{problem_size}{A numeric value represents the size of the problem to load.}
  \item{seed}{A numeric value specifies the seed for random number generation. Default is 123.}
  \item{block_size}{A numeric value represents the block size for block Vecchia. Default is 200.}
  \item{dimension}{A string specifies the data dimension. For scaled_block, can be an integer string (e.g., "8") or one of "2D", "3D", or "ST". For other types, must be "2D", "3D", or "ST". Default is "2D".}
  \item{data_path}{A string specifies the path for data storage. Default is "".}
  \item{distance_scale}{A numeric vector represents the distance scale vector. For scaled_block, one value per dimension. Default is NULL.}
  \item{nn_multiplier}{A numeric value represents the nearest neighbor multiplier for scaled_block. Default is NULL (uses 400).}
  \item{conditioning_size}{A numeric value represents the conditioning size. Default is NULL (uses 100).}
  \item{ncores}{A numeric value specifies the number of CPU cores. Default is NULL (uses system default).}
  \item{permutation}{A string specifies the permutation method, either "random", "morton", "kdtree", "hilbert", or "mmd". Default is NULL (uses "random").}
  \item{kernel_type}{A string specifies the kernel type for scaled_block, either "Matern12", "Matern32", "Matern52", "Matern72", or "PowerExponential". Default is NULL (uses "Matern72").}
}

\value{
A list contains the loaded data including x, y, m (measurements), data_path, seed, vecchia_type, vecchia_data (XPtr), and vecchia_hardware (XPtr).
}

\description{
This function loads data into a VecchiaGBData object using the provided configuration and computational settings.
}

\examples{
\dontrun{
vecchia_type <- "block"
kernel <- "univariate_matern_stationary"
initial_theta <- c(1.0, 0.5, 0.1)
problem_size <- 1000
seed <- 123
block_size <- 200
dimension <- "2D"

data_result <- load_data(vecchia_type=vecchia_type, kernel=kernel,
initial_theta=initial_theta, problem_size=problem_size, seed=seed,
block_size=block_size, dimension=dimension)
}
}
