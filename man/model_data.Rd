
% Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
% All rights reserved.
% Vecchia is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

% @file model_data.Rd
% @brief roxygen2 documentation for the R Interface of R_VecchiaModelData function.
% @version 1.0.0
% @author Generated for R wrapper
% @date 2025-01-01

\name{model_data}
\alias{model_data}
\title{Model Data function}

\usage{
model_data(vecchia_type = "block", kernel = "univariate_matern_stationary",
distance_matrix = "euclidean", lb = c(0.01, 0.01, 0.01),
ub = c(3.0, 3.0, 3.0), tol = 4.0, mle_itr = 100,
block_size = 200, dimension = "2D", data = NULL, matrix = NULL,
x = NULL, y = NULL, initial_theta = NULL, distance_scale = NULL,
nn_multiplier = NULL, conditioning_size = NULL, ncores = NULL,
permutation = NULL, kernel_type = NULL, seed = NULL)
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
      \item "UnivariateMaternStationary"
      \item "UnivariatePowExpStationary"
      \item "UnivariateSpacetimeMaternStationary"
      \item "Matern"
    }
  }
  \item{distance_matrix}{A string specifies the distance metric, either "euclidean" or "great_circle". Default is "euclidean".}
  \item{lb}{A numeric vector represents the lower bound for the computation.}
  \item{ub}{A numeric vector represents the upper bound for the computation.}
  \item{tol}{A numeric value specifies the tolerance for the computation. The final value is computed as 10^(-tol). Default is 4.0.}
  \item{mle_itr}{A numeric value specifies the maximum number of iterations for the computation. Default is 100.}
  \item{block_size}{A numeric value represents the block size for block Vecchia. Default is 200.}
  \item{dimension}{A string specifies the data dimension. For scaled_block, can be an integer string (e.g., "8") or one of "2D", "3D", or "ST". For other types, must be "2D", "3D", or "ST". Default is "2D".}
  \item{data}{A list of data vectors from load_data. Default is NULL.}
  \item{matrix}{A numeric vector of measurements. Default is NULL.}
  \item{x}{A numeric vector of x-coordinates. Default is NULL.}
  \item{y}{A numeric vector of y-coordinates. Default is NULL.}
  \item{initial_theta}{A numeric vector represents the initial parameter values. Default is NULL.}
  \item{distance_scale}{A numeric vector represents the distance scale vector. For scaled_block, one value per dimension. Default is NULL.}
  \item{nn_multiplier}{A numeric value represents the nearest neighbor multiplier for scaled_block. Default is NULL (uses 400).}
  \item{conditioning_size}{A numeric value represents the conditioning size. Default is NULL (uses 100).}
  \item{ncores}{A numeric value specifies the number of CPU cores. Default is NULL (uses system default).}
  \item{permutation}{A string specifies the permutation method, either "random", "morton", "kdtree", "hilbert", or "mmd". Default is NULL (uses "random").}
  \item{kernel_type}{A string specifies the kernel type for scaled_block, either "Matern12", "Matern32", "Matern52", "Matern72", or "PowerExponential". Default is NULL (uses "Matern72").}
  \item{seed}{A numeric value specifies the seed for random number generation. Default is NULL.}
}

\value{
A list contains the log_likelihood and estimated_theta (a vector of estimated kernel parameters).
}

\description{
This function models data based on the provided Vecchia approximation type, kernel, distance matrix, and other parameters.
}

\examples{
\dontrun{
vecchia_type <- "block"
kernel <- "univariate_matern_stationary"
lower_bound <- c(0.01, 0.01, 0.01)
upper_bound <- c(3.0, 3.0, 3.0)
dimension <- "2D"
problem_size <- 1000

data_result <- load_data(vecchia_type=vecchia_type, kernel=kernel,
problem_size=problem_size, seed=123, block_size=200, dimension=dimension)

theta <- model_data(vecchia_type=vecchia_type, kernel=kernel,
lb=lower_bound, ub=upper_bound, tol=4.0, mle_itr=100,
data=data_result)
}
}

