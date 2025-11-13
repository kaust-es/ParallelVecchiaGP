
% Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
% All rights reserved.
% Vecchia is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

% @file predict_data.Rd
% @brief roxygen2 documentation for the R Interface of R_VecchiaPredictData function.
% @version 1.0.0
% @author Generated for R wrapper
% @date 2025-01-01

\name{predict_data}
\alias{predict_data}
\title{Predict Data function}

\usage{
predict_data(vecchia_type = "block", kernel = "univariate_matern_stationary",
distance_matrix = "euclidean", estimated_theta = c(1.0, 0.5, 0.1),
block_size = 200, dimension = "2D", train_data = NULL,
test_data = NULL, train_locs = NULL, test_locs = NULL,
distance_scale = NULL, nn_multiplier = NULL,
conditioning_size = NULL, ncores = NULL, permutation = NULL,
kernel_type = NULL, seed = NULL, problem_size = NULL)
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
  \item{distance_matrix}{A string specifies the distance metric, either "euclidean" or "great_circle". Default is "euclidean".}
  \item{estimated_theta}{A numeric vector contains the estimated theta parameters.}
  \item{block_size}{A numeric value represents the block size for block Vecchia. Default is 200.}
  \item{dimension}{A string specifies the data dimension. For scaled_block, can be an integer string (e.g., "8") or one of "2D", "3D", or "ST". For other types, must be "2D", "3D", or "ST". Default is "2D".}
  \item{train_data}{A list contains the locations and measurements for training, or a file path string if train_locs is provided. Default is NULL.}
  \item{test_data}{A list contains the locations for testing, or a file path string if test_locs is provided. Default is NULL.}
  \item{train_locs}{A string specifies the file path to training locations CSV file. Default is NULL.}
  \item{test_locs}{A string specifies the file path to test locations CSV file. Default is NULL.}
  \item{distance_scale}{A numeric vector represents the distance scale vector. For scaled_block, one value per dimension. Default is NULL.}
  \item{nn_multiplier}{A numeric value represents the nearest neighbor multiplier for scaled_block. Default is NULL (uses 400).}
  \item{conditioning_size}{A numeric value represents the conditioning size. Default is NULL (uses 100).}
  \item{ncores}{A numeric value specifies the number of CPU cores. Default is NULL (uses system default).}
  \item{permutation}{A string specifies the permutation method, either "random", "morton", "kdtree", "hilbert", or "mmd". Default is NULL (uses "random").}
  \item{kernel_type}{A string specifies the kernel type for scaled_block, either "Matern12", "Matern32", "Matern52", "Matern72", or "PowerExponential". Default is NULL (uses "Matern72").}
  \item{seed}{A numeric value specifies the seed for random number generation. Default is NULL.}
  \item{problem_size}{A numeric value represents the problem size. Used for scaled_block when data is already loaded. Default is NULL.}
}

\value{
A vector of predicted values at the test locations.
}

\description{
This function predicts data based on the provided kernel, distance matrix, estimated theta, and other parameters.
}

\examples{
\dontrun{
vecchia_type <- "block"
kernel <- "univariate_matern_stationary"
estimated_theta <- c(1.0, 0.5, 0.1)
dimension <- "2D"

data_result <- load_data(vecchia_type=vecchia_type, kernel=kernel,
problem_size=1000, seed=123, block_size=200, dimension=dimension)

model_result <- model_data(vecchia_type=vecchia_type, kernel=kernel,
data=data_result, lb=c(0.01, 0.01, 0.01), ub=c(3.0, 3.0, 3.0))

test_x <- c(0.1, 0.2, 0.3)
test_y <- c(0.4, 0.5, 0.6)
test_data <- list(test_x, test_y)

predict_data(train_data=data_result, test_data=test_data,
kernel=kernel, estimated_theta=model_result$estimated_theta,
vecchia_type=vecchia_type)
}
}

