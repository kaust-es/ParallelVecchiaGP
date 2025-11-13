
% Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
% All rights reserved.
% Vecchia is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

% @file Vecchia-package.Rd
% @brief roxygen2 documentation for the Vecchia package.
% @version 1.0.0
% @author Generated for R wrapper
% @date 2025-01-01

\name{Vecchia-package}
\alias{Vecchia-package}
\alias{Vecchia}
\docType{package}
\title{Vecchia Package}

\description{
Vecchia is a package for computing the Vecchia approximation of a Gaussian process.
It is designed to be fast and memory efficient, and to work with both dense and sparse matrices.
The package supports parallel, block, and scaled block Vecchia approximations.
}

\details{
The Vecchia package provides efficient implementations of Vecchia approximations
for Gaussian processes. It supports multiple Vecchia types including:
\itemize{
  \item "parallel": Parallel Vecchia GP
  \item "block": Parallel Block Vecchia GP
  \item "scaled_block": Parallel Scaled Block Vecchia GP
}

The package supports various kernels and distance metrics (euclidean, great_circle)
for 2D, 3D, and space-time (ST) dimensions.

Main functions:
\itemize{
  \item \code{\link{load_data}}: Load and prepare data for Vecchia approximation
  \item \code{\link{model_data}}: Estimate model parameters using maximum likelihood
  \item \code{\link{predict_data}}: Make predictions at new locations
}
}

\author{
Qilong Pan, Sohayla Khaled, Mahmoud ElKarargy, Sameh Abdulah

Maintainer: Sameh Abdulah <sameh.abdulah@kaust.edu.sa>
}

\references{
\url{https://github.com/kaust-es/ParallelVecchiaGP/}
}

\keyword{package}

