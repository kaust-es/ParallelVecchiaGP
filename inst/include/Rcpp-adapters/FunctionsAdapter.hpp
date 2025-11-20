// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file FunctionsAdapter.hpp
 * @brief Header file for function adapters in the Vecchia software.
 * @details It provides declarations for functions that adapt Vecchia GP operations for R.
 * @version 1.0.0
 * @author Generated for R wrapper
 * @date 2025-01-01
**/

#ifndef VECCHIAGP_FUNCTIONSADAPTER_HPP
#define VECCHIAGP_FUNCTIONSADAPTER_HPP

#include <Rcpp.h>

namespace vecchia::adapters {

    /**
     * @brief Function to load Vecchia data.
     * @param[in] vecchia_type Type of Vecchia approximation ("parallel", "block", "scaled_block")
     * @param[in] kernel Kernel name (e.g., "Matern", "univariate_matern_stationary")
     * @param[in] initial_theta Initial parameter values
     * @param[in] distance_matrix Distance metric ("euclidean" or "great_circle")
     * @param[in] problem_size Number of data points
     * @param[in] seed Random seed
     * @param[in] block_size Block size for block Vecchia
     * @param[in] dimension Dimension ("2D", "3D", "ST", or integer string like "8" for scaled_block)
     * @param[in] data_path Path to data file (empty for synthetic data)
     * @param[in] distance_scale Optional distance scale vector (for scaled_block, one per dimension)
     * @param[in] nn_multiplier Optional NN multiplier (for scaled_block, default 400)
     * @param[in] conditioning_size Optional conditioning size (default 100)
     * @param[in] ncores Optional number of CPU cores (default uses system default)
     * @param[in] permutation Optional permutation method ("random", "morton", etc., default "random")
     * @param[in] kernel_type Optional kernel type (for scaled_block, default "Matern72")
     * @return R list containing x, y, and m (measurements)
     */
    Rcpp::List
    R_VecchiaLoadData(const std::string &vecchia_type, const std::string &kernel,
                     const std::vector<double> &initial_theta, const std::string &distance_matrix,
                     const int &problem_size, const int &seed, const int &block_size,
                     const std::string &dimension, const std::string &data_path,
                     Rcpp::Nullable<Rcpp::NumericVector> distance_scale = R_NilValue,
                     Rcpp::Nullable<int> nn_multiplier = R_NilValue,
                     Rcpp::Nullable<int> conditioning_size = R_NilValue,
                     Rcpp::Nullable<int> ncores = R_NilValue,
                     Rcpp::Nullable<std::string> permutation = R_NilValue,
                     Rcpp::Nullable<std::string> kernel_type = R_NilValue);

    /**
     * @brief Function to estimate parameters using MLE.
     * @param[in] vecchia_type Type of Vecchia approximation
     * @param[in] kernel Kernel name
     * @param[in] distance_matrix Distance metric
     * @param[in] lb Lower bounds for parameters
     * @param[in] ub Upper bounds for parameters
     * @param[in] tol Tolerance (10^-tol)
     * @param[in] mle_itr Maximum MLE iterations
     * @param[in] block_size Block size
     * @param[in] dimension Dimension ("2D", "3D", "ST", or integer string like "8" for scaled_block)
     * @param[in] data Optional data from previous load_data call
     * @param[in] matrix Optional measurements vector
     * @param[in] x Optional x coordinates
     * @param[in] y Optional y coordinates
     * @param[in] initial_theta Optional initial theta values (for scaled_block: [variance, nugget], matches --iTheta)
     * @param[in] distance_scale Optional distance scale vector (for scaled_block, one per dimension)
     * @param[in] nn_multiplier Optional NN multiplier (for scaled_block, default 400)
     * @param[in] conditioning_size Optional conditioning size (default 100)
     * @param[in] ncores Optional number of CPU cores (default uses system default)
     * @param[in] permutation Optional permutation method ("random", "morton", etc., default "random")
     * @param[in] kernel_type Optional kernel type (for scaled_block, default "Matern72")
     * @return List containing log_likelihood (double) and estimated_theta (NumericVector)
     *         For scaled_block, estimated_theta includes [variance, nugget, distance_scale[dim]]
     */
    Rcpp::List
    R_VecchiaModelData(const std::string &vecchia_type, const std::string &kernel,
                      const std::string &distance_matrix, const std::vector<double> &lb,
                      const std::vector<double> &ub, const double &tol, const int &mle_itr,
                      const int &block_size, const std::string &dimension,
                      SEXP data = R_NilValue,
                      Rcpp::Nullable<Rcpp::NumericVector> matrix = R_NilValue,
                      Rcpp::Nullable<Rcpp::NumericVector> x = R_NilValue,
                      Rcpp::Nullable<Rcpp::NumericVector> y = R_NilValue,
                      Rcpp::Nullable<Rcpp::NumericVector> initial_theta = R_NilValue,
                      Rcpp::Nullable<Rcpp::NumericVector> distance_scale = R_NilValue,
                      Rcpp::Nullable<int> nn_multiplier = R_NilValue,
                      Rcpp::Nullable<int> conditioning_size = R_NilValue,
                      Rcpp::Nullable<int> ncores = R_NilValue,
                      Rcpp::Nullable<std::string> permutation = R_NilValue,
                      Rcpp::Nullable<std::string> kernel_type = R_NilValue,
                      Rcpp::Nullable<int> seed = R_NilValue);

    /**
     * @brief Function to perform prediction.
     * @param[in] vecchia_type Type of Vecchia approximation
     * @param[in] kernel Kernel name
     * @param[in] distance_matrix Distance metric
     * @param[in] estimated_theta Estimated parameter values (full theta including distance_scale for scaled_block)
     * @param[in] block_size Block size
     * @param[in] dimension Dimension ("2D", "3D", "ST", or integer string like "8" for scaled_block)
     * @param[in] train_data Training data list [x, y, measurements] OR train data file path (CSV)
     * @param[in] test_data Test data list [x, y] OR test data file path (CSV)
     * @param[in] train_locs Optional train locations file path (CSV) - if provided, train_data should be file path
     * @param[in] test_locs Optional test locations file path (CSV) - if provided, test_data should be file path
     * @param[in] distance_scale Optional distance scale vector (for scaled_block, one per dimension)
     * @param[in] nn_multiplier Optional NN multiplier (for scaled_block, default 400)
     * @param[in] conditioning_size Optional conditioning size (default 100)
     * @param[in] ncores Optional number of CPU cores (default uses system default)
     * @param[in] permutation Optional permutation method ("random", "morton", etc., default "random")
     * @param[in] kernel_type Optional kernel type (for scaled_block, default "Matern72")
     * @return Vector of predicted values
     */
    Rcpp::NumericVector
    R_VecchiaPredictData(const std::string &vecchia_type, const std::string &kernel,
                        const std::string &distance_matrix, const std::vector<double> &estimated_theta,
                        const int &block_size, const std::string &dimension,
                        SEXP train_data, SEXP test_data,
                        Rcpp::Nullable<std::string> train_locs = R_NilValue,
                        Rcpp::Nullable<std::string> test_locs = R_NilValue,
                        Rcpp::Nullable<Rcpp::NumericVector> distance_scale = R_NilValue,
                        Rcpp::Nullable<int> nn_multiplier = R_NilValue,
                        Rcpp::Nullable<int> conditioning_size = R_NilValue,
                        Rcpp::Nullable<int> ncores = R_NilValue,
                        Rcpp::Nullable<std::string> permutation = R_NilValue,
                        Rcpp::Nullable<std::string> kernel_type = R_NilValue,
                        Rcpp::Nullable<int> seed = R_NilValue,
                        Rcpp::Nullable<int> problem_size = R_NilValue);

}

#endif // VECCHIAGP_FUNCTIONSADAPTER_HPP

