// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file RcppModules.cpp
 * @brief Rcpp module definitions for VecchiaGP.
 * @version 1.0.0
 * @author Generated for R wrapper
 * @date 2025-01-01
 **/

#include <Rcpp.h>

#include <Rcpp-adapters/FunctionsAdapter.hpp>

/** Expose C++ Object With the Given functions **/
RCPP_MODULE(VecchiaGB) {

    /** VecchiaGB Class **/
    using namespace Rcpp;

    function("load_data", &vecchia::adapters::R_VecchiaLoadData,
             List::create(
                 _["vecchia_type"] = "block",
                 _["kernel"] = "univariate_matern_stationary",
                 _["initial_theta"] = NumericVector::create(1.0, 0.5, 0.1),
                 _["distance_matrix"] = "euclidean",
                 _["problem_size"] = 2000,
                 _["seed"] = 123,
                 _["block_count"] = 200,
                 _["dimension"] = "2D",
                 _["data_path"] = "",
                 _["distance_scale"] = R_NilValue,
                 _["nn_multiplier"] = R_NilValue,
                 _["conditioning_size"] = R_NilValue,
                 _["ncores"] = R_NilValue,
                 _["permutation"] = R_NilValue,
                 _["kernel_type"] = R_NilValue
             ));

    List model_data_defaults;
    model_data_defaults["vecchia_type"] = "block";
    model_data_defaults["kernel"] = "univariate_matern_stationary";
    model_data_defaults["distance_matrix"] = "euclidean";
    model_data_defaults["lb"] = NumericVector::create(0.01, 0.01, 0.01);
    model_data_defaults["ub"] = NumericVector::create(3.0, 3.0, 3.0);
    model_data_defaults["tol"] = 4.0;
    model_data_defaults["mle_itr"] = 100;
    model_data_defaults["block_count"] = 200;
    model_data_defaults["dimension"] = "2D";
    model_data_defaults["data"] = R_NilValue;
    model_data_defaults["matrix"] = R_NilValue;
    model_data_defaults["x"] = R_NilValue;
    model_data_defaults["y"] = R_NilValue;
    model_data_defaults["initial_theta"] = R_NilValue;
    model_data_defaults["distance_scale"] = R_NilValue;
    model_data_defaults["nn_multiplier"] = R_NilValue;
    model_data_defaults["conditioning_size"] = R_NilValue;
    model_data_defaults["ncores"] = R_NilValue;
    model_data_defaults["permutation"] = R_NilValue;
    model_data_defaults["kernel_type"] = R_NilValue;
    model_data_defaults["seed"] = R_NilValue;
    
    function("model_data", &vecchia::adapters::R_VecchiaModelData, model_data_defaults);

    function("predict_data", &vecchia::adapters::R_VecchiaPredictData,
             List::create(
                 _["vecchia_type"] = "block",
                 _["kernel"] = "univariate_matern_stationary",
                 _["distance_matrix"] = "euclidean",
                 _["estimated_theta"] = NumericVector::create(1.0, 0.5, 0.1),
                 _["block_count"] = 200,
                 _["dimension"] = "2D",
                 _["train_data"] = R_NilValue,
                 _["test_data"] = R_NilValue,
                 _["train_locs"] = R_NilValue,
                 _["test_locs"] = R_NilValue,
                 _["distance_scale"] = R_NilValue,
                 _["nn_multiplier"] = R_NilValue,
                 _["conditioning_size"] = R_NilValue,
                 _["ncores"] = R_NilValue,
                 _["permutation"] = R_NilValue,
                 _["kernel_type"] = R_NilValue,
                 _["seed"] = R_NilValue,
                 _["problem_size"] = R_NilValue
             ));

}

