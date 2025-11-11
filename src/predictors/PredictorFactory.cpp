
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file EstimatorFactory.cpp
 * @brief Implementation of the EstimatorFactory class for creating estimators for different Vecchia types.
 * The factory creates a unique pointer to a concrete implementation of the Estimator class based on the Vecchia type specified.
 * If the required library is not enabled, it throws a runtime_error exception.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @date 2025-09-29
**/

#include <predictors/PredictorFactory.hpp>
#include <predictors/concrete/ScaledBlockPredictor.hpp>
#include <predictors/concrete/ParallelBlockPredictor.hpp>

using namespace vecchia::predictors;
using namespace vecchia::common;

template<typename T>
std::unique_ptr<PredictorFactory<T>> PredictorFactory<T>::CreatePredictor(VecchiaType aVecchiaType) {

    // Check the used Linear Algebra solver library and method type
    if (aVecchiaType == PARALLEL_BLOCK_VECCHIA_GP) {
        // Block Vecchia with GSL and OpenMP
        return std::make_unique<ParallelBlockPredictor<T>>();
    } 
    else if (aVecchiaType == PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
        // Scaled Block Vecchia with MAGMA+MPI
        return std::make_unique<ScaledBlockPredictor<T>>();
    }
    throw std::runtime_error("Invalid Vecchia type");
}
