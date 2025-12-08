
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file EstimatorFactory.cpp
 * @brief Implementation of the EstimatorFactory class for creating estimators for different Vecchia types.
 * The factory creates a unique pointer to a concrete implementation of the Estimator class based on the Vecchia type specified.
 * If the required library is not enabled, it throws a runtime_error exception.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-09-29
**/

#include <estimators/EstimatorFactory.hpp>
#include <estimators/concrete/ParallelEstimator.hpp>
#include <estimators/concrete/ParallelBlockEstimator.hpp>
#include <estimators/concrete/ScaledBlockEstimator.hpp>

using namespace vecchia::estimators;
using namespace vecchia::common;

template<typename T>
std::unique_ptr<EstimatorFactory<T>> EstimatorFactory<T>::CreateEstimator(VecchiaType aVecchiaType) {

    // Check the used Linear Algebra solver library and method type
    if (aVecchiaType == PARALLEL_VECCHIA_GP) {
        // Scalar Vecchia with KBLAS
        return std::make_unique<ParallelEstimator<T>>();
    }
    else if (aVecchiaType == PARALLEL_BLOCK_VECCHIA_GP) {
        // Block Vecchia with MAGMA
        return std::make_unique<ParallelBlockEstimator<T>>();
    } 
    else if (aVecchiaType == PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
        // Scaled Block Vecchia with MAGMA+MPI
        return std::make_unique<ScaledBlockEstimator<T>>();
    }
    throw std::runtime_error("Invalid Vecchia type");
}
