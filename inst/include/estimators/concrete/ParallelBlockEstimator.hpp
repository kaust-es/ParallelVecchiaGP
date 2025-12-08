
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file ParallelBlockEstimator.hpp
 * @brief Header file for the EstimatorFactory class, which creates estimators based on the input computation type.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_PARALLELBLOCKESTIMATOR_HPP
#define VECCHIAGP_PARALLELBLOCKESTIMATOR_HPP

#include <memory>

#include <estimators/EstimatorFactory.hpp>

namespace vecchia::estimators {

    /**
     * @class ParallelBlockEstimator
     * @brief A class that creates estimators based on the input computation type.
     * @tparam T Data Type: float or double.
     *
     */
    template<typename T>
    class ParallelBlockEstimator : public EstimatorFactory<T> {
    public:

        /**
         * @brief Estimates the parameters using ParallelBlock.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         * @param[in] apTheta The parameters.
         * @return The estimated parameters.
         */
        T Estimate(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData, const double *apTheta) override;

        /**
         * @brief Initializes the memory for the estimator.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         */
        void InitMemory(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData) override;
    };

    /**
    * @brief Instantiates the Estimator factory class for float and double types.
    * @tparam T Data Type: float or double
    *
    */
    VECCHIAGP_INSTANTIATE_CLASS(ParallelBlockEstimator)

}//namespace vecchia

#endif //VECCHIAGP_PARALLELBLOCKESTIMATOR_HPP