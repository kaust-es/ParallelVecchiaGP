
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file EstimatorFactory.hpp
 * @brief Header file for the EstimatorFactory class, which creates estimators based on the input computation type.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_PREDICTORFACTORY_HPP
#define VECCHIAGP_PREDICTORFACTORY_HPP

#include <memory>

#include <common/Definitions.hpp>
#include <configurations/Configurations.hpp>
#include <data-units/VecchiaGBData.hpp>

namespace vecchia::predictors {

    /**
     * @class PredictorFactory
     * @brief A class that creates predictors based on the input computation type.
     * @tparam T Data Type: float or double.
     *
     */
    template<typename T>
    class PredictorFactory {
    public:

        /**
         * @brief Creates a predictor based on the input computation type.
         * @param[in] aVecchiaType The computation type to create the solver for.
         * @return Pointer to the created linear algebra solver.
         *
         */
        static std::unique_ptr<predictors::PredictorFactory<T>> CreatePredictor(common::VecchiaType aVecchiaType);

        /**
         * @brief Predicts the parameters using the predictor.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         * @param[in] theta The parameters.
         * @param[in] apPredictions The predictions.
         * @return The estimated parameters.
         */
        virtual T Predict(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData, const double *apTheta) = 0;

        /**
         * @brief Initializes the memory for the predictor.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         */
        virtual void InitMemory(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData) = 0;
    };

    /**
    * @brief Instantiates the Estimator factory class for float and double types.
    * @tparam T Data Type: float or double
    *
    */
    VECCHIAGP_INSTANTIATE_CLASS(PredictorFactory)

}//namespace vecchia

#endif //VECCHIAGP_PREDICTORFACTORY_HPP