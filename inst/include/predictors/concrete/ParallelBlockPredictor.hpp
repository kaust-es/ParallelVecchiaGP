
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/** 
 * @file ParallelBlockPredictor.hpp
 * @brief Header file for ParallelBlockPredictor class - implements Block Vecchia prediction
 * @details This class implements the Block Vecchia approximation for Gaussian Process prediction
 *          using GSL for matrix operations and OpenMP for parallelization.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#ifndef VECCHIAGP_PARALLELBLOCKPREDICTOR_HPP
#define VECCHIAGP_PARALLELBLOCKPREDICTOR_HPP

#include <memory>
#include <vector>

#include <predictors/PredictorFactory.hpp>

namespace vecchia::predictors {

    /**
     * @class ParallelBlockPredictor
     * @brief Implements Block Vecchia approximation with GSL and OpenMP
     * @tparam T Data Type: float or double.
     *
     */
    template<typename T>
    class ParallelBlockPredictor : public PredictorFactory<T> {
    public:

        /**
         * @brief Constructor
         */
        ParallelBlockPredictor() = default;

        /**
         * @brief Destructor
         */
        ~ParallelBlockPredictor() = default;

        /**
         * @brief Predicts the parameters using Block Vecchia.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         * @param[in] apTheta The parameters.
         * @return The predictions.
         */
        virtual T Predict(configurations::Configurations &aConfigurations, 
                   std::unique_ptr<VecchiaGBData<T>> &aData, 
                   const double *apTheta) override;

        /**
         * @brief Initializes memory and data structures for Block Vecchia.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         */
        void InitMemory(configurations::Configurations &aConfigurations, 
                       std::unique_ptr<VecchiaGBData<T>> &aData) override;

    private:
        // TODO: Add private member variables if needed
    };

}//namespace vecchia::predictors

#endif //VECCHIAGP_PARALLELBLOCKPREDICTOR_HPP

