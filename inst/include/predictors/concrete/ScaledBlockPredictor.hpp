
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/** 
 * @file ScaledBlockPredictor.hpp
 * @brief Header file for ScaledBlockPredictor class - implements Scaled Block Vecchia prediction
 * @details This class implements the distributed Scaled Block Vecchia approximation for large-scale
 *          Gaussian Process prediction using MPI + MAGMA vbatched operations.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#ifndef VECCHIAGP_SCALEDBLOCKPREDICTOR_HPP
#define VECCHIAGP_SCALEDBLOCKPREDICTOR_HPP

#include <memory>
#include <vector>

#include <predictors/PredictorFactory.hpp>

namespace vecchia::predictors {

    /**
     * @class ScaledBlockPredictor
     * @brief Implements Scaled Block Vecchia approximation with MPI and MAGMA vbatched operations
     * @tparam T Data Type: float or double.
     *
     */
    template<typename T>
    class ScaledBlockPredictor : public PredictorFactory<T> {
    public:

        /**
         * @brief Constructor
         */
        ScaledBlockPredictor() = default;

        /**
         * @brief Destructor
         */
        ~ScaledBlockPredictor() = default;

        /**
         * @brief Predicts the parameters using Scaled Block Vecchia.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         * @param[in] apTheta The parameters.
         * @return The predictions.
         */
        virtual T Predict(configurations::Configurations &aConfigurations, 
                   std::unique_ptr<VecchiaGBData<T>> &aData, 
                   const double *apTheta) override;

        /**
         * @brief Initializes memory and data structures for Scaled Block Vecchia.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         */
        void InitMemory(configurations::Configurations &aConfigurations, 
                       std::unique_ptr<VecchiaGBData<T>> &aData) override;

    private:
        // TODO: Add private member variables for GPU data structures, block metadata, etc.
        // This will be populated as we implement the functionality
    };

    /**
    * @brief Instantiates the ScaledBlockEstimator class for float and double types.
    * @tparam T Data Type: float or double
    *
    */
    VECCHIAGP_INSTANTIATE_CLASS(ScaledBlockPredictor)

}//namespace vecchia::predictors

#endif //VECCHIAGP_SCALEDBLOCKPREDICTOR_HPP

