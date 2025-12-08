
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file Vecchia.hpp
 * @brief High-Level Wrapper class containing the static API for Vecchia operations.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_VECCHIAGP_HPP
#define VECCHIAGP_VECCHIAGP_HPP

#include <nlopt.hpp>

#include <configurations/Configurations.hpp>
#include <data-units/VecchiaGBData.hpp>

namespace vecchia::api {
    /**
     * @brief High-Level Wrapper class containing the static API for Vecchia operations.
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class VecchiaGP {
    public:

        /**
         * @brief Generates Data whether it's synthetic data or real.
         * @param[in] aConfigurations Reference to Configurations object containing user input data.
         * @param[out] aData Reference to an VecchiaData<T> object where generated data will be stored.
         * @return void
         *
         */
        static void VecchiaLoadData(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData);

        /**
         * @brief Estimates parameters using Maximum Likelihood Estimation
         * @param[in] aConfigurations Reference to Configurations object containing user input data.
         * @param[in,out] aData Reference to an VecchiaData<T> object containing needed descriptors, and locations.
         * @return the optimum log-likelihood value.
         *
         */
        static T VecchiaDataEstimation(configurations::Configurations &aConfigurations,
                                       std::unique_ptr<VecchiaGBData<T>> &aData,
                                       T *apMeasurementsMatrix = nullptr);


        /**
         * @brief Objective function used in optimization, and following the NLOPT objective function format.
         * @param[in] aTheta An array of length n containing the current point in the parameter space.
         * @param[in] aGrad  An array of length n where you can optionally return the gradient of the objective function.
         * @param[in] apInfo pointer containing needed configurations and data.
         * @return double MLE results.
         *
         */
        static double VecchiaMLETileAPI(const std::vector<double> &aTheta, std::vector<double> &aGrad, void *apInfo);

        /**
         * @brief Predict missing measurements values.
         * @param[in] aConfigurations Reference to Configurations object containing user input data.
         * @param[in, out] aData Reference to an VecchiaData<T> object containing needed descriptors, and locations.
         * @return void
         *
         */
        static void
        VecchiaPrediction(configurations::Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData, T *apMeasurementsMatrix);
                             
    };

    /**
     * @brief Instantiates the Vecchia class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(VecchiaGP)
}

#endif //VECCHIAGP_VECCHIAGP_HPP