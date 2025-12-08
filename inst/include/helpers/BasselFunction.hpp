
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file BasselFunction.hpp
 * @brief This file contains the BasselFunction class which provides methods for computing derivatives of the modified Bessel function of the second kind. These functions are crucial in statistical and mathematical computations, especially in fields such as geostatistics and spatial analysis.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_BASSELFUNCTION_HPP
#define VECCHIAGP_BASSELFUNCTION_HPP

#include <common/Definitions.hpp>

namespace vecchia::helpers {

    /**
     * @class BasselFunction
     * @brief The BasselFunction class provides methods for computing various derivatives of the modified Bessel function of the second kind, \( K_{\nu} \). This class is templated to support both float and double data types, enabling precision-based computations as required by different applications.
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class BasselFunction {

    public:
        /**
         * @brief Calculates the derivative of the modified Bessel function of the second kind (K_nu) with respect to its order, evaluated at input_value and order aOrder.
         * @param[in] aOrder The order of the Bessel function.
         * @param[in] aInputValue The input value at which to evaluate the derivative.
         * @return The value of the derivative of K_nu with respect to its order, evaluated at input_value and order aOrder.
         *
         */
        static T CalculateDerivativeBesselNu(const T &aOrder, const T &aInputValue);

        /**
         * @brief Calculates the second derivative of the modified Bessel function of the second kind (K_nu) with respect to its input, evaluated at input_value and order aOrder.
         * @param[in] aOrder The order of the Bessel function.
         * @param[in] aInputValue The input value at which to evaluate the second derivative.
         * @return The value of the second derivative of K_nu with respect to its input, evaluated at input_value and order aOrder.
         *
         */
        static T CalculateSecondDerivativeBesselNu(const T &aOrder, const T &aInputValue);

        /**
         * @brief Calculates the second derivative of the modified Bessel function of the second kind (K_nu) with respect to its input, evaluated at input_value and order aOrder.
         * @param[in] aOrder The order of the Bessel function.
         * @param[in] aInputValue The input value at which to evaluate the derivative.
         * @return The value of the derivative of K_nu with respect to its input, evaluated at input_value and order aOrder.
         *
         */
        static T CalculateSecondDerivativeBesselNuInput(const T &aOrder, const T &aInputValue);

    };

    /**
      * @brief Instantiates the DiskWriter class for float and double types.
      * @tparam T Data Type: float or double
      *
      */
    VECCHIAGP_INSTANTIATE_CLASS(BasselFunction)
}

#endif //VECCHIAGP_BASSELFUNCTION_HPP
