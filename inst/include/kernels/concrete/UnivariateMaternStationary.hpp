
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file UnivariateMaternStationary.hpp
 * @brief Defines the UnivariateMaternStationary class, a univariate stationary Matern kernel.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @author Suhas Shankar
 * @author Mary Lai Salvana
 * @date 2023-04-12
 *
 * This file provides the declaration of the UnivariateMaternStationary class, which is a subclass of the Kernel class
 * and represents a univariate stationary Matern kernel. It provides a method for generating a covariance matrix
 * using a set of input locations and kernel parameters.
 *
**/

#ifndef VECCHIAGP_UNIVARIATEMATERNSTATIONARY_HPP
#define VECCHIAGP_UNIVARIATEMATERNSTATIONARY_HPP

#include <kernels/Kernel.hpp>

namespace vecchia::kernels {

    /**
     * @class UnivariateMaternStationary
     * @brief A class represents a Univariate Matern Stationary kernel.
     * @details This class represents a Univariate Matern Stationary, which is a subclass of the Kernel class.
     * It provides a method for generating a covariance matrix using a set of input locations and kernel parameters.
     *
     */
    template<typename T>
    class UnivariateMaternStationary : public Kernel<T> {

    public:

        /**
         * @brief Constructs a new UnivariateMaternStationary object.
         * @details Initializes a new UnivariateMaternStationary object with default values.
         *
         */
        UnivariateMaternStationary();

        /**
         * @brief Virtual destructor to allow calls to the correct concrete destructor.
         *
         */
        ~UnivariateMaternStationary() override = default;

        /**
         * @brief Generates a covariance matrix using a set of locations and kernel parameters.
         * @copydoc Kernel::GenerateCovarianceMatrix()
         *
         */
        void
        GenerateCovarianceMatrix(double *apMatrixA, const int &aRowsNumber, const int &aColumnsNumber, const int &aRowOffset,
                                 const int &aColumnOffset, dataunits::Locations<T> &aLocation1,
                                 dataunits::Locations<T> &aLocation2, dataunits::Locations<T> &aLocation3,
                                 double *apLocalTheta, const int &aDistanceMetric) override;

        /**
         * @brief Creates a new UnivariateMaternStationary object.
         * @details This method creates a new UnivariateMaternStationary object and returns a pointer to it.
         * @return A pointer to the new UnivariateMaternStationary object.
         *
         */
        static Kernel<T> *Create();

    private:
        //// Used plugin name for static registration
        static bool plugin_name;
    };

    /**
     * @brief Instantiates the Data Generator class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(UnivariateMaternStationary)

}//namespace vecchia

#endif //VECCHIAGP_UNIVARIATEMATERNSTATIONARY_HPP
