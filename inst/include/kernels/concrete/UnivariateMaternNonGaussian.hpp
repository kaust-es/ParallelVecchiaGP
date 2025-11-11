
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file UnivariateMaternNonGaussian.hpp
 * @brief Defines the UnivariateMaternNonGaussian class, a Univariate Matern Non Gaussian kernel.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @author Suhas Shankar
 * @author Mary Lai Salvana
 * @date 2023-04-14
**/

#ifndef VECCHIAGP_UNIVARIATEMATERNNONGAUSSIAN_HPP
#define VECCHIAGP_UNIVARIATEMATERNNONGAUSSIAN_HPP

#include <kernels/Kernel.hpp>

namespace vecchia::kernels {

    /**
     * @class UnivariateMaternNonGaussian
     * @brief A class represents a Univariate Matern Non Gaussian kernel.
     * @details This class represents a Univariate Matern Non Gaussian, which is a subclass of the Kernel class.
     * It provides a method for generating a covariance matrix using a set of input locations and kernel parameters.
     *
     */
    template<typename T>
    class UnivariateMaternNonGaussian : public Kernel<T> {

    public:

        /**
         * @brief Constructs a new UnivariateMaternNonGaussian object.
         * @details Initializes a new UnivariateMaternNonGaussian object with default values.
         *
         */
        UnivariateMaternNonGaussian();

        /**
         * @brief Virtual destructor to allow calls to the correct concrete destructor.
         *
         */
        ~UnivariateMaternNonGaussian() override = default;

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
         * @brief Creates a new UnivariateMaternNonGaussian object.
         * @details This method creates a new UnivariateMaternNonGaussian object and returns a pointer to it.
         * @return A pointer to the new UnivariateMaternNonGaussian object.
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
    VECCHIAGP_INSTANTIATE_CLASS(UnivariateMaternNonGaussian)
}//namespace vecchia

#endif //VECCHIAGP_UNIVARIATEMATERNNONGAUSSIAN_HPP
