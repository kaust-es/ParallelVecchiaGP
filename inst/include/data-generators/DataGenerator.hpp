
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file DataGenerator.hpp
 * @brief Contains definition for abstract Data Generator Class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGBCPP_DATAGENERATOR_HPP
#define VECCHIAGBCPP_DATAGENERATOR_HPP

#include <memory>

#include <data-units/VecchiaGBData.hpp>
#include <configurations/Configurations.hpp>
#include <kernels/Kernel.hpp>

namespace vecchia::generators {

    /**
     * @class DataGenerator
     * @brief Abstract base class for generating synthetic or real data.
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class DataGenerator {

    public:

        /**
         * @brief Either generates synthetic data or reads data files.
         * @details This method generates the X, Y, and Z variables used to define the locations of the data points.
         * @param[in] aConfigurations Reference to the data configurations.
         * @param[in] aKernel Reference to the used Kernel.
         * @return unique Pointer to a populated data.
         *
         */
        virtual std::unique_ptr<VecchiaGBData<T>>
        CreateData(configurations::Configurations &aConfigurations,
                   vecchia::kernels::Kernel<T> &aKernel) = 0;

        /**
         * @brief Factory method for creating a data generator object.
         * @details This method creates a data generator object based on the specified configurations.
         * @param[in] aConfigurations Reference to the data configurations.
         * @return A unique pointer to the created data generator object.
         *
         */
        static std::unique_ptr<DataGenerator>
        CreateGenerator(configurations::Configurations &aConfigurations);

        /**
         * @brief Destructor for the data generator object.
         * @details This method frees the memory used by the data generator object.
         *
         */
        virtual ~DataGenerator();

    protected:

        /// Used enum for data generators types.
        static bool isSynthetic;
    };

    /**
     * @brief Instantiates the Data Generator class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(DataGenerator)
}//namespace vecchia

#endif //VECCHIAGBCPP_DATAGENERATOR_HPP