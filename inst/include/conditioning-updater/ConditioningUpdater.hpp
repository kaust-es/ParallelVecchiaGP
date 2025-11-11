
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file ConditioningUpdater.hpp
 * @brief Manages conditioning updater operations for VecchiaGB.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGBCPP_CONDITIONINGUPDATER_HPP
#define VECCHIAGBCPP_CONDITIONINGUPDATER_HPP

#include <configurations/Configurations.hpp>
#include <data-units/VecchiaGBData.hpp>
#include <data-units/Locations.hpp>

namespace vecchia::conditioningupdater {

    /**
     * @class ConditioningUpdater
     * @brief Abstract base class for conditioning updater.
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class ConditioningUpdater {

    public:

        /**
         * @brief Reads data from external sources into VecchiaGB format.
         * @return void
         *
         */
        virtual void
        Update(configurations::Configurations &aConfigurations, ::VecchiaGBData<T> &aData, int i_block) = 0;

        /**
         * @brief Save cluster and neighbor files.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         * @return void
         *
         */
        void SaveClusterAndNeighborFiles(configurations::Configurations &aConfigurations, ::VecchiaGBData<T> &aData);

    };

    /**
     * @brief Instantiates the Synthetic Data Generator class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(ConditioningUpdater)
} // namespace vecchia

#endif //VECCHIAGBCPP_CONDITIONINGUPDATER_HPP
