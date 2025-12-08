
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file KnnConditioningUpdater.hpp
 * @brief A class for generating synthetic data.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGBCPP_KNNCONDITIONINGUPDATER_HPP
#define VECCHIAGBCPP_KNNCONDITIONINGUPDATER_HPP

    #include <conditioning-updater/ConditioningUpdater.hpp>

namespace vecchia::conditioningupdater::knn {

    /**
    * @class KnnConditioningUpdater
     * @brief A class for updating the conditioning using KNN.
     * @tparam T Data Type: float or double
     * @details This class generates synthetic data for use in testing machine learning models.
     *
     */
    template<typename T>
    class KnnConditioningUpdater : public ConditioningUpdater<T> {

    public:

        /**
         * @brief Constructor.
         * @param[in] aConfigurations The configurations.
         * @param[in] aData The data.
         * @return void
         *
         */
        KnnConditioningUpdater(configurations::Configurations &aConfigurations, ::VecchiaGBData<T> &aData);

        /**
         * @brief Updates the conditioning using KNN.
         * @copydoc ConditioningUpdater::Update()
         *
         */
        void
        Update(configurations::Configurations &aConfigurations, ::VecchiaGBData<T> &aData, int i_block) override;

    };

    /**
     * @brief Instantiates the KnnConditioningUpdater class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(KnnConditioningUpdater)
} // namespace vecchia

#endif //VECCHIAGBCPP_KNNCONDITIONINGUPDATER_HPP