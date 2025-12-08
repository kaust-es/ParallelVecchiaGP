
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file LocationGenerator.hpp
 * @brief Generates and manages spatial locations for VecchiaGB.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGBCPP_LOCATIONGENERATOR_HPP
#define VECCHIAGBCPP_LOCATIONGENERATOR_HPP

#include <data-units/Locations.hpp>
#include <data-units/Point.hpp>

namespace vecchia::generators {

    /**
     * @class LocationGenerator
     * @brief Generates spatial locations based on given parameters.
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class LocationGenerator {

    public:

        /**
         * @brief Generates the data locations.
         * @details This method generates the X, Y, and Z variables used to define the locations of the data points.
         * @param[in] aN The number of data points.
         * @param[in] aTimeSlot The time slot.
         * @param[in] aDimension The dimension of the locations.
         * @param[out] aLocations Reference to the Locations object where the generated data will be stored.
         * @return void
         *
         */
        static void GenerateLocations(const int &aN, const int &aTimeSlot, const common::Dimension &aDimension,
                                      dataunits::Locations<T> &aLocations);

        /**
         * @brief Generate uniform distribution between rangeLow , rangeHigh.
         * @param[in] aRangeLow The Lower range.
         * @param[in] aRangeHigh The Higher range.
         * @return The scaled uniform distribution between the two bounds.
         *
         */
        static T UniformDistribution(const T &aRangeLow, const T &aRangeHigh);

        /**
         * @brief Sort locations in Morton order (input points must be in [0;1]x[0;1] square]).
         * @param[in] aN The problem size divided by P-Grid.
         * @param[in] aDimension Dimension of locations.
         * @param[in,out] aLocations Locations to be sorted.
         * @return void
         *
         */
        static void
        SortLocations(const int &aN, const common::Dimension &aDimension, dataunits::Locations<T> &aLocations);

        /**
         * @brief Randomly reordering the locations.
         * @param[in] aN The number of data points.
         * @param[in] aDimension The dimension of the locations.
         * @param[in,out] aLocations Locations to be reordered.
         * @return void
         */
        static void RandomReordering(const int &aN, const common::Dimension &aDimension, dataunits::Locations<T> &aLocations);

        // TODO: Implment all spatial statistics reordering methods.
        // /**
        //  * @brief KD-Tree reordering of the locations.
        //  * @param[in] aN The number of data points.
        //  * @param[in] aDimension The dimension of the locations.
        //  * @param[in,out] aLocations Locations to be reordered.
        //  * @return void
        //  */
        // static void KDTreeReordering(const int &aN, const common::Dimension &aDimension, dataunits::Locations<T> &aLocations);
    
    /**
     * @brief Reordering the locations.
     * @param[in,out] aLocations Locations to be reordered.
     * @param[in] aDimension The dimension of the locations.
     * @param[in] aPoints Points to be reordered.
     * @param[in,out] aClusterReordering Cluster reordering.
     * @param[in] aNClusters The number of clusters.
     */
    static void ReorderIndex(dataunits::Locations<T> &aLocations, const common::Dimension &aDimension, 
        std::vector<dataunits::Point<T>> &aPoints, int *aClusterReordering, int aNClusters, bool aTimeFlag);

    /**
        * @brief Reordering the locations.
        * @param[in] aN The number of data points.
        * @param[in] aDimension The dimension of the locations.
        * @param[in] aPoints Points to be reordered.
        * @param[in,out] aClusterReordering Cluster reordering.
        * @param[in] aNClusters The number of clusters.
        */
    static void ClusterToBatch(int num_loc, int batchCount, int *batchNum, int *batchNumAccum, 
        int *batchIndex, dataunits::Locations<T> &locations, T *h_obs, dataunits::Locations<T> &locations_new, T *h_obs_new, 
        std::vector<dataunits::Point<T>> &aPoints, dataunits::Locations<T> &locsCentroid, bool time_flag);
};

    /**
     * @brief Instantiates the Data Generator class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(LocationGenerator)
}//namespace vecchia

#endif //VECCHIAGBCPP_LOCATIONGENERATOR_HPP
