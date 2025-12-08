
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file DistanceCalculationHelpers.hpp
 * @brief Contains the definition of the DistanceCalculationHelpers class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_DistanceCalculationHelpers_HPP
#define VECCHIAGP_DistanceCalculationHelpers_HPP

#include <data-units/Locations.hpp>
#include <data-units/Point.hpp>

namespace vecchia::helpers {

    /**
     * @Class DistanceCalculationHelpers
     * @brief Class to calculate the distance between two points.
     * @tparam T Data Type: float or double.
     *
     */

    template<typename T>
    class DistanceCalculationHelpers {
    public:
        /**
         * @brief Calculates the Euclidean distance between two points.
         * @param[in] aLocations1 Reference to the first set of locations.
         * @param[in] aLocations2 Reference to the second set of locations.
         * @param[in] aIdxLocation1 Index of the first location in the first set.
         * @param[in] aIdxLocation2 Index of the second location in the second set.
         * @param[in] aDistanceMetric Flag indicating the distance metric to use (1 for Manhattan distance, 2 for Euclidean distance).
         * @param[in] aFlagZ Flag indicating whether the points are in 2D or 3D space (0 for 2D, 1 for 3D).
         * @return The Euclidean distance between the two points.
         *
         */
        static T CalculateDistance(vecchia::dataunits::Locations<T> &aLocations1,
                                   vecchia::dataunits::Locations<T> &aLocations2, const int &aIdxLocation1,
                                   const int &aIdxLocation2, const int &aDistanceMetric, const int &aFlagZ);

        /**
         * @brief Calculates the great-circle distance between two points on Earth using the Haversine formula.
         * @param[in] aLatitude1 Latitude of the first point in degrees.
         * @param[in] aLongitude1 Longitude of the first point in degrees.
         * @param[in] aLatitude2 Latitude of the second point in degrees.
         * @param[in] aLongitude2 Longitude of the second point in degrees.
         * @return The distance between the two points in kilometers.
         *
         */
        static T DistanceEarth(T &aLatitude1, T &aLongitude1, T &aLatitude2, T &aLongitude2);

        /**
         * @brief Converts an angle from degrees to radians.
         * @details This function converts an angle from degrees to radians using the conversion factor π/180.
         * @param[in] aDegree The angle in degrees.
         * @return The angle converted to radians.
         *
         */
        static T DegreeToRadian(T aDegree);

        /**
         * @brief Calculates the Euclidean distance between two points.
         * @param[in] aPoint1 The first point.
         * @param[in] aPoint2 The second point.
         * @return The Euclidean distance between the two points.
         *
         */
        static inline T EuclideanDistance(const vecchia::dataunits::Point<T> &aPoint1,
                                   const vecchia::dataunits::Point<T> &aPoint2);
        /**
         * @brief Calculates the distance threshold.
         * @param[in] distance_scale The distance scale.
         * @param[in] numPointsTotal The number of total points.
         * @param[in] m The number of nearest neighbors.
         * @param[in] nn_multiplier The number of nearest neighbors multiplier.
         * @return The distance threshold.
         *
         */
        static T CalculateDistanceThreshold(const std::vector<T>& distance_scale, int numPointsTotal, int m, int nn_multiplier);

    };
    /**
      * @brief Instantiates the PredictionHelpers class for float and double types.
      * @tparam T Data Type: float or double
      *
      */
    VECCHIAGP_INSTANTIATE_CLASS(DistanceCalculationHelpers)
}
#endif //VECCHIAGP_DistanceCalculationHelpers_HPP
