// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file Point.hpp
 * @brief Header file for the Point class, which represents a point in 3D space with coordinates and cluster information.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_POINT_HPP
#define VECCHIAGP_POINT_HPP

#include <iostream>
#include <vector>
#include <common/Definitions.hpp>

namespace vecchia::dataunits {

    /**
     * @class Point
     * @brief A class representing a point in 3D space with coordinates and cluster information.
     * @tparam T Data Type: float or double
     */
    template<typename T>
    class Point {
    public:
        /**
         * @brief Default constructor.
         * @details Initializes coordinates to zero and cluster to -1.
         */
        Point();

        /**
         * @brief Constructor with initial values.
         * @param[in] aCoordinates Array of coordinates.
         * @param[in] aCluster Cluster ID.
         */
        Point(const T aCoordinates[3], const int &aCluster);

        /**
         * @brief Copy constructor.
         * @param[in] aPoint Point to be copied.
         */
        Point(const Point<T> &aPoint) = default;

        /**
         * @brief Destructor.
         */
        ~Point() = default;

        /**
         * @brief Assignment operator.
         * @param[in] aPoint Point to assign.
         * @return Reference to this Point.
         */
        Point<T>& operator=(const Point<T> &aPoint) = default;

        /**
         * @brief Equality operator.
         * @param[in] aPoint Point to compare with.
         * @return True if points are equal, false otherwise.
         */
        bool operator==(const Point<T> &aPoint) const;

        /**
         * @brief Addition assignment operator.
         * @param[in] aPoint Point to add.
         * @return Reference to this Point.
         */
        Point<T>& operator+=(const Point<T> &aPoint);

        /**
         * @brief Division assignment operator.
         * @param[in] aCardinality Divisor value.
         * @return Reference to this Point.
         */
        Point<T>& operator/=(const int &aCardinality);

        /**
         * @brief Setter for coordinates.
         * @param[in] aCoordinates Array of coordinates.
         * @return void
         */
        void SetCoordinates(const T aCoordinates[3]);

        /**
         * @brief Getter for coordinates.
         * @return Pointer to coordinates array.
         */
        const T* GetCoordinates() const;

        /**
         * @brief Setter for cluster.
         * @param[in] aCluster Cluster ID.
         * @return void
         */
        void SetCluster(const int &aCluster);

        /**
         * @brief Getter for cluster.
         * @return Cluster ID.
         */
        int GetCluster() const;

        /**
         * @brief Reset coordinates to zero and set cluster.
         * @param[in] aCluster Cluster ID to set.
         * @return void
         */
        void ResetToZero(const int &aCluster = -1);

        /**
         * @brief Print point information.
         * @return void
         */
        void Print() const;

    private:
        /// Array of coordinates (x, y, z).
        T mCoordinates[3];
        /// Cluster ID.
        int mCluster;
    };

    /**
     * @brief Instantiates the Point class for float and double types.
     * @tparam T Data Type: float or double
     */
    VECCHIAGP_INSTANTIATE_CLASS(Point)

}//namespace vecchia

#endif //VECCHIAGP_POINT_HPP
