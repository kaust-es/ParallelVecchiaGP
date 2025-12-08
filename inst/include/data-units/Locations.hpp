
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file Locations.hpp
 * @brief Header file for the Locations class, which contains methods to set and get location data.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_LOCATIONS_HPP
#define VECCHIAGP_LOCATIONS_HPP

#include <common/Definitions.hpp>

namespace vecchia::dataunits {

    // TODO (Option A): Full N-dimensional refactor
    // Current implementation: Locations stores exactly 3 arrays (X, Y, Z)
    // Proposed refactor for arbitrary N dimensions:
    // - Add: T** mpCoordinates (array of N coordinate arrays)
    // - Add: int mDimensionSize (number of dimensions)
    // - Replace: GetLocationX/Y/Z() with GetCoordinate(int dim_idx)
    // - Update: ~30 files that use GetLocationX/Y/Z()
    // Impact: All Parallel/Block Vecchia code, clustering, KNN, distance calculations
    // For now: Scaled Block Vecchia will manage its own N-dimensional arrays

    /**
     * @class Locations
     * @brief A class containing methods to set and get location data.
     * @tparam T Data Type: float or double
     */
    template<typename T>
    class Locations {
    public:
        /**
         * @brief Constructor.
         * @param[in] aSize The number of data points.
         * @param[in] aDimension The dimensionality of the data points.
         * @return void
         *
         */
        Locations(const int &aSize, const vecchia::common::Dimension &aDimension);

        /**
         * @brief Default copy constructor.
         * @param[in] aLocations Locations to be copied.
         *
         */
        Locations(const Locations<T> &aLocations) = default;

        /**
         * @brief destructor for Locations.
         *
         */
        ~Locations();

        /**
         * @brief Setter for LocationX.
         * @param[in] aLocationX Reference to X data.
         * @return void
         *
         */
        void SetLocationX(T &aLocationX, const int &aSize);

        /**
         * @brief Getter for LocationX.
         * @return Pointer to X data.
         *
         */
        T *GetLocationX();

        /**
         * @brief Setter for LocationY.
         * @param[in] aLocationY Reference to Y data.
         * @return void
         *
         */
        void SetLocationY(T &aLocationY, const int &aSize);

        /**
         * @brief Getter for LocationY.
         * @return Pointer to Y data.
         *
         */
        T *GetLocationY();

        /**
         * @brief Setter for LocationZ.
         * @param[in] aLocationZ Reference to Z data.
         * @return void
         *
         */
        void SetLocationZ(T &aLocationZ, const int &aSize);

        /**
         * @brief Getter for LocationZ.
         * @return Pointer to Z data.
         *
         */
        T *GetLocationZ();

        /**
         * @brief Setter for mSize.
         * @param[in] aSize.
         * @return void
         *
         */
        void SetSize(const int &aSize);

        /**
         * @brief Getter for mSize.
         * @return Locations size.
         *
         */
        int GetSize();

        /**
         * @brief Setter for Dimensions.
         * @param[in] aDimension.
         * @return void
         *
         */
        void SetDimension(const common::Dimension &aDimension);

        /**
         * @brief Getter for Dimension.
         * @return Locations dimension.
         *
         */
        common::Dimension GetDimension();

    private:
        /// Pointer to X data.
        T *mpLocationX = nullptr;
        /// Pointer to Y data.
        T *mpLocationY = nullptr;
        /// Pointer to Z data.
        T *mpLocationZ = nullptr;
        /// Size of each dimension
        int mSize = 1;
        /// Data dimensions
        common::Dimension mDimension = common::Dimension2D;
    };

    /**
    * @brief Instantiates the Linear Algebra methods class for float and double types.
    * @tparam T Data Type: float or double
    *
    */
    VECCHIAGP_INSTANTIATE_CLASS(Locations)
}//namespace vecchia

#endif //VECCHIAGP_LOCATIONS_HPP