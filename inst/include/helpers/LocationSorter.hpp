
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file LocationSorter.hpp
 * @brief Provides location sorting and reordering methods for spatial statistics.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-18
**/

#ifndef VECCHIAGP_LOCATIONSORTER_HPP
#define VECCHIAGP_LOCATIONSORTER_HPP

#include <data-units/Locations.hpp>
#include <configurations/Configurations.hpp>
#include <common/Definitions.hpp>

namespace vecchia::helpers {

    /**
     * @class LocationSorter
     * @brief A singleton class that provides various location sorting and reordering methods.
     * @tparam T Data Type: float or double
     */
    template<typename T>
    class LocationSorter {

    public:
        /**
         * @brief Get the singleton instance of LocationSorter.
         * @return Reference to the singleton instance.
         */
        static LocationSorter<T>& GetInstance();

        /**
         * @brief Apply sorting/reordering based on the ordering method.
         * @param[in] aOrderingMethod The ordering method to apply.
         * @param[in] aN The number of locations.
         * @param[in] aDimension The dimension of the locations.
         * @param[in,out] aLocations Locations to be sorted/reordered.
         * @param[in,out] aObservations Optional observations array to be reordered along with locations (default: nullptr).
         */
        void ApplyReordering(const common::OrderingMethod &aOrderingMethod, const int &aN,
                           const common::Dimension &aDimension, dataunits::Locations<T> &aLocations, 
                           T *aObservations = nullptr);

        /**
         * @brief Random reordering of locations.
         * @param[in] aN The number of locations.
         * @param[in] aDimension The dimension of the locations.
         * @param[in,out] aLocations Locations to be reordered.
         * @param[in,out] aObservations Optional observations array to be reordered along with locations.
         */
        void RandomReordering(const int &aN, const common::Dimension &aDimension,
                            dataunits::Locations<T> &aLocations, T *aObservations = nullptr);

        /**
         * @brief Morton (Z-order) reordering of locations.
         * @param[in] aN The number of locations.
         * @param[in] aDimension The dimension of the locations.
         * @param[in,out] aLocations Locations to be reordered.
         * @param[in,out] aObservations Optional observations array to be reordered along with locations.
         */
        void MortonReordering(const int &aN, const common::Dimension &aDimension,
                            dataunits::Locations<T> &aLocations, T *aObservations = nullptr);

        /**
         * @brief KD-Tree reordering of locations.
         * @param[in] aN The number of locations.
         * @param[in] aDimension The dimension of the locations.
         * @param[in,out] aLocations Locations to be reordered.
         * @param[in,out] aObservations Optional observations array to be reordered along with locations.
         */
        void KDTreeReordering(const int &aN, const common::Dimension &aDimension,
                            dataunits::Locations<T> &aLocations, T *aObservations = nullptr);

        /**
         * @brief Hilbert curve reordering of locations.
         * @param[in] aN The number of locations.
         * @param[in] aDimension The dimension of the locations.
         * @param[in,out] aLocations Locations to be reordered.
         * @param[in,out] aObservations Optional observations array to be reordered along with locations.
         */
        void HilbertReordering(const int &aN, const common::Dimension &aDimension,
                             dataunits::Locations<T> &aLocations, T *aObservations = nullptr);

        /**
         * @brief Maximum Minimum Distance (MMD) reordering of locations.
         * @param[in] aN The number of locations.
         * @param[in] aDimension The dimension of the locations.
         * @param[in,out] aLocations Locations to be reordered.
         * @param[in,out] aObservations Optional observations array to be reordered along with locations.
         */
        void MMDReordering(const int &aN, const common::Dimension &aDimension,
                         dataunits::Locations<T> &aLocations, T *aObservations = nullptr);

    private:
        // Private constructor for singleton pattern
        LocationSorter() = default;
        ~LocationSorter() = default;

        // Delete copy constructor and assignment operator
        LocationSorter(const LocationSorter&) = delete;
        LocationSorter& operator=(const LocationSorter&) = delete;

        // Helper structures for sorting
        struct TreeNode2D {
            int dim;
            T x, y;
            TreeNode2D *left, *right;
        };

        struct TreeNode3D {
            int dim;
            T x, y, z;
            TreeNode3D *left, *right;
        };

        // Helper methods for Morton encoding/decoding
        static uint32_t Part1By1(uint32_t x);
        static uint32_t Compact1By1(uint32_t x);
        static uint32_t EncodeMorton2(uint32_t x, uint32_t y);
        static uint32_t DecodeMorton2X(uint32_t code);
        static uint32_t DecodeMorton2Y(uint32_t code);

        static uint64_t Part1By3(uint64_t x);
        static uint64_t Compact1By3(uint64_t x);
        static uint64_t EncodeMorton3(uint64_t x, uint64_t y, uint64_t z);
        static uint64_t DecodeMorton3X(uint64_t code);
        static uint64_t DecodeMorton3Y(uint64_t code);
        static uint64_t DecodeMorton3Z(uint64_t code);

        // Helper methods for Hilbert encoding/decoding
        static uint32_t EncodeHilbert2(uint32_t x, uint32_t y);
        static void DecodeHilbert2(uint32_t result, uint32_t &x, uint32_t &y);
        static uint64_t EncodeHilbert3(uint64_t x, uint64_t y, uint64_t z);
        static void DecodeHilbert3(uint64_t result, uint64_t &x, uint64_t &y, uint64_t &z);

        // Helper methods for KD-Tree
        TreeNode2D* BuildKDTree2D(std::vector<std::tuple<T, T>>& data, int depth);
        void TraverseKDTree2D(TreeNode2D* root, std::vector<std::tuple<T, T>>& result);
        void FreeKDTree2D(TreeNode2D* root);

        TreeNode3D* BuildKDTree3D(std::vector<std::tuple<T, T, T>>& data, int depth);
        void TraverseKDTree3D(TreeNode3D* root, std::vector<std::tuple<T, T, T>>& result);
        void FreeKDTree3D(TreeNode3D* root);
    };

    /**
     * @brief Instantiates the LocationSorter class for float and double types.
     * @tparam T Data Type: float or double
     */
    VECCHIAGP_INSTANTIATE_CLASS(LocationSorter)

}//namespace vecchia::helpers

#endif //VECCHIAGP_LOCATIONSORTER_HPP

