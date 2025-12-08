// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file BatchPreparationUtility.hpp
 * @brief Utility class for preparing batches from clustering results
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-20
**/

#ifndef VECCHIAGP_BATCHPREPARATIONUTILITY_HPP
#define VECCHIAGP_BATCHPREPARATIONUTILITY_HPP

#include <data-clustering/ClusteringStrategy.hpp>
#include <data-units/VecchiaGBData.hpp>
#include <configurations/Configurations.hpp>
#include <common/Definitions.hpp>

namespace vecchia::helpers {

    /**
     * @brief Utility class for converting clustering results to VecchiaGBData structures
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class BatchPreparationUtility {
    public:
        /**
         * @brief Process clustering results and prepare batch operations
         * @param[in] aConfigurations Configuration parameters
         * @param[in,out] aData Data structure to populate
         * @param[in,out] aClusteringResult Results from clustering strategy
         * @return void
         *
         */
        static void PrepareBatchesFromClustering(
            configurations::Configurations &aConfigurations,
            VecchiaGBData<T> &aData,
            clustering::ClusteringResult<T> &aClusteringResult);

    private:
        /**
         * @brief Populate centroid locations in data structure
         */
        static void PopulateCentroids(
            configurations::Configurations &aConfigurations,
            VecchiaGBData<T> &aData,
            clustering::ClusteringResult<T> &aClusteringResult);

        /**
         * @brief Apply reordering to centroids and update permutation indices
         */
        static void ReorderCentroids(
            configurations::Configurations &aConfigurations,
            VecchiaGBData<T> &aData,
            clustering::ClusteringResult<T> &aClusteringResult);

        /**
         * @brief Calculate first cluster size and batch count
         */
        static void CalculateBatchInfo(
            configurations::Configurations &aConfigurations,
            VecchiaGBData<T> &aData,
            clustering::ClusteringResult<T> &aClusteringResult,
            std::vector<int> &aClusterCounts);

        /**
         * @brief Combine first clusters if needed
         */
        static void CombineFirstClusters(
            VecchiaGBData<T> &aData,
            clustering::ClusteringResult<T> &aClusteringResult,
            std::vector<int> &aClusterCounts);

        /**
         * @brief Prepare batch arrays and accumulators
         */
        static void PrepareBatchArrays(
            VecchiaGBData<T> &aData,
            const std::vector<int> &aClusterCounts);

        /**
         * @brief Reorder locations and observations into batches
         */
        static void ReorderDataIntoBatches(
            configurations::Configurations &aConfigurations,
            VecchiaGBData<T> &aData,
            clustering::ClusteringResult<T> &aClusteringResult);
    };

    /**
     * @brief Instantiates the BatchPreparationUtility class for float and double types.
     * @tparam T Data Type: float or double
     */
    VECCHIAGP_INSTANTIATE_CLASS(BatchPreparationUtility)

}//namespace vecchia::helpers

#endif //VECCHIAGP_BATCHPREPARATIONUTILITY_HPP

