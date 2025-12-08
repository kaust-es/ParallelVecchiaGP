
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file ClusteringStrategy.hpp
* @version 1.0.0
* @brief Abstract base class for clustering strategies in Vecchia approximation
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#ifndef VECCHIAGBCPP_CLUSTERINGSTRATEGY_HPP
#define VECCHIAGBCPP_CLUSTERINGSTRATEGY_HPP

#include <memory>
#include <vector>

#include <common/Definitions.hpp>
#include <data-units/Locations.hpp>
#include <data-units/Point.hpp>
#include <data-units/BlockInfo.hpp>
#include <configurations/Configurations.hpp>
#include <utilities/TimingData.hpp>

namespace vecchia::clustering {

    /**
     * @struct ClusteringResult
     * @brief Result of clustering operation containing assignments and metadata
     *
     */
    template<typename T>
    struct ClusteringResult {
        /// Point to cluster assignment (size = num_points)
        std::vector<int> assignments;
        
        /// Size of each cluster/batch (size = num_clusters)
        std::vector<int> batchSizes;
        
        /// Total number of clusters
        int numClusters;
        
        /// True if point-wise (Scalar Vecchia), false if clustered
        bool isPointWise;
        
        /// Cluster centroids
        std::unique_ptr<dataunits::Locations<T>> centroids;
        
        /// Points with cluster assignments
        std::vector<dataunits::Point<T>> points;
        
        /// Block information for Scaled Block Vecchia (optional)
        std::vector<dataunits::BlockInfo> blockInfos;
        
        /// Block information for test/prediction data (optional)
        std::vector<dataunits::BlockInfo> blockInfos_test;
        
        /// Timing information from clustering operations
        utilities::TimingData timingData;
        
        ClusteringResult() : numClusters(0), isPointWise(false) {}
    };

    /**
     * @class ClusteringStrategy
     * @brief Abstract base class for clustering strategies
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class ClusteringStrategy {

    public:

        /**
         * @brief Virtual destructor
         *
         */
        virtual ~ClusteringStrategy() = default;

        /**
         * @brief Compute clusters for the given locations
         * @param[in] aLocations Input locations to cluster
         * @param[in] aConfigurations Configuration parameters
         * @return ClusteringResult containing assignments and metadata
         *
         */
        virtual ClusteringResult<T> ComputeClusters(
            dataunits::Locations<T> &aLocations,
            configurations::Configurations &aConfigurations) = 0;

        /**
         * @brief Get the number of clusters
         * @return Number of clusters
         *
         */
        virtual int GetNumClusters() const = 0;

    protected:
        /// Number of clusters
        int mNumClusters;
    };

    /**
     * @brief Instantiates the ClusteringStrategy class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(ClusteringStrategy)
    VECCHIAGP_INSTANTIATE_CLASS(ClusteringResult)
}//namespace vecchia

#endif //VECCHIAGBCPP_CLUSTERINGSTRATEGY_HPP

