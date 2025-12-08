
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file DistributedClusteringStrategy.hpp
* @version 1.0.0
* @brief Distributed clustering strategy for Scaled Block Vecchia with MPI
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#ifndef VECCHIAGBCPP_DISTRIBUTEDCLUSTERINGSTRATEGY_HPP
#define VECCHIAGBCPP_DISTRIBUTEDCLUSTERINGSTRATEGY_HPP

#include <data-clustering/ClusteringStrategy.hpp>

namespace vecchia::clustering {

    /**
     * @class DistributedClusteringStrategy
     * @brief Distributed clustering with MPI partitioning (Scaled Block Vecchia)
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class DistributedClusteringStrategy : public ClusteringStrategy<T> {

    public:

        /**
         * @brief Constructor
         * @param[in] aMethod Clustering method: "kmeans++" or "random"
         * @param[in] aDistanceScale Per-dimension distance scaling
         * @param[in] aNNMultiplier NN multiplier for coarse-to-fine search
         * @param[in] aNumBlocksPerProcess Number of blocks per MPI process
         * @param[in] aMaxIter Maximum iterations for k-means
         * @param[in] aSeed Random seed
         *
         */
        DistributedClusteringStrategy(const std::string &aMethod,
                                     const std::vector<double> &aDistanceScale,
                                     int aNNMultiplier,
                                     int aNumBlocksPerProcess,
                                     int aMaxIter,
                                     int aSeed);

        /**
         * @brief Compute clusters with MPI partitioning
         * @param[in] aLocations Input locations
         * @param[in] aConfigurations Configuration parameters
         * @return ClusteringResult with distributed cluster assignments
         *
         */
        ClusteringResult<T> ComputeClusters(
            dataunits::Locations<T> &aLocations,
            configurations::Configurations &aConfigurations) override;

        /**
         * @brief Get the number of clusters per process
         * @return Number of local clusters
         *
         */
        int GetNumClusters() const override { return mNumBlocksPerProcess; }

    private:

        /**
         * @brief Partition points across MPI ranks by distance scale
         * @param[in] aPoints Input points (all local)
         * @param[out] aPartitionedPoints Output partitioned points
         * @param[in] aDimension Dimension
         *
         */
        void PartitionAcrossRanks(
            const std::vector<dataunits::Point<T>> &aPoints,
            std::vector<dataunits::Point<T>> &aPartitionedPoints,
            common::Dimension aDimension);

        /**
         * @brief Perform local clustering within this rank
         * @param[in] aPoints Local points
         * @param[in] aDimension Dimension
         * @return Local cluster assignments
         *
         */
        std::vector<int> LocalClustering(
            const std::vector<dataunits::Point<T>> &aPoints,
            common::Dimension aDimension);

        /**
         * @brief AllGather cluster centers from all ranks
         * @param[in] aLocalCenters Local centers
         * @param[out] aAllCenters All centers with rank info
         * @param[in] aDimension Dimension
         *
         */
        void AllGatherCenters(
            const std::vector<std::vector<T>> &aLocalCenters,
            std::vector<std::pair<std::vector<T>, int>> &aAllCenters,
            common::Dimension aDimension);

        /**
         * @brief Reorder centers globally
         * @param[in,out] aAllCenters All centers
         * @param[out] aPermutation Global permutation
         *
         */
        void ReorderCenters(
            std::vector<std::pair<std::vector<T>, int>> &aAllCenters,
            std::vector<int> &aPermutation);

        /**
         * @brief K-means++ clustering
         * @param[in] aPoints Input points
         * @param[in] aDimension Dimension
         * @return Cluster assignments
         *
         */
        std::vector<int> KMeansPlusPlusClustering(
            const std::vector<dataunits::Point<T>> &aPoints,
            common::Dimension aDimension);

        /**
         * @brief Random clustering
         * @param[in] aPoints Input points
         * @param[in] aDimension Dimension
         * @return Cluster assignments
         *
         */
        std::vector<int> RandomClustering(
            const std::vector<dataunits::Point<T>> &aPoints,
            common::Dimension aDimension);

        /**
         * @brief Convert locations to points
         * @param[in] aLocations Input locations
         * @param[in] aDimension Dimension
         * @return Vector of points
         *
         */
        std::vector<dataunits::Point<T>> ConvertToPoints(
            dataunits::Locations<T> &aLocations,
            common::Dimension aDimension);

        /**
         * @brief Count cluster sizes
         * @param[in] aPoints Points with assignments
         * @return Cluster sizes
         *
         */
        std::vector<int> CountClusterSizes(
            const std::vector<dataunits::Point<T>> &aPoints);

        /// Clustering method
        std::string mMethod;
        
        /// Per-dimension distance scaling
        std::vector<double> mDistanceScale;
        
        /// NN multiplier for coarse-to-fine search
        int mNNMultiplier;
        
        /// Number of blocks per process
        int mNumBlocksPerProcess;
        
        /// Maximum iterations for k-means
        int mMaxIter;
        
        /// Random seed
        int mSeed;
    };

    /**
     * @brief Instantiates the DistributedClusteringStrategy class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(DistributedClusteringStrategy)
}//namespace vecchia

#endif //VECCHIAGBCPP_DISTRIBUTEDCLUSTERINGSTRATEGY_HPP

