
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file LocalClusteringStrategy.hpp
* @version 1.0.0
* @brief Local clustering strategy for Block Vecchia (k-means++/random)
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#ifndef VECCHIAGBCPP_LOCALCLUSTERINGSTRATEGY_HPP
#define VECCHIAGBCPP_LOCALCLUSTERINGSTRATEGY_HPP

#include <data-clustering/ClusteringStrategy.hpp>

namespace vecchia::clustering {

    /**
     * @class LocalClusteringStrategy
     * @brief Local clustering using k-means++ or random (Block Vecchia)
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class LocalClusteringStrategy : public ClusteringStrategy<T> {

    public:

        /**
         * @brief Constructor
         * @param[in] aMaxIter Maximum iterations for k-means
         * @param[in] aNumClusters Number of clusters
         * @param[in] aSeed Random seed
         *
         */
        LocalClusteringStrategy(int aMaxIter, int aNumClusters, int aSeed);

        /**
         * @brief Compute clusters using k-means++ or random
         * @param[in] aLocations Input locations
         * @param[in] aConfigurations Configuration parameters
         * @return ClusteringResult with cluster assignments
         *
         */
        ClusteringResult<T> ComputeClusters(
            dataunits::Locations<T> &aLocations,
            configurations::Configurations &aConfigurations) override;

        /**
         * @brief Get the number of clusters
         * @return Number of clusters
         *
         */
        int GetNumClusters() const override { return this->mNumClusters; }

        /**
         * @brief Parallel K-means clustering algorithm.
         * @param[in,out] aPoints The vector of points to be clustered.
         * @param[in,out] aCentroids The vector of centroids.
         * @param[in] aEpochs The number of iterations.
         * @param[in] aNumberOfClusters The number of clusters.
         * @param[in] aNumThreads Number of threads for parallel execution.
         * @return void
         */
        void KMeansParallel(std::vector<dataunits::Point<T>> &aPoints, 
            std::vector<dataunits::Point<T>> &aCentroids,
            int aEpochs, int aNumberOfClusters, int aNumThreads);

        /**
         * @brief Initialize centroids randomly
         * @param[in] aPoints Input points
         * @return Initial centroids
         *
         */
        std::vector<dataunits::Point<T>> RandomInitializer(
            const std::vector<dataunits::Point<T>> &aPoints);

        /**
         * @brief Compute clusters for prediction - exact code from Predict function
         * Loads test locations from CSV file and performs clustering
         * @param[in] aConfigurations Configuration parameters (contains test locations path)
         * @return ClusteringResult with points and centroids (exact format from Predict)
         *
         */
        ClusteringResult<T> ComputeClustersForPrediction(
            configurations::Configurations &aConfigurations);

    private:


        /**
         * @brief Random clustering algorithm
         * @param[in] aPoints Input points
         * @param[in] aDimension Dimension of points
         * @return Cluster assignments
         *
         */
        std::vector<int> RandomClustering(
            const std::vector<dataunits::Point<T>> &aPoints,
            common::Dimension aDimension);

        /**
         * @brief Count points in each cluster
         * @param[in] aPoints Points with cluster assignments
         * @return Cluster sizes
         *
         */
        std::vector<int> CountClusterSizes(
            const std::vector<dataunits::Point<T>> &aPoints);

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

        /// Clustering method
        std::string mMethod;
        
        /// Maximum iterations for k-means
        int mMaxIter;
        
        /// Random seed
        int mSeed;
    };

    /**
     * @brief Instantiates the LocalClusteringStrategy class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(LocalClusteringStrategy)
}//namespace vecchia

#endif //VECCHIAGBCPP_LOCALCLUSTERINGSTRATEGY_HPP

