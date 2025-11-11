
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file NoClusteringStrategy.hpp
* @version 1.0.0
* @brief No clustering strategy for Scalar Vecchia (point-wise)
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#ifndef VECCHIAGBCPP_NOCLUSTERINGSTRATEGY_HPP
#define VECCHIAGBCPP_NOCLUSTERINGSTRATEGY_HPP

#include <data-clustering/ClusteringStrategy.hpp>

namespace vecchia::clustering {

    /**
     * @class NoClusteringStrategy
     * @brief No clustering - each point is its own cluster (Scalar Vecchia)
     * @tparam T Data Type: float or double
     *
     */
    template<typename T>
    class NoClusteringStrategy : public ClusteringStrategy<T> {

    public:

        /**
         * @brief Constructor
         * @param[in] aNumPoints Total number of points
         *
         */
        explicit NoClusteringStrategy(int aNumPoints);

        /**
         * @brief Compute clusters (trivial - each point is its own cluster)
         * @param[in] aLocations Input locations
         * @param[in] aConfigurations Configuration parameters
         * @return ClusteringResult with point-wise assignments
         *
         */
        ClusteringResult<T> ComputeClusters(
            dataunits::Locations<T> &aLocations,
            configurations::Configurations &aConfigurations) override;

        /**
         * @brief Get the number of clusters (equals number of points)
         * @return Number of clusters
         *
         */
        int GetNumClusters() const override { return this->mNumClusters; }

    private:
        /// Total number of points
        int mNumPoints;
    };

    /**
     * @brief Instantiates the NoClusteringStrategy class for float and double types.
     * @tparam T Data Type: float or double
     *
     */
    VECCHIAGP_INSTANTIATE_CLASS(NoClusteringStrategy)
}//namespace vecchia

#endif //VECCHIAGBCPP_NOCLUSTERINGSTRATEGY_HPP

