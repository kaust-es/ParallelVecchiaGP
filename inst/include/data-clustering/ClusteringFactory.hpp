
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file ClusteringFactory.hpp
* @version 1.0.0
* @brief Factory for creating clustering strategies
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#ifndef VECCHIAGBCPP_CLUSTERINGFACTORY_HPP
#define VECCHIAGBCPP_CLUSTERINGFACTORY_HPP

#include <memory>

#include <data-clustering/ClusteringStrategy.hpp>
#include <data-clustering/concrete/NoClusteringStrategy.hpp>
#include <data-clustering/concrete/LocalClusteringStrategy.hpp>
#include <data-clustering/concrete/DistributedClusteringStrategy.hpp>
#include <configurations/Configurations.hpp>
#include <common/Definitions.hpp>

namespace vecchia::clustering {

    /**
     * @class ClusteringFactory
     * @brief Factory for creating appropriate clustering strategy based on Vecchia type
     *
     */
    class ClusteringFactory {

    public:

        /**
         * @brief Create clustering strategy based on Vecchia type
         * @tparam T Data Type: float or double
         * @param[in] aVecchiaType Type of Vecchia approximation
         * @param[in] aConfigurations Configuration parameters
         * @return Unique pointer to clustering strategy
         *
         */
        template<typename T>
        static std::unique_ptr<ClusteringStrategy<T>> Create(
            common::VecchiaType aVecchiaType,
            configurations::Configurations &aConfigurations) {
            
            switch (aVecchiaType) {
                
                case common::VecchiaType::PARALLEL_VECCHIA_GP: {
                    // Scalar Vecchia - no clustering, point-wise
                    return std::make_unique<NoClusteringStrategy<T>>(
                        aConfigurations.GetProblemSize());
                }
                
                case common::VecchiaType::PARALLEL_BLOCK_VECCHIA_GP: {
                    // Block Vecchia - local clustering
                    return std::make_unique<LocalClusteringStrategy<T>>(
                        aConfigurations.GetKMeansMaxIter(),
                        aConfigurations.GetBlockSize(),
                        aConfigurations.GetSeed());
                }
                
                case common::VecchiaType::PARALLEL_SCALED_BLOCK_VECCHIA_GP: {
                    // Scaled Block Vecchia - distributed clustering
                    return std::make_unique<DistributedClusteringStrategy<T>>(
                        aConfigurations.GetClusteringMethod(),
                        aConfigurations.GetDistanceScale(),
                        aConfigurations.GetNNMultiplier(),
                        aConfigurations.GetBlockSize(),
                        aConfigurations.GetKMeansMaxIter(),
                        aConfigurations.GetSeed());
                }
                
                default:
                    throw std::runtime_error("Unknown VecchiaType in ClusteringFactory");
            }
        }
    };
}//namespace vecchia

#endif //VECCHIAGBCPP_CLUSTERINGFACTORY_HPP

