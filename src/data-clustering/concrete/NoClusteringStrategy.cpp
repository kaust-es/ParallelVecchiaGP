
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file NoClusteringStrategy.cpp
* @version 1.0.0
* @brief Implementation of NoClusteringStrategy
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#include <numeric>

#include <data-clustering/concrete/NoClusteringStrategy.hpp>
#include <utilities/Logger.hpp>

using namespace vecchia::clustering;
using namespace vecchia::common;
using namespace vecchia::dataunits;
using namespace vecchia::configurations;

template<typename T>
NoClusteringStrategy<T>::NoClusteringStrategy(int aNumPoints) 
    : mNumPoints(aNumPoints) {
    this->mNumClusters = aNumPoints;
}

template<typename T>
ClusteringResult<T> NoClusteringStrategy<T>::ComputeClusters(
    Locations<T> &aLocations,
    Configurations &aConfigurations) {
    
    LOGGER("** NoClusteringStrategy: Point-wise (Scalar Vecchia) **")
    
    ClusteringResult<T> result;
    
    // Each point is its own cluster: [0, 1, 2, ..., N-1]
    result.assignments.resize(mNumPoints);
    std::iota(result.assignments.begin(), result.assignments.end(), 0);
    
    // For Scalar Vecchia with conditioning size cs:
    // batchCount = num_loc - cs + 1
    // Each batch has size 1 (point-wise)
    int cs = aConfigurations.GetConditioningSize();
    int batchCount = mNumPoints - cs + 1;
    
    result.batchSizes.resize(batchCount, 1);  // All batches are size 1
    result.numClusters = mNumPoints;
    result.isPointWise = true;  // Important flag for backend adapter
    
    // Centroids are the points themselves
    result.centroids = std::make_unique<Locations<T>>(mNumPoints, aLocations.GetDimension());
    for (int i = 0; i < mNumPoints; i++) {
        result.centroids->GetLocationX()[i] = aLocations.GetLocationX()[i];
        result.centroids->GetLocationY()[i] = aLocations.GetLocationY()[i];
        if (aLocations.GetDimension() == Dimension3D || aLocations.GetDimension() == DimensionST) {
            result.centroids->GetLocationZ()[i] = aLocations.GetLocationZ()[i];
        }
    }
    
    // Convert to Points for compatibility
    result.points.reserve(mNumPoints);
    for (int i = 0; i < mNumPoints; i++) {
        Point<T> point;
        T coords[3] = {aLocations.GetLocationX()[i], aLocations.GetLocationY()[i], 0};
        if (aLocations.GetDimension() == Dimension3D || aLocations.GetDimension() == DimensionST) {
            coords[2] = aLocations.GetLocationZ()[i];
        }
        point.SetCoordinates(coords);
        point.SetCluster(i);  // Each point is its own cluster
        result.points.push_back(point);
    }
    
    LOGGER("** Point-wise clustering complete: " << mNumPoints << " points, " 
           << batchCount << " batches **")
    
    return result;
}

