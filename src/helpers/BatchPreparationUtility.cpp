// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file BatchPreparationUtility.cpp
 * @brief Implementation of BatchPreparationUtility
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-20
**/

#include <helpers/BatchPreparationUtility.hpp>
#include <helpers/LocationSorter.hpp>
#include <data-generators/LocationGenerator.hpp>
#include <utilities/Logger.hpp>
#include <cmath>
#include <cstdlib>

using namespace vecchia::helpers;
using namespace vecchia::configurations;
using namespace vecchia::dataunits;
using namespace vecchia::clustering;
using namespace vecchia::generators;
using namespace vecchia::common;

template<typename T>
void BatchPreparationUtility<T>::PrepareBatchesFromClustering(
    Configurations &aConfigurations,
    VecchiaGBData<T> &aData,
    ClusteringResult<T> &aClusteringResult) {
    
    // Step 1: Populate centroid locations
    PopulateCentroids(aConfigurations, aData, aClusteringResult);
    
    // Step 2: Reorder centroids based on permutation method
    ReorderCentroids(aConfigurations, aData, aClusteringResult);
    
    // Step 3: Count points in each cluster
    std::vector<int> clusterCounts(aClusteringResult.numClusters, 0);
    for (const auto &point : aClusteringResult.points) {
        clusterCounts[point.GetCluster()]++;
    }
    
    // Step 4: Calculate first cluster size and batch count
    CalculateBatchInfo(aConfigurations, aData, aClusteringResult, clusterCounts);
    
    // Step 5: Combine first clusters if needed
    CombineFirstClusters(aData, aClusteringResult, clusterCounts);
    
    // Step 6: Prepare batch arrays
    PrepareBatchArrays(aData, clusterCounts);
    
    // Step 7: Reorder data into batches
    ReorderDataIntoBatches(aConfigurations, aData, aClusteringResult);
}

template<typename T>
void BatchPreparationUtility<T>::PopulateCentroids(
    Configurations &aConfigurations,
    VecchiaGBData<T> &aData,
    ClusteringResult<T> &aClusteringResult) {
    
    int numClusters = aClusteringResult.numClusters;
    auto &centroids = aClusteringResult.centroids;
    
    for (int i = 0; i < numClusters; i++) {
        aData.GetCentroidsLocations()->GetLocationX()[i] = centroids->GetLocationX()[i];
        aData.GetCentroidsLocations()->GetLocationY()[i] = centroids->GetLocationY()[i];
        if (aConfigurations.GetDimension() != Dimension2D) {
            aData.GetCentroidsLocations()->GetLocationZ()[i] = centroids->GetLocationZ()[i];
        }
        aData.GetPremIndex()[i] = i;  // Initial ordering before permutation
    }
}

template<typename T>
void BatchPreparationUtility<T>::ReorderCentroids(
    Configurations &aConfigurations,
    VecchiaGBData<T> &aData,
    ClusteringResult<T> &aClusteringResult) {
    
    int numClusters = aClusteringResult.numClusters;
    auto &centroids = aClusteringResult.centroids;
    
    if(aConfigurations.GetDimension() == Dimension2D) {
        LOGGER("-------You are using the 2D ordering.-------");
    } else {
        LOGGER("------You are using the 3D ordering.------");
    }
    
    // Apply the reordering method to centroids
    LocationSorter<T>::GetInstance().ApplyReordering(
        aConfigurations.GetPermutation(), 
        numClusters, 
        aConfigurations.GetDimension(), 
        *aData.GetCentroidsLocations());
    
    // Find the mapping from reordered centroids to original indices
    for (int i = 0; i < numClusters; i++) {
        T coords[3];
        coords[0] = aData.GetCentroidsLocations()->GetLocationX()[i];
        coords[1] = aData.GetCentroidsLocations()->GetLocationY()[i];
        coords[2] = (aConfigurations.GetDimension() != Dimension2D) ? 
                    aData.GetCentroidsLocations()->GetLocationZ()[i] : 0;
        
        // Find matching centroid in original list
        for (int j = 0; j < numClusters; j++) {
            T dist = std::abs(centroids->GetLocationX()[j] - coords[0]) +
                    std::abs(centroids->GetLocationY()[j] - coords[1]);
            
            if (aConfigurations.GetDimension() != Dimension2D) {
                dist += std::abs(centroids->GetLocationZ()[j] - coords[2]);
            }
            
            if (dist < 1e-10) {
                aData.GetPremIndex()[i] = j;
                break;
            }
        }
    }
}

template<typename T>
void BatchPreparationUtility<T>::CalculateBatchInfo(
    Configurations &aConfigurations,
    VecchiaGBData<T> &aData,
    ClusteringResult<T> &aClusteringResult,
    std::vector<int> &aClusterCounts) {
    
    int numClusters = aClusteringResult.numClusters;
    
    // Calculate first cluster size to meet conditioning requirement
    int sizeFirstClusters = 0;
    int firstClusterCount = 0;
    for (int i = 0; i < numClusters; ++i) {
        sizeFirstClusters += aClusterCounts[aData.GetPremIndex()[i]];
        if (sizeFirstClusters >= aConfigurations.GetConditioningSize()) {
            firstClusterCount = i + 1;
            break;
        }
    }
    
    aData.GetFirstClusterCount()[0] = firstClusterCount;
    int batchCount = numClusters - firstClusterCount + 1;
    aData.SetBatchCount(batchCount);
    
    LOGGER("First cluster count: " << firstClusterCount)
    LOGGER("Batch count: " << batchCount)
}

template<typename T>
void BatchPreparationUtility<T>::CombineFirstClusters(
    VecchiaGBData<T> &aData,
    ClusteringResult<T> &aClusteringResult,
    std::vector<int> &aClusterCounts) {
    
    int firstClusterCount = aData.GetFirstClusterCount()[0];
    
    if (firstClusterCount > 1) {
        int firstClusterIdx = aData.GetPremIndex()[0];
        int totalSize = 0;
        
        // Combine all first clusters into one
        for (int i = 0; i < firstClusterCount; i++) {
            int clusterIdx = aData.GetPremIndex()[i];
            totalSize += aClusterCounts[clusterIdx];
            
            // Reassign all points from this cluster to the first cluster
            for (auto &point : aClusteringResult.points) {
                if (point.GetCluster() == clusterIdx) {
                    point.SetCluster(firstClusterIdx);
                }
            }
            
            aData.GetPremIndex()[i] = firstClusterIdx;
        }
        
        aClusterCounts[firstClusterIdx] = totalSize;
    }
}

template<typename T>
void BatchPreparationUtility<T>::PrepareBatchArrays(
    VecchiaGBData<T> &aData,
    const std::vector<int> &aClusterCounts) {
    
    int batchCount = aData.GetBatchCount();
    int firstClusterCount = aData.GetFirstClusterCount()[0];
    
    // Allocate batch number array
    aData.SetBatchNum((int *)calloc(batchCount + 1, sizeof(int)));
    for (int i = 0; i < batchCount; ++i) {
        aData.GetBatchNum()[i] = aClusterCounts[aData.GetPremIndex()[i + firstClusterCount - 1]];
    }
    
    // Calculate accumulated batch sizes
    aData.SetBatchNumAccum((int *)calloc(batchCount + 1, sizeof(int)));
    aData.SetBatchNumSquareAccum((int *)calloc(batchCount + 1, sizeof(int)));
    
    for (int i = 1; i < (batchCount + 1); ++i) {
        aData.GetBatchNumAccum()[i] = aData.GetBatchNumAccum()[i - 1] + aData.GetBatchNum()[i - 1];
        aData.GetBatchNumSquareAccum()[i] = aData.GetBatchNumSquareAccum()[i - 1] + 
                                             aData.GetBatchNum()[i - 1] * aData.GetBatchNum()[i - 1];
    }
}

template<typename T>
void BatchPreparationUtility<T>::ReorderDataIntoBatches(
    Configurations &aConfigurations,
    VecchiaGBData<T> &aData,
    ClusteringResult<T> &aClusteringResult) {
    
    int batchCount = aData.GetBatchCount();
    int firstClusterCount = aData.GetFirstClusterCount()[0];
    
    // Allocate new observation array
    T *hostObservationsNew = (T *)malloc(aConfigurations.GetProblemSize() * sizeof(T));
    aData.SetHostObservationsNew(hostObservationsNew);
    
    // Generate new locations structure for reordering
    auto *locationsNew = new Locations<T>(aConfigurations.GetProblemSize(), 
                                          aConfigurations.GetDimension());
    LocationGenerator<T>::GenerateLocations(
        aConfigurations.GetProblemSize(), 
        aConfigurations.GetTimeSlot(), 
        aConfigurations.GetDimension(), 
        *locationsNew);
    aData.SetNewLocations(*locationsNew);
    
    // Reorder clusters into batches
    int *batchIndex = aData.GetPremIndex() + firstClusterCount - 1;
    LocationGenerator<T>::ClusterToBatch(
        aConfigurations.GetProblemSize(), 
        batchCount, 
        aData.GetBatchNum(), 
        aData.GetBatchNumAccum(), 
        batchIndex, 
        *aData.GetLocations(), 
        aData.GetHostObservations(), 
        *locationsNew, 
        hostObservationsNew, 
        aClusteringResult.points, 
        *aData.GetCentroidsLocations(), 
        aConfigurations.GetDimension() != Dimension2D);
    
    LOGGER("--------------Reordering Done-----------------")
}

