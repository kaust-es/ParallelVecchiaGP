
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file KnnConditioningUpdater.cpp
 * @brief Implementation of the KnnConditioningUpdater class
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <cstring>
#include <cmath>

#include <conditioning-updater/concrete/KnnConditioningUpdater.hpp>
#include <data-units/Locations.hpp>
#include <data-generators/LocationGenerator.hpp>
#include <helpers/DistanceCalculationHelpers.hpp>

using namespace vecchia::configurations;
using namespace vecchia::conditioningupdater::knn;
using namespace vecchia::dataunits;
using namespace vecchia::generators;
using namespace vecchia::helpers;

template<typename T>
KnnConditioningUpdater<T>::KnnConditioningUpdater(Configurations &aConfigurations, VecchiaGBData<T> &aData) {
    // Generate new locations for reordering
    auto *conditioning_locations  = new Locations<T>(aConfigurations.GetConditioningSize() * aData.GetBatchCount(), aConfigurations.GetDimension());
    aData.SetConditioningLocations(*conditioning_locations);
    
    int k = aConfigurations.GetConditioningSize();
    int batchCount = aData.GetBatchCount();
    
    T* h_obs_conditioning = (T *)malloc(batchCount * aConfigurations.GetConditioningSize() * sizeof(T));
    if (!h_obs_conditioning) {
        throw std::runtime_error("Memory allocation failed for conditioning observations");
    }
    aData.SetHostConditioningObs(h_obs_conditioning);
    
    // For Parallel Vecchia: Use the reordered main locations (GetLocations), not NewLocations
    // For Block Vecchia: Use NewLocations which contains the cluster-reordered data
    Locations<T>* source_locations = (aConfigurations.GetVecchiaType() == vecchia::common::PARALLEL_VECCHIA_GP) ? 
                                     aData.GetLocations() : aData.GetNewLocations();
    T* source_observations = (aConfigurations.GetVecchiaType() == vecchia::common::PARALLEL_VECCHIA_GP) ?
                            aData.GetHostObservations() : aData.GetHostObservationsNew();
    
    // Copy the first batch (no conditioning needed for first batch)
    memcpy(aData.GetHostConditioningObs(), source_observations, sizeof(T) * k);
    memcpy(aData.GetConditioningLocations()->GetLocationX(), source_locations->GetLocationX(), sizeof(T) * k);
    memcpy(aData.GetConditioningLocations()->GetLocationY(), source_locations->GetLocationY(), sizeof(T) * k);
    if (aConfigurations.GetDimension() == vecchia::common::Dimension3D)
    {
        memcpy(aData.GetConditioningLocations()->GetLocationZ(), source_locations->GetLocationZ(), sizeof(T) * k);
    }
}

template<typename T>
void KnnConditioningUpdater<T>::Update(Configurations &aConfigurations, VecchiaGBData<T> &aData, int i_block) {
    int k = aConfigurations.GetConditioningSize();
    bool is_parallel = (aConfigurations.GetVecchiaType() == vecchia::common::PARALLEL_VECCHIA_GP);

    // Select source locations and observations based on Vecchia type
    Locations<T>* source_locations = is_parallel ? aData.GetLocations() : aData.GetNewLocations();
    T* source_observations = is_parallel ? aData.GetHostObservations() : aData.GetHostObservationsNew();
    
    T *query_point = (T *)calloc(3, sizeof(T));
    int query_index;
    int l0, l1;  // Range of locations to search
    
    if (is_parallel) {
        // Parallel Vecchia: Each location i finds k nearest from [0..i-1]
        // For batch i_block (i_block >= 1), we're processing location k + i_block - 1
        query_index = k + i_block - 1;
        
        query_point[0] = source_locations->GetLocationX()[query_index];
        query_point[1] = source_locations->GetLocationY()[query_index];
        if (aConfigurations.GetDimension() == vecchia::common::Dimension3D) {
            query_point[2] = source_locations->GetLocationZ()[query_index];
        }
        
        // Search among all previous locations [0..query_index-1]
        l0 = 0;
        l1 = query_index;
    } else {
        // Block Vecchia: Search near centroid
        int centroid_index = i_block + aData.GetFirstClusterCount()[0] - 1;
        
        if (centroid_index < 0 || centroid_index >= aData.GetCentroidsLocations()->GetSize()) {
            throw std::runtime_error("Centroid index " + std::to_string(centroid_index) + 
                                     " out of bounds (size: " + std::to_string(aData.GetCentroidsLocations()->GetSize()) + ")");
        }
        
        query_point[0] = aData.GetCentroidsLocations()->GetLocationX()[centroid_index];
        query_point[1] = aData.GetCentroidsLocations()->GetLocationY()[centroid_index];
        if (aConfigurations.GetDimension() == vecchia::common::Dimension3D) {
            query_point[2] = aData.GetCentroidsLocations()->GetLocationZ()[centroid_index];
        }
        
        // Search in the range before this batch
        l0 = 0;
        l1 = aData.GetBatchNumAccum()[i_block];
    }
    
    if (l1 <= l0) {
        free(query_point);
        return;
    }
    
    T *distances = (T *)malloc(sizeof(T) * (l1 - l0));
    int *indices = (int *)malloc(sizeof(int) * (l1 - l0));
    
    // Compute distances to all candidates
    for (int j = l0; j < l1; j++) {
        if (j >= source_locations->GetSize()) {
            throw std::runtime_error("Index " + std::to_string(j) + 
                                     " out of bounds for locations (size: " + std::to_string(source_locations->GetSize()) + ")");
        }
        
        T distance;
        if (aConfigurations.GetDistanceMetric() == common::GREAT_CIRCLE_DISTANCE) {
            distance = DistanceCalculationHelpers<T>::DistanceEarth(
                query_point[0], query_point[1], 
                source_locations->GetLocationX()[j], source_locations->GetLocationY()[j]);
            if (aConfigurations.GetDimension() == vecchia::common::Dimension3D) {
                T dz = query_point[2] - source_locations->GetLocationZ()[j];
                distance = sqrt(distance * distance + dz * dz);
            }
        } else {
            // Euclidean distance
            T dx = query_point[0] - source_locations->GetLocationX()[j];
            T dy = query_point[1] - source_locations->GetLocationY()[j];
            distance = sqrt(dx * dx + dy * dy);
            if (aConfigurations.GetDimension() == vecchia::common::Dimension3D) {
                T dz = query_point[2] - source_locations->GetLocationZ()[j];
                distance = sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        distances[j - l0] = distance;
        indices[j - l0] = j;
    }
    int actual_k = (k > (l1 - l0)) ? (l1 - l0) : k;
    // Selection sort to find k nearest neighbors
    for (int sort_i = 0; sort_i < actual_k; sort_i++) {
        int min_idx = sort_i;
        for (int sort_j = sort_i + 1; sort_j < l1 - l0; sort_j++) {
            if (distances[sort_j] < distances[min_idx]) {
                min_idx = sort_j;
            }
        }
        // Swap
        T temp_dist = distances[sort_i];
        distances[sort_i] = distances[min_idx];
        distances[min_idx] = temp_dist;
        
        int temp_idx = indices[sort_i];
        indices[sort_i] = indices[min_idx];
        indices[min_idx] = temp_idx;
    }
    
    // Store the k nearest neighbors
    // Note: i_block starts from 1 (not 0) because first k elements are the initial block
    int offset = i_block * k;
    
    for (int sort_i = 0; sort_i < actual_k; sort_i++) {
        aData.GetConditioningLocations()->GetLocationX()[offset + sort_i] = source_locations->GetLocationX()[indices[sort_i]];
        aData.GetConditioningLocations()->GetLocationY()[offset + sort_i] = source_locations->GetLocationY()[indices[sort_i]];
        if (aConfigurations.GetDimension() == vecchia::common::Dimension3D) {
            aData.GetConditioningLocations()->GetLocationZ()[offset + sort_i] = source_locations->GetLocationZ()[indices[sort_i]];
        }
        aData.GetHostConditioningObs()[offset + sort_i] = source_observations[indices[sort_i]];
    }
    free(distances);
    free(indices);
    free(query_point);
}

