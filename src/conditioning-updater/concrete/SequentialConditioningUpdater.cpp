
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file SequentialConditioningUpdater.cpp
 * @brief Implementation of the SequentialConditioningUpdater class
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <cstring>
#include <conditioning-updater/concrete/SequentialConditioningUpdater.hpp>
#include <data-units/Locations.hpp>

using namespace vecchia::configurations;
using namespace vecchia::conditioningupdater::sequential;
using namespace vecchia::dataunits;
using namespace vecchia::common;

template<typename T>
SequentialConditioningUpdater<T>::SequentialConditioningUpdater(Configurations &aConfigurations, VecchiaGBData<T> &aData) {
    // Generate new locations for reordering (same initialization as KnnConditioningUpdater)
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
    Locations<T>* source_locations = (aConfigurations.GetVecchiaType() == PARALLEL_VECCHIA_GP) ? 
                                     aData.GetLocations() : aData.GetNewLocations();
    T* source_observations = (aConfigurations.GetVecchiaType() == PARALLEL_VECCHIA_GP) ?
                            aData.GetHostObservations() : aData.GetHostObservationsNew();
    
    // Copy the first batch (no conditioning needed for first batch)
    memcpy(aData.GetHostConditioningObs(), source_observations, sizeof(T) * k);
    memcpy(aData.GetConditioningLocations()->GetLocationX(), source_locations->GetLocationX(), sizeof(T) * k);
    memcpy(aData.GetConditioningLocations()->GetLocationY(), source_locations->GetLocationY(), sizeof(T) * k);
    if (aConfigurations.GetDimension() == Dimension3D)
    {
        memcpy(aData.GetConditioningLocations()->GetLocationZ(), source_locations->GetLocationZ(), sizeof(T) * k);
    }
}

template<typename T>
void SequentialConditioningUpdater<T>::Update(Configurations &aConfigurations, VecchiaGBData<T> &aData, int i_block) {
    int k = aConfigurations.GetConditioningSize();
    bool is_parallel = (aConfigurations.GetVecchiaType() == PARALLEL_VECCHIA_GP);

    // Select source locations and observations based on Vecchia type
    Locations<T>* source_locations = is_parallel ? aData.GetLocations() : aData.GetNewLocations();
    T* source_observations = is_parallel ? aData.GetHostObservations() : aData.GetHostObservationsNew();
    
    if (is_parallel) {
        // Parallel Vecchia: Use sequential selection (last k points from previous locations)
        // For batch i_block (i_block >= 1), we use locations from [0..k + i_block - 1]
        // We take the last k points from this range
        int query_index = k + i_block - 1;
        int start_idx = (query_index >= k) ? (query_index - k + 1) : 0;
        int end_idx = query_index + 1;
        int actual_k = (k > (end_idx - start_idx)) ? (end_idx - start_idx) : k;
        
        // Copy the last k points (or available points) from the range [start_idx..end_idx)
        int copy_start = end_idx - actual_k;
        int offset = i_block * k;
        
        memcpy(aData.GetConditioningLocations()->GetLocationX() + offset, 
               source_locations->GetLocationX() + copy_start, 
               sizeof(T) * actual_k);
        memcpy(aData.GetConditioningLocations()->GetLocationY() + offset, 
               source_locations->GetLocationY() + copy_start, 
               sizeof(T) * actual_k);
        if (aConfigurations.GetDimension() == Dimension3D) {
            memcpy(aData.GetConditioningLocations()->GetLocationZ() + offset, 
                   source_locations->GetLocationZ() + copy_start, 
                   sizeof(T) * actual_k);
        }
        memcpy(aData.GetHostConditioningObs() + offset, 
               source_observations + copy_start, 
               sizeof(T) * actual_k);
    } else {
        // Block Vecchia: Use sequential selection (last cs points from accumulated previous blocks)
        // Exactly matching ParallelBlockVecchiaGP behavior:
        // memcpy(locations_con->x + i * cs, locations_new->x + batchNumAccum[i] - cs, sizeof(T) * cs);
        int* batchNumAccum = aData.GetBatchNumAccum();
        int offset = i_block * k;
        int copy_start = batchNumAccum[i_block] - k;
        
        memcpy(aData.GetConditioningLocations()->GetLocationX() + offset, 
               source_locations->GetLocationX() + copy_start, 
               sizeof(T) * k);
        memcpy(aData.GetConditioningLocations()->GetLocationY() + offset, 
               source_locations->GetLocationY() + copy_start, 
               sizeof(T) * k);
        if (aConfigurations.GetDimension() == Dimension3D) {
            memcpy(aData.GetConditioningLocations()->GetLocationZ() + offset, 
                   source_locations->GetLocationZ() + copy_start, 
                   sizeof(T) * k);
        }
        memcpy(aData.GetHostConditioningObs() + offset, 
               source_observations + copy_start, 
               sizeof(T) * k);
    }
}
