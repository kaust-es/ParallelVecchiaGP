
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file LocationGenerator.cpp
 * @brief Generates and manages spatial locations for VecchiaGB.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <cmath>
#include <algorithm>
#include <cfloat>

#include <common/Definitions.hpp>
#include <data-units/Locations.hpp>
#include <data-generators/LocationGenerator.hpp>
#include <helpers/ByteHandler.hpp>

using namespace vecchia::generators;
using namespace vecchia::common;
using namespace vecchia::dataunits;
using namespace vecchia::helpers;

// TODO (Option A): Full N-dimensional refactor
// This function currently only supports 2D/3D/ST (max 3 dimensions).
// For full N-dimensional support (e.g., 10D for Scaled Block Vecchia):
// 1. Change Locations<T> to store coordinates as double** (N arrays)
// 2. Replace GetLocationX/Y/Z() with GetCoordinate(dim_idx)
// 3. Update all downstream code to use dimension loops instead of X/Y/Z
// 4. This affects ~30 files including clustering, KNN, distance calculations
// For now, Scaled Block Vecchia uses its own N-dimensional data structures.

template<typename T>
void LocationGenerator<T>::GenerateLocations(const int &aN, const int &aTimeSlot, const Dimension &aDimension,
                                             Locations<T> &aLocations) {

    aLocations.SetSize(aN);
    int index = 0;
    aLocations.SetDimension(aDimension);

    int rootN;
    if (aDimension == Dimension3D) {
        //Cubic root.
        rootN = ceil(cbrt(aN));
    } else {
        //Square root.
        rootN = ceil(sqrt(aN));
    }

    int *grid = new int[rootN]();
    for (auto i = 0; i < rootN; i++) {
        grid[i] = i + 1;
    }
    T range_low = -0.4, range_high = 0.4;

    for (auto i = 0; i < rootN && index < aN; i++) {
        for (auto j = 0; j < rootN && index < aN; j++) {
            if (aDimension == Dimension3D) {
                for (auto k = 0; k < rootN && index < aN; k++) {
                    aLocations.GetLocationX()[index] =
                            (grid[i] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                    aLocations.GetLocationY()[index] =
                            (grid[j] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                    aLocations.GetLocationZ()[index] =
                            (grid[k] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                    index++;
                }
            } else {
                aLocations.GetLocationX()[index] =
                        (grid[i] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                aLocations.GetLocationY()[index] =
                        (grid[j] - 0.5 + UniformDistribution(range_low, range_high)) / rootN;
                if (aDimension == DimensionST) {
                    aLocations.GetLocationZ()[index] = 1.0;
                }
                index++;
            }
        }
    }
    delete[] grid;
    if (aDimension != DimensionST) {
        SortLocations(aN, aDimension, aLocations);
    } else {
        for (auto i = 0; i < aN; i++) {
            aLocations.GetLocationX()[i] = aLocations.GetLocationX()[i];
            aLocations.GetLocationY()[i] = aLocations.GetLocationY()[i];
            aLocations.GetLocationZ()[i] = (T) (i / aTimeSlot + 1);
        }
    }
}

template<typename T>
T LocationGenerator<T>::UniformDistribution(const T &aRangeLow, const T &aRangeHigh) {
    T myRand = (T) rand() / (T) (1.0 + RAND_MAX);
    T range = aRangeHigh - aRangeLow;
    return (myRand * range) + aRangeLow;
}

template<typename T>
void
LocationGenerator<T>::SortLocations(const int &aN, const Dimension &aDimension, Locations<T> &aLocations) {

    // Some sorting as Morton order.
    uint16_t x, y, z;
    uint64_t vectorZ[aN];

    // Encode data into vector z
    for (auto i = 0; i < aN; i++) {
        x = (uint16_t) (aLocations.GetLocationX()[i] * (double) UINT16_MAX + .5);
        y = (uint16_t) (aLocations.GetLocationY()[i] * (double) UINT16_MAX + .5);
        if (aDimension != Dimension2D) {
            z = (uint16_t) (aLocations.GetLocationZ()[i] * (double) UINT16_MAX + .5);
        } else {
            z = (uint16_t) 0.0;
        }
        vectorZ[i] = (SpreadBits(z) << 2) + (SpreadBits(y) << 1) + SpreadBits(x);
    }
    // Sort vector z
    std::sort(vectorZ, vectorZ + aN, CompareUint64);

    // Decode data from vector z
    for (auto i = 0; i < aN; i++) {
        x = ReverseSpreadBits(vectorZ[i] >> 0);
        y = ReverseSpreadBits(vectorZ[i] >> 1);
        z = ReverseSpreadBits(vectorZ[i] >> 2);
        aLocations.GetLocationX()[i] = (double) x / (double) UINT16_MAX;
        aLocations.GetLocationY()[i] = (double) y / (double) UINT16_MAX;
        if (aDimension == Dimension3D) {
            aLocations.GetLocationZ()[i] = (double) z / (double) UINT16_MAX;
        }
    }
}

template<typename T>
void LocationGenerator<T>::RandomReordering(const int &aN, const Dimension &aDimension, Locations<T> &aLocations) {
    int seed = 42; // Set your desired seed value
    srand(seed);

    for (int i = aN - 1; i > 0; i--){
        int j = rand() % (i + 1);
        // Swap x values
        double tempX = aLocations.GetLocationX()[i];
        aLocations.GetLocationX()[i] = aLocations.GetLocationX()[j];
        aLocations.GetLocationX()[j] = tempX;

        // Swap y values
        double tempY = aLocations.GetLocationY()[i];
        aLocations.GetLocationY()[i] = aLocations.GetLocationY()[j];
        aLocations.GetLocationY()[j] = tempY;

        // Swap z values
        if (aDimension == Dimension3D) {
            double tempZ = aLocations.GetLocationZ()[i];
            aLocations.GetLocationZ()[i] = aLocations.GetLocationZ()[j];
            aLocations.GetLocationZ()[j] = tempZ;
        }
    }
}


  /*
  clusters reordering index
  */
  bool hasDuplicates(int *array, int length)
  {
    for (int i = 0; i < length - 1; i++)
    {
      for (int j = i + 1; j < length; j++)
      {
        if (array[i] == array[j])
        {
          throw std::runtime_error("Duplicate index found: index[" + std::to_string(i) + "] and index[" + 
                                   std::to_string(j) + "] share " + std::to_string(array[j]));
          return false;
        }
      }
    }
    return false;
  }

template<typename T>
void LocationGenerator<T>::ReorderIndex(Locations<T> &aLocations, const Dimension &aDimension, std::vector<Point<T>> &aPoints, int *aClusterReordering, int aNClusters, bool aTimeFlag)
  {
// in/out: clusterReordering, 0, 1, 2, 3, ...., n -> 10, 2, 3, ...,
#pragma omp parallel for
    for (int i = 0; i < aNClusters; ++i)
    {
      double _dist_min = DBL_MAX;
      int min_index = 0; // Track the index of the minimum distance
      for (int j = 0; j < aNClusters; ++j)
      {
        double dx = aLocations.GetLocationX()[i] - aPoints[j].GetCoordinates()[0];
        double dy = aLocations.GetLocationY()[i] - aPoints[j].GetCoordinates()[1];
        double _dist_temp = dx * dx + dy * dy; // Use squared distance
        if (aDimension == Dimension3D){
          double dz = aLocations.GetLocationZ()[i] - aPoints[j].GetCoordinates()[2];
          _dist_temp += dz*dz;
        }
        if (_dist_temp < _dist_min)
        {
          _dist_min = _dist_temp;
          min_index = j;
        }
      }
      aClusterReordering[i] = min_index;
    }

    // check if there is same orders
    if (hasDuplicates(aClusterReordering, aNClusters))
    {
      throw std::runtime_error("Reordering validation failed: Unknown issues detected in reordering");
    }
  }


template<typename T>
void LocationGenerator<T>::ClusterToBatch(int num_loc, int batchCount, int *batchNum, int *batchNumAccum, 
    int *batchIndex, Locations<T> &locations, T *h_obs, Locations<T> &locations_new, T *h_obs_new, 
    std::vector<Point<T>> &points, Locations<T> &locsCentroid, bool time_flag)
  {
    // reconstruct the new locations and observations
    for (int i = 0; i < batchCount; ++i)
    {
      // batchIndex[i] is the current cluster, it has batchNum[i] locations
      int _id = 0;
      for (int j = 0; j < batchNum[i]; ++j)
      {
        while (true)
        {
          if ((points[_id].GetCluster() == batchIndex[i]) || _id >= num_loc)
            break;
          else
            _id++;
        }
        if (_id < num_loc)
        {
          locations_new.GetLocationX()[batchNumAccum[i] + j] = locations.GetLocationX()[_id];
          locations_new.GetLocationY()[batchNumAccum[i] + j] = locations.GetLocationY()[_id];
          if (time_flag)
          {
            locations_new.GetLocationZ()[batchNumAccum[i] + j] = locations.GetLocationZ()[_id];
          }
          h_obs_new[batchNumAccum[i] + j] = h_obs[_id];
          _id++;
        }
      }
    }
  }