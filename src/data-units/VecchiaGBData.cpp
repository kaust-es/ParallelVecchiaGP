
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file VecchiaGBData.cpp
 * @brief Contains the implementation of the VecchiaGBData class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <data-units/VecchiaGBData.hpp>
#include <data-clustering/ClusteringStrategy.hpp>
#include <utilities/EnumStringParser.hpp>

using namespace vecchia::dataunits;
using namespace vecchia::common;
using namespace vecchia::clustering;

template<typename T>
VecchiaGBData<T>::VecchiaGBData(const int &aSize, const Dimension &aDimension, const int &aBlockSize) {
    this->mpLocations = new Locations<T>(aSize, aDimension);
    this->mNumberOfClusters = (aSize < 2 * aBlockSize) ? aSize : aBlockSize;
    this->mpCentroidsLocations = new Locations<T>(this->mNumberOfClusters, aDimension);
    this->mpPremIndex = new int[this->mNumberOfClusters];
    this->mpHostObservations = new T[aSize];
    this->mpFirstClusterCount = new int[1];
    this->mpNewLocations = new Locations<T>(aSize, aDimension);
}

template<typename T>
VecchiaGBData<T>::VecchiaGBData(const int &aSize, const std::string &aDimension) {

    this->mpLocations = new Locations<T>(aSize, GetInputDimension(aDimension));
}

template<typename T>
VecchiaGBData<T>::~VecchiaGBData() {
    // Clean up Location objects allocated in constructor
    delete this->mpLocations;
    delete this->mpCentroidsLocations;
    delete this->mpNewLocations;
    delete this->mpConditioningLocations;
    
    // Clean up basic arrays allocated in constructor
    delete[] this->mpPremIndex;
    delete[] this->mpHostObservations;
    delete[] this->mpFirstClusterCount;
    delete[] this->mpBatchNum;
    delete[] this->mpBatchNumAccum;
    delete[] this->mpBatchNumSquareAccum;
    delete[] this->mpClusterNum;
    
    // Clean up MAGMA-allocated host memory
    if (this->mpHostCovariance) {
        magma_free_cpu(this->mpHostCovariance);
    }
    if (this->mpHostCovarianceConditioning) {
        magma_free_cpu(this->mpHostCovarianceConditioning);
    }
    // For Parallel Vecchia, these are managed explicitly, not freed in destructor
    if (mNumGPUs == 0) {
        if (this->mpHostObservationsConditioning) {
            free(this->mpHostObservationsConditioning);
        }
        if (this->mpHostCovarianceCross) {
            free(this->mpHostCovarianceCross);
        }
    }
    if (this->mpHostLogDetResults) {
        magma_free_cpu(this->mpHostLogDetResults);
    }
    if (this->mpHostNorm2Results) {
        magma_free_cpu(this->mpHostNorm2Results);
    }
    if (this->mpHostInfo) {
        magma_free_cpu(this->mpHostInfo);
    }
    if (this->mpHostConst1) {
        magma_free_cpu(this->mpHostConst1);
    }
    if (this->mpHostLDAConditioning) {
        magma_free_cpu(this->mpHostLDAConditioning);
    }
    if (this->mpHostLDDAConditioning) {
        magma_free_cpu(this->mpHostLDDAConditioning);
    }
    
    // Clean up MAGMA-allocated device memory
    if (this->mpDeviceCovariance) {
        magma_free(this->mpDeviceCovariance);
    }
    if (this->mpDeviceObservations) {
        magma_free(this->mpDeviceObservations);
    }
    if (this->mpDeviceObservationsCopy) {
        magma_free(this->mpDeviceObservationsCopy);
    }
    if (this->mpDeviceCovarianceConditioning) {
        magma_free(this->mpDeviceCovarianceConditioning);
    }
    // Skip freeing single-pointer device arrays if part of multi-GPU arrays
    if (mNumGPUs == 0) {
        if (this->mpDeviceObservationsConditioning) {
            magma_free(this->mpDeviceObservationsConditioning);
        }
        if (this->mpDeviceObservationsConditioningCopy) {
            magma_free(this->mpDeviceObservationsConditioningCopy);
        }
        if (this->mpDeviceCovarianceCross) {
            magma_free(this->mpDeviceCovarianceCross);
        }
        if (this->mpDeviceCovarianceOffset) {
            magma_free(this->mpDeviceCovarianceOffset);
        }
        if (this->mpDeviceMuOffset) {
            magma_free(this->mpDeviceMuOffset);
        }
    }
    if (this->mpDeviceBatchNum) {
        magma_free(this->mpDeviceBatchNum);
    }
    if (this->mpDeviceInfo) {
        magma_free(this->mpDeviceInfo);
    }
    if (this->mpDeviceConst1) {
        magma_free(this->mpDeviceConst1);
    }
    if (this->mpDeviceLDA) {
        magma_free(this->mpDeviceLDA);
    }
    if (this->mpDeviceLDDA) {
        magma_free(this->mpDeviceLDDA);
    }
    if (this->mpDeviceLDAConditioning) {
        magma_free(this->mpDeviceLDAConditioning);
    }
    if (this->mpDeviceLDDAConditioning) {
        magma_free(this->mpDeviceLDDAConditioning);
    }
    
    // Multi-GPU Parallel Vecchia arrays not freed in destructor
    // These are managed explicitly by the application lifecycle
    
    // Clean up batch array pointers for Block Vecchia
    if (this->mpHostCovarianceArray) {
        magma_free_cpu(this->mpHostCovarianceArray);
    }
    if (this->mpDeviceCovarianceArray) {
        magma_free(this->mpDeviceCovarianceArray);
    }
    if (this->mpHostObservationsArray) {
        magma_free_cpu(this->mpHostObservationsArray);
    }
    if (this->mpDeviceObservationsArray) {
        magma_free(this->mpDeviceObservationsArray);
    }
    if (this->mpHostObservationsArrayCopy) {
        magma_free_cpu(this->mpHostObservationsArrayCopy);
    }
    if (this->mpDeviceObservationsArrayCopy) {
        magma_free(this->mpDeviceObservationsArrayCopy);
    }
    // Clean up vecchia conditioning arrays
    if (mNumGPUs > 0 && this->mpDeviceCovarianceConditioningArray) {
        for (int g = 0; g < mNumGPUs; g++) {
            cudaSetDevice(g);
            if (this->mpDeviceCovarianceConditioningArray[g]) {
                cudaFree(this->mpDeviceCovarianceConditioningArray[g]);
            }
        }
        delete[] this->mpDeviceCovarianceConditioningArray;
    } else if (this->mpDeviceCovarianceConditioningArray) {
        magma_free(this->mpDeviceCovarianceConditioningArray);
    }
    
    if (mNumGPUs > 0 && this->mpDeviceCovarianceCrossArray) {
        for (int g = 0; g < mNumGPUs; g++) {
            cudaSetDevice(g);
            if (this->mpDeviceCovarianceCrossArray[g]) {
                cudaFree(this->mpDeviceCovarianceCrossArray[g]);
            }
        }
        delete[] this->mpDeviceCovarianceCrossArray;
    } else if (this->mpDeviceCovarianceCrossArray) {
        magma_free(this->mpDeviceCovarianceCrossArray);
    }
    
    if (mNumGPUs > 0 && this->mpDeviceCovarianceOffsetArray) {
        for (int g = 0; g < mNumGPUs; g++) {
            cudaSetDevice(g);
            if (this->mpDeviceCovarianceOffsetArray[g]) {
                cudaFree(this->mpDeviceCovarianceOffsetArray[g]);
            }
        }
        delete[] this->mpDeviceCovarianceOffsetArray;
    } else if (this->mpDeviceCovarianceOffsetArray) {
        magma_free(this->mpDeviceCovarianceOffsetArray);
    }
    
    if (mNumGPUs > 0 && this->mpDeviceMuOffsetArray) {
        for (int g = 0; g < mNumGPUs; g++) {
            cudaSetDevice(g);
            if (this->mpDeviceMuOffsetArray[g]) {
                cudaFree(this->mpDeviceMuOffsetArray[g]);
            }
        }
        delete[] this->mpDeviceMuOffsetArray;
    } else if (this->mpDeviceMuOffsetArray) {
        magma_free(this->mpDeviceMuOffsetArray);
    }
    
    if (mNumGPUs > 0 && this->mpDeviceObservationsConditioningArray) {
        for (int g = 0; g < mNumGPUs; g++) {
            cudaSetDevice(g);
            if (this->mpDeviceObservationsConditioningArray[g]) {
                cudaFree(this->mpDeviceObservationsConditioningArray[g]);
            }
        }
        delete[] this->mpDeviceObservationsConditioningArray;
    } else if (this->mpDeviceObservationsConditioningArray) {
        magma_free(this->mpDeviceObservationsConditioningArray);
    }
    
    // Host arrays
    if (this->mpHostCovarianceConditioningArray) {
        magma_free_cpu(this->mpHostCovarianceConditioningArray);
    }
    if (this->mpHostCovarianceCrossArray) {
        magma_free_cpu(this->mpHostCovarianceCrossArray);
    }
    if (this->mpHostCovarianceOffsetArray) {
        magma_free_cpu(this->mpHostCovarianceOffsetArray);
    }
    if (this->mpHostMuOffsetArray) {
        magma_free_cpu(this->mpHostMuOffsetArray);
    }
    if (this->mpHostObservationsConditioningArray) {
        magma_free_cpu(this->mpHostObservationsConditioningArray);
    }
    if (this->mpHostObservationsConditioningArrayCopy) {
        magma_free_cpu(this->mpHostObservationsConditioningArrayCopy);
    }
    if (this->mpDeviceObservationsConditioningArrayCopy) {
        magma_free(this->mpDeviceObservationsConditioningArrayCopy);
    }
}

template<typename T>
Locations<T> *VecchiaGBData<T>::GetLocations() {
    return this->mpLocations;
}

template<typename T>
void VecchiaGBData<T>::SetLocations(Locations<T> &aLocation) {

    if (this->mpLocations) {
        delete this->mpLocations;
    }
    this->mpLocations = &aLocation;
    this->mpLocations->SetLocationX(*aLocation.GetLocationX(), aLocation.GetSize());
    this->mpLocations->SetLocationY(*aLocation.GetLocationY(), aLocation.GetSize());
    if (aLocation.GetLocationZ()) {
        this->mpLocations->SetLocationZ(*aLocation.GetLocationZ(), aLocation.GetSize());
    }
}

template<typename T>
Locations<T> *VecchiaGBData<T>::GetCentroidsLocations() {
    return this->mpCentroidsLocations;
}

template<typename T>
void VecchiaGBData<T>::SetCentroidsLocations(Locations<T> &aCentroidsLocations) {
    if (this->mpCentroidsLocations) {
        delete this->mpCentroidsLocations;
    }
    this->mpCentroidsLocations = &aCentroidsLocations;
    this->mpCentroidsLocations->SetLocationX(*aCentroidsLocations.GetLocationX(), aCentroidsLocations.GetSize());
    this->mpCentroidsLocations->SetLocationY(*aCentroidsLocations.GetLocationY(), aCentroidsLocations.GetSize());
    if (aCentroidsLocations.GetLocationZ()) {
        this->mpCentroidsLocations->SetLocationZ(*aCentroidsLocations.GetLocationZ(), aCentroidsLocations.GetSize());
    }
}

template<typename T>
int *VecchiaGBData<T>::GetPremIndex() {
    return this->mpPremIndex;
}

template<typename T>
void VecchiaGBData<T>::SetMleIterations(const int &aMleIterations) {
    this->mMleIterations = aMleIterations;
}

template<typename T>
int VecchiaGBData<T>::GetMleIterations() {
    return this->mMleIterations;
}

template<typename T>
int VecchiaGBData<T>::GetBatchCount() {
    return this->mBatchCount;
}

template<typename T>
void VecchiaGBData<T>::SetBatchCount(const int &aBatchCount) {
    this->mBatchCount = aBatchCount;
}

template<typename T>
double *VecchiaGBData<T>::GetHostObservations() {
    return this->mpHostObservations;
}

template<typename T>
void VecchiaGBData<T>::SetHostObservations(double *aHostObservations) {
    this->mpHostObservations = aHostObservations;
}


template<typename T>
double *VecchiaGBData<T>::GetHostObservationsNew() {
    return this->mpHostObservationsNew;
}

template<typename T>
void VecchiaGBData<T>::SetHostObservationsNew(double *aHostObservations) {
    this->mpHostObservationsNew = aHostObservations;
}
template<typename T>
int *VecchiaGBData<T>::GetFirstClusterCount() {
    return this->mpFirstClusterCount;
}

template<typename T>
int *VecchiaGBData<T>::GetBatchNumAccum() {
    return this->mpBatchNumAccum;
}

template<typename T>
void VecchiaGBData<T>::SetBatchNumAccum(int *aBatchNumAccum) {
    this->mpBatchNumAccum = aBatchNumAccum;
}

template<typename T>
void VecchiaGBData<T>::SetBatchNumSquareAccum(int *aBatchNumSquareAccum) {
    this->mpBatchNumSquareAccum = aBatchNumSquareAccum;
}

template<typename T>
Locations<T> *VecchiaGBData<T>::GetNewLocations() {
    return this->mpNewLocations;
}

template<typename T>
void VecchiaGBData<T>::SetNewLocations(Locations<T> &aNewLocations) {
    if (this->mpNewLocations) {
        delete this->mpNewLocations;
    }
    this->mpNewLocations = &aNewLocations;
    this->mpNewLocations->SetLocationX(*aNewLocations.GetLocationX(), aNewLocations.GetSize());
    this->mpNewLocations->SetLocationY(*aNewLocations.GetLocationY(), aNewLocations.GetSize());
    if (aNewLocations.GetLocationZ()) {
        this->mpNewLocations->SetLocationZ(*aNewLocations.GetLocationZ(), aNewLocations.GetSize());
    }
}

template<typename T>
Locations<T> *VecchiaGBData<T>::GetConditioningLocations() {
    return this->mpConditioningLocations;
}

template<typename T>
void VecchiaGBData<T>::SetConditioningLocations(Locations<T> &aConditioningLocations) {
    if (this->mpConditioningLocations) {
        delete this->mpConditioningLocations;
    }
    this->mpConditioningLocations = &aConditioningLocations;
    this->mpConditioningLocations->SetLocationX(*aConditioningLocations.GetLocationX(), aConditioningLocations.GetSize());
    this->mpConditioningLocations->SetLocationY(*aConditioningLocations.GetLocationY(), aConditioningLocations.GetSize());
    if (aConditioningLocations.GetLocationZ()) {
        this->mpConditioningLocations->SetLocationZ(*aConditioningLocations.GetLocationZ(), aConditioningLocations.GetSize());
    }
}

template<typename T>
int *VecchiaGBData<T>::GetBatchNum() {
    return this->mpBatchNum;
}

template<typename T>
void VecchiaGBData<T>::SetBatchNum(int *aBatchNum) {
    this->mpBatchNum = aBatchNum;
}

template<typename T>
void VecchiaGBData<T>::SetTestClusteringResult(const vecchia::clustering::ClusteringResult<T>& aClusteringResult) {
    // ClusteringResult contains unique_ptr, so we need to manually copy the members
    auto result = std::make_unique<vecchia::clustering::ClusteringResult<T>>();
    result->assignments = aClusteringResult.assignments;
    result->batchSizes = aClusteringResult.batchSizes;
    result->numClusters = aClusteringResult.numClusters;
    result->isPointWise = aClusteringResult.isPointWise;
    result->points = aClusteringResult.points;
    result->blockInfos = aClusteringResult.blockInfos;
    result->blockInfos_test = aClusteringResult.blockInfos_test;
    
    // Deep copy centroids if they exist
    if (aClusteringResult.centroids) {
        int numCentroids = aClusteringResult.centroids->GetSize();
        vecchia::common::Dimension dim = aClusteringResult.centroids->GetDimension();
        result->centroids = std::make_unique<vecchia::dataunits::Locations<T>>(numCentroids, dim);
        result->centroids->SetLocationX(*(aClusteringResult.centroids->GetLocationX()), numCentroids);
        result->centroids->SetLocationY(*(aClusteringResult.centroids->GetLocationY()), numCentroids);
        if (dim == vecchia::common::Dimension3D || dim == vecchia::common::DimensionST) {
            if (aClusteringResult.centroids->GetLocationZ()) {
                result->centroids->SetLocationZ(*(aClusteringResult.centroids->GetLocationZ()), numCentroids);
            }
        }
    }
    
    mpTestClusteringResult = std::move(result);
}