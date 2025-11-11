
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file VecchiaMemoryBuffers.cpp
 * @brief Implementation of VecchiaMemoryBuffers class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-16
**/

#include <data-units/VecchiaMemoryBuffers.hpp>

using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateCovarianceMemory(size_t aHostSize, size_t aDeviceSize) {
    mCovarianceMemory = HostDeviceMemory<T>(aHostSize, aDeviceSize);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateObservationsMemory(size_t aSize) {
    mObservationsMemory = MagmaDeviceMemory<T>(aSize);
    mObservationsCopyMemory = MagmaDeviceMemory<T>(aSize);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateConditioningCovarianceMemory(size_t aHostSize, size_t aDeviceSize) {
    mConditioningCovMemory = HostDeviceMemory<T>(aHostSize, aDeviceSize);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateConditioningObservationsMemory(size_t aHostSize, size_t aDeviceSize) {
    mConditioningObsMemory = HostDeviceMemory<T>(aHostSize, aDeviceSize);
    mConditioningObsCopyMemory = MagmaDeviceMemory<T>(aDeviceSize);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateCrossCovarianceMemory(size_t aHostSize, size_t aDeviceSize) {
    mCrossCovMemory = HostDeviceMemory<T>(aHostSize, aDeviceSize);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateOffsetMemory(size_t aCovOffsetSize, size_t aMuOffsetSize) {
    mCovOffsetMemory = MagmaDeviceMemory<T>(aCovOffsetSize);
    mMuOffsetMemory = MagmaDeviceMemory<T>(aMuOffsetSize);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateResultMemory(size_t aSize) {
    mLogDetResults = MagmaHostMemory<T>(aSize);
    mNorm2Results = MagmaHostMemory<T>(aSize);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateCovarianceArrayPointers(size_t aBatchCount) {
    mHostCovarianceArray = MagmaHostMemory<T*>(aBatchCount);
    mDeviceCovarianceArray = MagmaDeviceMemory<T*>(aBatchCount);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateObservationsArrayPointers(size_t aBatchCount) {
    mHostObservationsArray = MagmaHostMemory<T*>(aBatchCount);
    mDeviceObservationsArray = MagmaDeviceMemory<T*>(aBatchCount);
    mHostObservationsArrayCopy = MagmaHostMemory<T*>(aBatchCount);
    mDeviceObservationsArrayCopy = MagmaDeviceMemory<T*>(aBatchCount);
}

template<typename T>
void VecchiaMemoryBuffers<T>::AllocateConditioningArrayPointers(size_t aBatchCount) {
    mHostCovarianceConditioningArray = MagmaHostMemory<T*>(aBatchCount);
    mDeviceCovarianceConditioningArray = MagmaDeviceMemory<T*>(aBatchCount);
    mHostCovarianceCrossArray = MagmaHostMemory<T*>(aBatchCount);
    mDeviceCovarianceCrossArray = MagmaDeviceMemory<T*>(aBatchCount);
    mHostCovarianceOffsetArray = MagmaHostMemory<T*>(aBatchCount);
    mDeviceCovarianceOffsetArray = MagmaDeviceMemory<T*>(aBatchCount);
    mHostMuOffsetArray = MagmaHostMemory<T*>(aBatchCount);
    mDeviceMuOffsetArray = MagmaDeviceMemory<T*>(aBatchCount);
    mHostObservationsConditioningArray = MagmaHostMemory<T*>(aBatchCount);
    mDeviceObservationsConditioningArray = MagmaDeviceMemory<T*>(aBatchCount);
    mHostObservationsConditioningArrayCopy = MagmaHostMemory<T*>(aBatchCount);
    mDeviceObservationsConditioningArrayCopy = MagmaDeviceMemory<T*>(aBatchCount);
}
