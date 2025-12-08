
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file BatchMetadata.cpp
 * @brief Implementation of BatchMetadata class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-16
**/

#include <data-units/BatchMetadata.hpp>

using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
BatchMetadata<T>::BatchMetadata(int aBatchCount)
    : mBatchCount(aBatchCount),
      mpBatchNum(nullptr),
      mpBatchNumAccum(nullptr),
      mpBatchNumSquareAccum(nullptr),
      mHostLDA(aBatchCount),
      mHostLDDA(aBatchCount),
      mDeviceLDA(aBatchCount + 1),
      mDeviceLDDA(aBatchCount + 1),
      mHostInfo(aBatchCount),
      mDeviceInfo(aBatchCount + 1),
      mHostConst1(aBatchCount),
      mDeviceConst1(aBatchCount + 1),
      mDeviceBatchNum(aBatchCount + 1),
      mHostLDAConditioning(aBatchCount),
      mHostLDDAConditioning(aBatchCount),
      mDeviceLDAConditioning(aBatchCount + 1),
      mDeviceLDDAConditioning(aBatchCount + 1) {
    
    // Initialize constant arrays
    for (int i = 0; i < aBatchCount; ++i) {
        mHostConst1.Get()[i] = 1;
        mHostInfo.Get()[i] = 0;  // 0 indicates success
    }
}

template<typename T>
void BatchMetadata<T>::SetBatchNum(int* aBatchNum) {
    if (mpBatchNum && mpBatchNum != aBatchNum) {
        delete[] mpBatchNum;
    }
    mpBatchNum = aBatchNum;
}

template<typename T>
void BatchMetadata<T>::SetBatchNumAccum(int* aBatchNumAccum) {
    if (mpBatchNumAccum && mpBatchNumAccum != aBatchNumAccum) {
        delete[] mpBatchNumAccum;
    }
    mpBatchNumAccum = aBatchNumAccum;
}

template<typename T>
void BatchMetadata<T>::SetBatchNumSquareAccum(int* aBatchNumSquareAccum) {
    if (mpBatchNumSquareAccum && mpBatchNumSquareAccum != aBatchNumSquareAccum) {
        delete[] mpBatchNumSquareAccum;
    }
    mpBatchNumSquareAccum = aBatchNumSquareAccum;
}
