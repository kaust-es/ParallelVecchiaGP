
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file VecchiaMemoryBuffers.hpp
 * @brief Memory buffers for covariance matrices, observations, and conditioning data.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-16
**/

#ifndef VECCHIAGP_VECCHIAMEMORYBUFFERS_HPP
#define VECCHIAGP_VECCHIAMEMORYBUFFERS_HPP

#include <magma_v2.h>
#include <memory>
#include <common/Definitions.hpp>
#include <helpers/GPUMemoryManager.hpp>

namespace vecchia::dataunits {

    /**
     * @class VecchiaMemoryBuffers
     * @brief Manages memory buffers for Vecchia approximation computations.
     * 
     * This class encapsulates all memory buffers needed for covariance matrices,
     * observations, and conditioning data. It uses RAII principles for automatic
     * memory management and provides clear ownership semantics.
     */
    template<typename T>
    class VecchiaMemoryBuffers {
    public:
        /**
         * @brief Default constructor.
         */
        VecchiaMemoryBuffers() = default;

        /**
         * @brief Default destructor (RAII handles cleanup).
         */
        ~VecchiaMemoryBuffers() = default;

        // Delete copy operations
        VecchiaMemoryBuffers(const VecchiaMemoryBuffers&) = delete;
        VecchiaMemoryBuffers& operator=(const VecchiaMemoryBuffers&) = delete;

        // Allow move operations
        VecchiaMemoryBuffers(VecchiaMemoryBuffers&&) noexcept = default;
        VecchiaMemoryBuffers& operator=(VecchiaMemoryBuffers&&) noexcept = default;

        // ========== Covariance Matrices ==========
        
        /**
         * @brief Allocates host and device memory for covariance matrices.
         * @param[in] aHostSize Size of host covariance buffer.
         * @param[in] aDeviceSize Size of device covariance buffer.
         */
        void AllocateCovarianceMemory(size_t aHostSize, size_t aDeviceSize);
        
        T* GetHostCovariance() { return mCovarianceMemory.GetHost(); }
        const T* GetHostCovariance() const { return mCovarianceMemory.GetHost(); }
        
        T* GetDeviceCovariance() { return mCovarianceMemory.GetDevice(); }
        const T* GetDeviceCovariance() const { return mCovarianceMemory.GetDevice(); }

        // ========== Observations ==========
        
        /**
         * @brief Allocates device memory for observations and its copy.
         * @param[in] aSize Size of observations buffer.
         */
        void AllocateObservationsMemory(size_t aSize);
        
        T* GetDeviceObservations() { return mObservationsMemory.Get(); }
        const T* GetDeviceObservations() const { return mObservationsMemory.Get(); }
        
        T* GetDeviceObservationsCopy() { return mObservationsCopyMemory.Get(); }
        const T* GetDeviceObservationsCopy() const { return mObservationsCopyMemory.Get(); }

        // ========== Conditioning Data ==========
        
        /**
         * @brief Allocates memory for conditioning covariance matrices.
         * @param[in] aHostSize Size of host conditioning covariance.
         * @param[in] aDeviceSize Size of device conditioning covariance.
         */
        void AllocateConditioningCovarianceMemory(size_t aHostSize, size_t aDeviceSize);
        
        T* GetHostConditioningCov() { return mConditioningCovMemory.GetHost(); }
        const T* GetHostConditioningCov() const { return mConditioningCovMemory.GetHost(); }
        
        T* GetDeviceConditioningCov() { return mConditioningCovMemory.GetDevice(); }
        const T* GetDeviceConditioningCov() const { return mConditioningCovMemory.GetDevice(); }

        /**
         * @brief Allocates memory for conditioning observations.
         * @param[in] aHostSize Size of host conditioning observations.
         * @param[in] aDeviceSize Size of device conditioning observations.
         */
        void AllocateConditioningObservationsMemory(size_t aHostSize, size_t aDeviceSize);
        
        T* GetHostConditioningObs() { return mConditioningObsMemory.GetHost(); }
        const T* GetHostConditioningObs() const { return mConditioningObsMemory.GetHost(); }
        
        T* GetDeviceConditioningObs() { return mConditioningObsMemory.GetDevice(); }
        const T* GetDeviceConditioningObs() const { return mConditioningObsMemory.GetDevice(); }
        
        T* GetDeviceObservationsConditioningCopy() { return mConditioningObsCopyMemory.Get(); }
        const T* GetDeviceObservationsConditioningCopy() const { return mConditioningObsCopyMemory.Get(); }

        /**
         * @brief Allocates memory for cross-covariance matrices.
         * @param[in] aHostSize Size of host cross-covariance.
         * @param[in] aDeviceSize Size of device cross-covariance.
         */
        void AllocateCrossCovarianceMemory(size_t aHostSize, size_t aDeviceSize);
        
        T* GetHostCrossCov() { return mCrossCovMemory.GetHost(); }
        const T* GetHostCrossCov() const { return mCrossCovMemory.GetHost(); }
        
        T* GetDeviceCrossCov() { return mCrossCovMemory.GetDevice(); }
        const T* GetDeviceCrossCov() const { return mCrossCovMemory.GetDevice(); }

        /**
         * @brief Allocates memory for covariance and mean offsets.
         * @param[in] aCovOffsetSize Size of covariance offset.
         * @param[in] aMuOffsetSize Size of mean offset.
         */
        void AllocateOffsetMemory(size_t aCovOffsetSize, size_t aMuOffsetSize);
        
        T* GetDeviceCovOffset() { return mCovOffsetMemory.Get(); }
        const T* GetDeviceCovOffset() const { return mCovOffsetMemory.Get(); }
        
        T* GetDeviceMuOffset() { return mMuOffsetMemory.Get(); }
        const T* GetDeviceMuOffset() const { return mMuOffsetMemory.Get(); }

        // ========== Result Buffers ==========
        
        /**
         * @brief Allocates memory for log-determinant and norm results.
         * @param[in] aSize Size of result arrays (typically batch count).
         */
        void AllocateResultMemory(size_t aSize);
        
        T* GetLogDetResults() { return mLogDetResults.Get(); }
        const T* GetLogDetResults() const { return mLogDetResults.Get(); }
        
        T* GetNorm2Results() { return mNorm2Results.Get(); }
        const T* GetNorm2Results() const { return mNorm2Results.Get(); }

        // ========== Batch Array Pointers ==========
        
        /**
         * @brief Allocates batch array pointers for covariance matrices.
         * @param[in] aBatchCount Number of batches.
         */
        void AllocateCovarianceArrayPointers(size_t aBatchCount);
        
        T** GetHostCovarianceArray() { return mHostCovarianceArray.Get(); }
        const T** GetHostCovarianceArray() const { return const_cast<const T**>(mHostCovarianceArray.Get()); }
        
        T** GetDeviceCovarianceArray() { return mDeviceCovarianceArray.Get(); }
        const T** GetDeviceCovarianceArray() const { return const_cast<const T**>(mDeviceCovarianceArray.Get()); }

        /**
         * @brief Allocates batch array pointers for observations.
         * @param[in] aBatchCount Number of batches.
         */
        void AllocateObservationsArrayPointers(size_t aBatchCount);
        
        T** GetHostObservationsArray() { return mHostObservationsArray.Get(); }
        const T** GetHostObservationsArray() const { return const_cast<const T**>(mHostObservationsArray.Get()); }
        
        T** GetDeviceObservationsArray() { return mDeviceObservationsArray.Get(); }
        const T** GetDeviceObservationsArray() const { return const_cast<const T**>(mDeviceObservationsArray.Get()); }
        
        T** GetHostObservationsArrayCopy() { return mHostObservationsArrayCopy.Get(); }
        const T** GetHostObservationsArrayCopy() const { return const_cast<const T**>(mHostObservationsArrayCopy.Get()); }
        
        T** GetDeviceObservationsArrayCopy() { return mDeviceObservationsArrayCopy.Get(); }
        const T** GetDeviceObservationsArrayCopy() const { return const_cast<const T**>(mDeviceObservationsArrayCopy.Get()); }

        /**
         * @brief Allocates conditioning batch array pointers.
         * @param[in] aBatchCount Number of batches.
         */
        void AllocateConditioningArrayPointers(size_t aBatchCount);
        
        T** GetHostCovarianceConditioningArray() { return mHostCovarianceConditioningArray.Get(); }
        T** GetDeviceCovarianceConditioningArray() { return mDeviceCovarianceConditioningArray.Get(); }
        T** GetHostCovarianceCrossArray() { return mHostCovarianceCrossArray.Get(); }
        T** GetDeviceCovarianceCrossArray() { return mDeviceCovarianceCrossArray.Get(); }
        T** GetHostCovarianceOffsetArray() { return mHostCovarianceOffsetArray.Get(); }
        T** GetDeviceCovarianceOffsetArray() { return mDeviceCovarianceOffsetArray.Get(); }
        T** GetHostMuOffsetArray() { return mHostMuOffsetArray.Get(); }
        T** GetDeviceMuOffsetArray() { return mDeviceMuOffsetArray.Get(); }
        T** GetHostObservationsConditioningArray() { return mHostObservationsConditioningArray.Get(); }
        T** GetDeviceObservationsConditioningArray() { return mDeviceObservationsConditioningArray.Get(); }
        T** GetHostObservationsConditioningArrayCopy() { return mHostObservationsConditioningArrayCopy.Get(); }
        T** GetDeviceObservationsConditioningArrayCopy() { return mDeviceObservationsConditioningArrayCopy.Get(); }

    private:
        // Covariance matrices
        helpers::HostDeviceMemory<T> mCovarianceMemory;

        // Observations
        helpers::MagmaDeviceMemory<T> mObservationsMemory;
        helpers::MagmaDeviceMemory<T> mObservationsCopyMemory;

        // Conditioning covariance
        helpers::HostDeviceMemory<T> mConditioningCovMemory;

        // Conditioning observations
        helpers::HostDeviceMemory<T> mConditioningObsMemory;
        helpers::MagmaDeviceMemory<T> mConditioningObsCopyMemory;

        // Cross-covariance
        helpers::HostDeviceMemory<T> mCrossCovMemory;

        // Offsets
        helpers::MagmaDeviceMemory<T> mCovOffsetMemory;
        helpers::MagmaDeviceMemory<T> mMuOffsetMemory;

        // Result buffers
        helpers::MagmaHostMemory<T> mLogDetResults;
        helpers::MagmaHostMemory<T> mNorm2Results;

        // Batch array pointers (array of pointers for MAGMA batch operations)
        helpers::MagmaHostMemory<T*> mHostCovarianceArray;
        helpers::MagmaDeviceMemory<T*> mDeviceCovarianceArray;
        helpers::MagmaHostMemory<T*> mHostObservationsArray;
        helpers::MagmaDeviceMemory<T*> mDeviceObservationsArray;
        helpers::MagmaHostMemory<T*> mHostObservationsArrayCopy;
        helpers::MagmaDeviceMemory<T*> mDeviceObservationsArrayCopy;

        // Conditioning batch array pointers
        helpers::MagmaHostMemory<T*> mHostCovarianceConditioningArray;
        helpers::MagmaDeviceMemory<T*> mDeviceCovarianceConditioningArray;
        helpers::MagmaHostMemory<T*> mHostCovarianceCrossArray;
        helpers::MagmaDeviceMemory<T*> mDeviceCovarianceCrossArray;
        helpers::MagmaHostMemory<T*> mHostCovarianceOffsetArray;
        helpers::MagmaDeviceMemory<T*> mDeviceCovarianceOffsetArray;
        helpers::MagmaHostMemory<T*> mHostMuOffsetArray;
        helpers::MagmaDeviceMemory<T*> mDeviceMuOffsetArray;
        helpers::MagmaHostMemory<T*> mHostObservationsConditioningArray;
        helpers::MagmaDeviceMemory<T*> mDeviceObservationsConditioningArray;
        helpers::MagmaHostMemory<T*> mHostObservationsConditioningArrayCopy;
        helpers::MagmaDeviceMemory<T*> mDeviceObservationsConditioningArrayCopy;
    };

    /**
     * @brief Instantiates the VecchiaMemoryBuffers class for float and double types.
     * @tparam T Data Type: float or double
     */
    VECCHIAGP_INSTANTIATE_CLASS(VecchiaMemoryBuffers)

} // namespace vecchia::dataunits

#endif //VECCHIAGP_VECCHIAMEMORYBUFFERS_HPP

