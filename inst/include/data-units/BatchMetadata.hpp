
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file BatchMetadata.hpp
 * @brief Contains batch processing metadata for Vecchia approximation.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-16
**/

#ifndef VECCHIAGP_BATCHMETADATA_HPP
#define VECCHIAGP_BATCHMETADATA_HPP

#include <magma_v2.h>
#include <memory>
#include <common/Definitions.hpp>
#include <helpers/GPUMemoryManager.hpp>

namespace vecchia::dataunits {

    /**
     * @class BatchMetadata
     * @brief Encapsulates metadata for batch operations in Vecchia approximation.
     * 
     * This class manages information about batch sizes, leading dimensions,
     * and other metadata needed for MAGMA batch operations. It uses RAII
     * principles to ensure proper memory management.
     */
    template<typename T>
    class BatchMetadata {
    public:
        /**
         * @brief Constructs BatchMetadata with specified batch count.
         * @param[in] aBatchCount Number of batches.
         * @throws std::runtime_error if memory allocation fails.
         */
        explicit BatchMetadata(int aBatchCount);

        /**
         * @brief Default destructor (RAII handles cleanup).
         */
        ~BatchMetadata() = default;

        // Delete copy operations
        BatchMetadata(const BatchMetadata&) = delete;
        BatchMetadata& operator=(const BatchMetadata&) = delete;

        // Allow move operations
        BatchMetadata(BatchMetadata&&) noexcept = default;
        BatchMetadata& operator=(BatchMetadata&&) noexcept = default;

        /**
         * @brief Gets the batch count.
         * @return Number of batches.
         */
        int GetBatchCount() const { return mBatchCount; }

        /**
         * @brief Gets the batch number array (size of each batch).
         * @return Pointer to batch numbers array.
         */
        int* GetBatchNum() { return mpBatchNum; }
        const int* GetBatchNum() const { return mpBatchNum; }

        /**
         * @brief Sets the batch number array.
         * @param[in] aBatchNum Pointer to batch numbers array.
         * @note This takes ownership of the pointer.
         */
        void SetBatchNum(int* aBatchNum);

        /**
         * @brief Gets the accumulated batch number array.
         * @return Pointer to accumulated batch numbers.
         */
        int* GetBatchNumAccum() { return mpBatchNumAccum; }
        const int* GetBatchNumAccum() const { return mpBatchNumAccum; }

        /**
         * @brief Sets the accumulated batch number array.
         * @param[in] aBatchNumAccum Pointer to accumulated batch numbers.
         * @note This takes ownership of the pointer.
         */
        void SetBatchNumAccum(int* aBatchNumAccum);

        /**
         * @brief Gets the accumulated batch number squared array.
         * @return Pointer to accumulated batch numbers squared.
         */
        int* GetBatchNumSquareAccum() { return mpBatchNumSquareAccum; }
        const int* GetBatchNumSquareAccum() const { return mpBatchNumSquareAccum; }

        /**
         * @brief Sets the accumulated batch number squared array.
         * @param[in] aBatchNumSquareAccum Pointer to accumulated batch numbers squared.
         * @note This takes ownership of the pointer.
         */
        void SetBatchNumSquareAccum(int* aBatchNumSquareAccum);

        // Host leading dimension arrays
        magma_int_t* GetHostLDA() { return mHostLDA.Get(); }
        const magma_int_t* GetHostLDA() const { return mHostLDA.Get(); }
        
        magma_int_t* GetHostLDDA() { return mHostLDDA.Get(); }
        const magma_int_t* GetHostLDDA() const { return mHostLDDA.Get(); }

        // Device leading dimension arrays
        magma_int_t* GetDeviceLDA() { return mDeviceLDA.Get(); }
        const magma_int_t* GetDeviceLDA() const { return mDeviceLDA.Get(); }
        
        magma_int_t* GetDeviceLDDA() { return mDeviceLDDA.Get(); }
        const magma_int_t* GetDeviceLDDA() const { return mDeviceLDDA.Get(); }

        // Info and constant arrays
        magma_int_t* GetHostInfo() { return mHostInfo.Get(); }
        const magma_int_t* GetHostInfo() const { return mHostInfo.Get(); }
        
        magma_int_t* GetDeviceInfo() { return mDeviceInfo.Get(); }
        const magma_int_t* GetDeviceInfo() const { return mDeviceInfo.Get(); }
        
        magma_int_t* GetHostConst1() { return mHostConst1.Get(); }
        const magma_int_t* GetHostConst1() const { return mHostConst1.Get(); }
        
        magma_int_t* GetDeviceConst1() { return mDeviceConst1.Get(); }
        const magma_int_t* GetDeviceConst1() const { return mDeviceConst1.Get(); }
        
        magma_int_t* GetDeviceBatchNum() { return mDeviceBatchNum.Get(); }
        const magma_int_t* GetDeviceBatchNum() const { return mDeviceBatchNum.Get(); }

        // Conditioning-related leading dimensions
        magma_int_t* GetHostLDAConditioning() { return mHostLDAConditioning.Get(); }
        const magma_int_t* GetHostLDAConditioning() const { return mHostLDAConditioning.Get(); }
        
        magma_int_t* GetHostLDDAConditioning() { return mHostLDDAConditioning.Get(); }
        const magma_int_t* GetHostLDDAConditioning() const { return mHostLDDAConditioning.Get(); }
        
        magma_int_t* GetDeviceLDAConditioning() { return mDeviceLDAConditioning.Get(); }
        const magma_int_t* GetDeviceLDAConditioning() const { return mDeviceLDAConditioning.Get(); }
        
        magma_int_t* GetDeviceLDDAConditioning() { return mDeviceLDDAConditioning.Get(); }
        const magma_int_t* GetDeviceLDDAConditioning() const { return mDeviceLDDAConditioning.Get(); }

    private:
        int mBatchCount;  ///< Number of batches

        // Batch size information (not MAGMA-allocated, managed separately)
        int* mpBatchNum;              ///< Size of each batch
        int* mpBatchNumAccum;         ///< Accumulated batch sizes
        int* mpBatchNumSquareAccum;   ///< Accumulated squared batch sizes

        // Leading dimension arrays (MAGMA-allocated with RAII)
        helpers::MagmaHostMemory<magma_int_t> mHostLDA;    ///< Host leading dimension A
        helpers::MagmaHostMemory<magma_int_t> mHostLDDA;   ///< Host leading dimension device A
        helpers::MagmaDeviceMemory<magma_int_t> mDeviceLDA;  ///< Device leading dimension A
        helpers::MagmaDeviceMemory<magma_int_t> mDeviceLDDA; ///< Device leading dimension device A

        // Info and constant arrays (MAGMA-allocated with RAII)
        helpers::MagmaHostMemory<magma_int_t> mHostInfo;      ///< Host info array for error checking
        helpers::MagmaDeviceMemory<magma_int_t> mDeviceInfo;  ///< Device info array for error checking
        helpers::MagmaHostMemory<magma_int_t> mHostConst1;    ///< Host constant array (all 1s)
        helpers::MagmaDeviceMemory<magma_int_t> mDeviceConst1;///< Device constant array (all 1s)
        helpers::MagmaDeviceMemory<magma_int_t> mDeviceBatchNum; ///< Device batch numbers

        // Conditioning-related leading dimensions (MAGMA-allocated with RAII)
        helpers::MagmaHostMemory<magma_int_t> mHostLDAConditioning;    ///< Host LDA for conditioning
        helpers::MagmaHostMemory<magma_int_t> mHostLDDAConditioning;   ///< Host LDDA for conditioning
        helpers::MagmaDeviceMemory<magma_int_t> mDeviceLDAConditioning;  ///< Device LDA for conditioning
        helpers::MagmaDeviceMemory<magma_int_t> mDeviceLDDAConditioning; ///< Device LDDA for conditioning
    };

    /**
     * @brief Instantiates the BatchMetadata class for float and double types.
     * @tparam T Data Type: float or double
     */
    VECCHIAGP_INSTANTIATE_CLASS(BatchMetadata)

} // namespace vecchia::dataunits

#endif //VECCHIAGP_BATCHMETADATA_HPP

