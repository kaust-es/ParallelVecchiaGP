
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file GPUMemoryManager.hpp
 * @brief RAII-based memory management wrapper for MAGMA GPU memory allocations.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-16
**/

#ifndef VECCHIAGP_GPUMEMORYMANAGER_HPP
#define VECCHIAGP_GPUMEMORYMANAGER_HPP

#include <magma_v2.h>
#include <stdexcept>
#include <string>
#include <common/Definitions.hpp>

namespace vecchia::helpers {

    /**
     * @brief RAII wrapper for MAGMA host memory (CPU-side pinned memory).
     * Automatically allocates and deallocates pinned host memory using MAGMA API.
     * @tparam T Data type (float, double, int, etc.)
     */
    template<typename T>
    class MagmaHostMemory { 
    public:
        /**
         * @brief Constructs and allocates host memory.
         * @param[in] aSize Number of elements to allocate.
         * @throws std::runtime_error if allocation fails.
         */
        explicit MagmaHostMemory(size_t aSize = 0) : mpData(nullptr), mSize(aSize) {
            if (aSize > 0) {
                magma_int_t status = magma_malloc_cpu(reinterpret_cast<void**>(&mpData), aSize * sizeof(T));
                if (status != MAGMA_SUCCESS || mpData == nullptr) {
                    throw std::runtime_error("Failed to allocate MAGMA host memory of size " + std::to_string(aSize));
                }
            }
        }

        /**
         * @brief Deallocates host memory.
         */
        ~MagmaHostMemory() {
            if (mpData) {
                magma_free_cpu(mpData);
                mpData = nullptr;
            }
        }

        // Delete copy constructor and assignment operator to prevent double-free
        MagmaHostMemory(const MagmaHostMemory&) = delete;
        MagmaHostMemory& operator=(const MagmaHostMemory&) = delete;

        /**
         * @brief Move constructor.
         * @param[in] aOther The object to move from.
         */
        MagmaHostMemory(MagmaHostMemory&& aOther) noexcept : mpData(aOther.mpData), mSize(aOther.mSize) {
            aOther.mpData = nullptr;
            aOther.mSize = 0;
        }

        /**
         * @brief Move assignment operator.
         * @param[in] aOther The object to move from.
         * @return Reference to this object.
         */
        MagmaHostMemory& operator=(MagmaHostMemory&& aOther) noexcept {
            if (this != &aOther) {
                if (mpData) {
                    magma_free_cpu(mpData);
                }
                mpData = aOther.mpData;
                mSize = aOther.mSize;
                aOther.mpData = nullptr;
                aOther.mSize = 0;
            }
            return *this;
        }

        /**
         * @brief Gets the raw pointer to the allocated memory.
         * @return Pointer to the memory.
         */
        T* Get() { return mpData; }
        const T* Get() const { return mpData; }

        /**
         * @brief Gets the size of the allocated memory in elements.
         * @return Number of elements.
         */
        size_t Size() const { return mSize; }

        /**
         * @brief Releases ownership of the pointer without deallocating.
         * @return The raw pointer.
         * @note After calling this, the caller is responsible for deallocation.
         */
        T* Release() {
            T* temp = mpData;
            mpData = nullptr;
            mSize = 0;
            return temp;
        }

        /**
         * @brief Checks if the memory is allocated.
         * @return true if memory is allocated, false otherwise.
         */
        explicit operator bool() const { return mpData != nullptr; }

    private:
        T* mpData;      ///< Pointer to the allocated host memory
        size_t mSize;   ///< Number of elements allocated
    };
    /**
      * @brief Instantiates the MagmaHostMemory class for float and double types.
      * @tparam T Data Type: float or double
      *
      */
      VECCHIAGP_INSTANTIATE_CLASS(MagmaHostMemory)
    /**
     * @brief RAII wrapper for MAGMA device memory (GPU-side memory).
     * Automatically allocates and deallocates device memory using MAGMA API.
     * @tparam T Data type (float, double, int, etc.)
     */
    template<typename T>
    class MagmaDeviceMemory {
    public:
        /**
         * @brief Constructs and allocates device memory.
         * @param[in] aSize Number of elements to allocate.
         * @throws std::runtime_error if allocation fails.
         */
        explicit MagmaDeviceMemory(size_t aSize = 0) : mpData(nullptr), mSize(aSize) {
            if (aSize > 0) {
                magma_int_t status = magma_malloc(reinterpret_cast<void**>(&mpData), aSize * sizeof(T));
                if (status != MAGMA_SUCCESS || mpData == nullptr) {
                    throw std::runtime_error("Failed to allocate MAGMA device memory of size " + std::to_string(aSize));
                }
            }
        }

        /**
         * @brief Deallocates device memory.
         */
        ~MagmaDeviceMemory() {
            if (mpData) {
                magma_free(mpData);
                mpData = nullptr;
            }
        }

        // Delete copy constructor and assignment operator to prevent double-free
        MagmaDeviceMemory(const MagmaDeviceMemory&) = delete;
        MagmaDeviceMemory& operator=(const MagmaDeviceMemory&) = delete;

        /**
         * @brief Move constructor.
         * @param[in] aOther The object to move from.
         */
        MagmaDeviceMemory(MagmaDeviceMemory&& aOther) noexcept : mpData(aOther.mpData), mSize(aOther.mSize) {
            aOther.mpData = nullptr;
            aOther.mSize = 0;
        }

        /**
         * @brief Move assignment operator.
         * @param[in] aOther The object to move from.
         * @return Reference to this object.
         */
        MagmaDeviceMemory& operator=(MagmaDeviceMemory&& aOther) noexcept {
            if (this != &aOther) {
                if (mpData) {
                    magma_free(mpData);
                }
                mpData = aOther.mpData;
                mSize = aOther.mSize;
                aOther.mpData = nullptr;
                aOther.mSize = 0;
            }
            return *this;
        }

        /**
         * @brief Gets the raw pointer to the allocated memory.
         * @return Pointer to the memory.
         */
        T* Get() { return mpData; }
        const T* Get() const { return mpData; }

        /**
         * @brief Gets the size of the allocated memory in elements.
         * @return Number of elements.
         */
        size_t Size() const { return mSize; }

        /**
         * @brief Releases ownership of the pointer without deallocating.
         * @return The raw pointer.
         * @note After calling this, the caller is responsible for deallocation.
         */
        T* Release() {
            T* temp = mpData;
            mpData = nullptr;
            mSize = 0;
            return temp;
        }

        /**
         * @brief Checks if the memory is allocated.
         * @return true if memory is allocated, false otherwise.
         */
        explicit operator bool() const { return mpData != nullptr; }

    private:
        T* mpData;      ///< Pointer to the allocated device memory
        size_t mSize;   ///< Number of elements allocated
    };
    /**
      * @brief Instantiates the MagmaDeviceMemory class for float and double types.
      * @tparam T Data Type: float or double
      *
      */
      VECCHIAGP_INSTANTIATE_CLASS(MagmaDeviceMemory)    
    /**
     * @brief RAII wrapper for paired host and device memory.
     * Manages both host (CPU) and device (GPU) memory allocations together.
     * @tparam T Data type (float, double, int, etc.)
     */
    template<typename T>
    class HostDeviceMemory {
    public:
        /**
         * @brief Constructs and allocates both host and device memory.
         * @param[in] aHostSize Number of elements to allocate on host.
         * @param[in] aDeviceSize Number of elements to allocate on device.
         * @throws std::runtime_error if allocation fails.
         */
        HostDeviceMemory(size_t aHostSize = 0, size_t aDeviceSize = 0)
            : mHostMemory(aHostSize), mDeviceMemory(aDeviceSize) {}

        /**
         * @brief Gets the host memory pointer.
         * @return Pointer to host memory.
         */
        T* GetHost() { return mHostMemory.Get(); }
        const T* GetHost() const { return mHostMemory.Get(); }

        /**
         * @brief Gets the device memory pointer.
         * @return Pointer to device memory.
         */
        T* GetDevice() { return mDeviceMemory.Get(); }
        const T* GetDevice() const { return mDeviceMemory.Get(); }

        /**
         * @brief Gets the host memory size.
         * @return Number of elements allocated on host.
         */
        size_t GetHostSize() const { return mHostMemory.Size(); }

        /**
         * @brief Gets the device memory size.
         * @return Number of elements allocated on device.
         */
        size_t GetDeviceSize() const { return mDeviceMemory.Size(); }

        /**
         * @brief Checks if both host and device memory are allocated.
         * @return true if both are allocated, false otherwise.
         */
        explicit operator bool() const { return mHostMemory && mDeviceMemory; }

    private:
        MagmaHostMemory<T> mHostMemory;     ///< Host memory wrapper
        MagmaDeviceMemory<T> mDeviceMemory; ///< Device memory wrapper
    };
    /**
      * @brief Instantiates the HostDeviceMemory class for float and double types.
      * @tparam T Data Type: float or double
      *
      */
      VECCHIAGP_INSTANTIATE_CLASS(HostDeviceMemory)

} // namespace vecchia::helpers

#endif //VECCHIAGP_GPUMEMORYMANAGER_HPP

