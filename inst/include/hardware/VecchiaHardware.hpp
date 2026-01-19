
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file VecchiaHardware.hpp
 * @brief Contains the definition of the VecchiaHardware class.
 * @details Handles initialization of:
 *          - KBLAS (for Parallel/Scalar Vecchia)
 *          - MAGMA (for Block Vecchia)
 *          - MAGMA + MPI (for Scaled Block Vecchia)
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_VECCHIAGP_HARDWARE_HPP
#define VECCHIAGP_VECCHIAGP_HARDWARE_HPP

#include <magma_v2.h>
#include <cuComplex.h>
#ifdef USE_KBLAS
#include <kblas.h>
#endif

#include <common/Definitions.hpp>

/**
 * @brief Class represents the hardware configuration for VecchiaGP.
 * @details Handles initialization based on Vecchia approximation type:
 *          - PARALLEL_VECCHIA_GP: KBLAS + GPU
 *          - PARALLEL_BLOCK_VECCHIA_GP: MAGMA + GPU
 *          - PARALLEL_SCALED_BLOCK_VECCHIA_GP: MAGMA + GPU + MPI
 */
class VecchiaHardware {

public:
    /**
     * @brief Constructor for VecchiaHardware.
     * @details Initializes hardware based on Vecchia type:
     *          - PARALLEL_VECCHIA_GP: KBLAS + GPU
     *          - PARALLEL_BLOCK_VECCHIA_GP: MAGMA + GPU
     *          - PARALLEL_SCALED_BLOCK_VECCHIA_GP: MAGMA + GPU + MPI
     * @param[in] aVecchiaType The Vecchia approximation type
     * @param[in] aCoreNumber Number of CPU cores (OpenMP threads)
     * @param[in] aGpuNumber Number of GPUs (currently only single GPU supported)
     */
    explicit VecchiaHardware(const vecchia::common::VecchiaType &aVecchiaType, 
                            const int &aCoreNumber,
                            const int &aGpuNumber);

    /**
     * @brief Destructor - cleans up all hardware resources
     */
    ~VecchiaHardware();

    /**
     * @brief Static hardware initialization (alternative to constructor)
     * @param[in] aVecchiaType The Vecchia approximation type
     * @param[in] aCoreNumber Number of CPU cores
     * @param[in] aGpuNumber Number of GPUs
     */
    static void InitHardware(const vecchia::common::VecchiaType &aVecchiaType, 
                            const int &aCoreNumber, 
                            const int &aGpuNumber);

    /**
     * @brief Finalize hardware resources
     */
    void FinalizeHardware();

    /**
     * @brief Get number of OpenMP threads
     * @return Thread count
     */
    static int GetOMPThreads() { return mOMPThreads; }

    // ========== MAGMA Methods (for Block & Scaled Block) ==========
    
    /**
     * @brief Get the primary MAGMA queue
     * @return MAGMA queue (NULL if using KBLAS)
     */
    static magma_queue_t GetQueue() { return mQueues[0]; }
    
    /**
     * @brief Get secondary MAGMA queue
     * @return MAGMA queue (NULL if using KBLAS)
     */
    static magma_queue_t GetQueue2() { return mQueues[1]; }
    
#ifdef USE_KBLAS
    // ========== KBLAS Methods (for Parallel/Scalar) ==========
    
    /**
     * @brief Get KBLAS handle for specified GPU
     * @param[in] aGpuId GPU device ID
     * @return KBLAS handle (NULL if using MAGMA)
     */
    static kblasHandle_t GetKblasHandle(int aGpuId = 0);
    
    /**
     * @brief Get number of KBLAS handles (one per GPU)
     * @return Number of handles
     */
    static int GetNumKblasHandles() { return mNumKblasHandles; }
#endif // USE_KBLAS

    // ========== MPI Methods (for Scaled Block only) ==========
    
    /**
     * @brief Check if MPI is initialized
     * @return true if MPI is active
     */
    static bool IsMPIInitialized() { return mIsMPIInit; }
    
    /**
     * @brief Get MPI rank
     * @return Rank (0 if MPI not initialized)
     */
    static int GetMPIRank() { return mMPIRank; }
    
    /**
     * @brief Get MPI size
     * @return Size (1 if MPI not initialized)
     */
    static int GetMPISize() { return mMPISize; }
    
    /**
     * @brief Get local GPU ID for this MPI rank
     * @return GPU device ID
     */
    static int GetLocalGPUId() { return mLocalGPUId; }

private:
    // ========== Initialization Helpers ==========
    
#ifdef USE_KBLAS
    /**
     * @brief Initialize KBLAS handles and GPU context
     * @param[in] aGpuNumber Number of GPUs to use
     */
    void InitKBLAS(const int &aGpuNumber);
#endif
    
    /**
     * @brief Initialize MAGMA queues and GPU context
     * @param[in] aGpuNumber Number of GPUs to use
     */
    void InitMAGMA(const int &aGpuNumber);
    
    /**
     * @brief Initialize MPI and determine GPU assignment
     */
    void InitMPI();
    
#ifdef USE_KBLAS
    /**
     * @brief Finalize KBLAS resources
     */
    void FinalizeKBLAS();
#endif
    
    /**
     * @brief Finalize MAGMA resources
     */
    void FinalizeMAGMA();
    
    /**
     * @brief Finalize MPI resources
     */
    void FinalizeMPI();

    // ========== Static Members ==========
    
    // Common
    static int mOMPThreads;
    static bool mIsInitialized;
    
    // MAGMA (for Block & Scaled Block)
    static magma_queue_t mQueues[3];  // 2 queues + NULL sentinel
    
#ifdef USE_KBLAS
    // KBLAS (for Parallel/Scalar)
    static kblasHandle_t* mKblasHandles;  // Array of handles (multi-GPU)
    static int mNumKblasHandles;
#endif
    
    // MPI (for Scaled Block only)
    static bool mIsMPIInit;
    static int mMPIRank;
    static int mMPISize;
    static int mLocalGPUId;  // GPU ID for this MPI rank
    static int mMPIInstanceCount;  // Reference counter for MPI-using instances
};

#endif // VECCHIAGP_VECCHIAGP_HARDWARE_HPP