
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file VecchiaHardware.cpp
 * @brief Contains the implementation of the VecchiaHardware class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2025-09-29
**/
#include <omp.h>
#include <stdexcept>
#include <string>
#include <cstdlib>

#include <cuda_runtime.h>


#ifdef USE_MPI
#include <mpi.h>
#endif

#include <hardware/VecchiaHardware.hpp>
#include <helpers/CommunicatorMPI.hpp>
#include <utilities/Logger.hpp>
#include <utilities/EnumStringParser.hpp>

using namespace vecchia::common;
using namespace vecchia::helpers;

// ========== Initialize Static Members ==========
int VecchiaHardware::mOMPThreads = 1;
bool VecchiaHardware::mIsInitialized = false;
// MAGMA
magma_queue_t VecchiaHardware::mQueues[3] = {NULL, NULL, NULL};
#ifdef USE_KBLAS
// KBLAS
kblasHandle_t* VecchiaHardware::mKblasHandles = nullptr;
int VecchiaHardware::mNumKblasHandles = 0;
#endif
// MPI
bool VecchiaHardware::mIsMPIInit = false;
int VecchiaHardware::mMPIRank = 0;
int VecchiaHardware::mMPISize = 1;
int VecchiaHardware::mLocalGPUId = 0;
int VecchiaHardware::mMPIInstanceCount = 0;  // Reference counter for MPI-using instances

// ========== Constructor ==========

VecchiaHardware::VecchiaHardware(const VecchiaType &aVecchiaType, 
                                 const int &aCoreNumber,
                                 const int &aGpuNumber) {
    
    // Store configuration
    mOMPThreads = aCoreNumber;
    
    // Set OpenMP threads
    omp_set_num_threads(aCoreNumber);
    
    LOGGER("** Initializing VecchiaGP Hardware **")
    LOGGER("   CPU Cores: " + std::to_string(aCoreNumber))
    LOGGER("   GPUs: " + std::to_string(aGpuNumber))
    
    // Conditional initialization based on Vecchia type
    switch (aVecchiaType) {
        
#ifdef USE_KBLAS
        case VecchiaType::PARALLEL_VECCHIA_GP:
            // Scalar Vecchia: KBLAS + MAGMA (MAGMA for memory management, KBLAS for computations)
            LOGGER("   Backend: KBLAS + MAGMA")
            InitMAGMA(aGpuNumber);  // Initialize MAGMA first for memory management
            InitKBLAS(aGpuNumber);  // Then initialize KBLAS for strided batched operations
            break;
#endif
            
        case VecchiaType::PARALLEL_BLOCK_VECCHIA_GP:
            // Block Vecchia: MAGMA only
            LOGGER("   Backend: MAGMA")
            InitMAGMA(aGpuNumber);
            break;
            
        case VecchiaType::PARALLEL_SCALED_BLOCK_VECCHIA_GP:
            // Scaled Block Vecchia: MAGMA + MPI
            LOGGER("   Backend: MAGMA + MPI")
            InitMPI();      // Must be first to determine GPU assignment
            mMPIInstanceCount++;  // Increment reference counter for MPI-using instances
            InitMAGMA(1);   // One GPU per MPI rank
            omp_set_num_threads(aCoreNumber);
            break;
            
        default:
            throw std::runtime_error("Unknown Vecchia type in hardware initialization");
    }
    
    mIsInitialized = true;
    LOGGER("** Hardware initialization complete **")
}

// ========== Initialization Methods ==========

#ifdef USE_KBLAS
void VecchiaHardware::InitKBLAS(const int &aGpuNumber) {
    mNumKblasHandles = aGpuNumber;
    mKblasHandles = new kblasHandle_t[aGpuNumber];
    
    for (int g = 0; g < aGpuNumber; g++) {
        cudaSetDevice(g);
        kblasCreate(&mKblasHandles[g]);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error during KBLAS initialization on GPU " + 
                                    std::to_string(g) + ": " + cudaGetErrorString(err));
        }
    }
    
    LOGGER("   KBLAS: Initialized " + std::to_string(aGpuNumber) + " handle(s)")
}
#endif // USE_KBLAS

void VecchiaHardware::InitMAGMA(const int &aGpuNumber) {
    // Initialize MAGMA library
    magma_init();
    
    // For now, we only support single GPU with MAGMA
    // For MPI, the GPU is already set by InitMPI
    int device = (mIsMPIInit) ? mLocalGPUId : 0;
    
#if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
    magma_setdevice(device);
#else
    throw std::runtime_error("No GPU support found in MAGMA");
#endif
    
    // Create MAGMA queues if not already created
    if (mQueues[0] == NULL) {
        magma_queue_create(device, &mQueues[0]);
        magma_queue_create(device, &mQueues[1]);
        mQueues[2] = NULL;  // Sentinel
    }
    
    LOGGER("   MAGMA: Initialized on GPU " + std::to_string(device))
}

void VecchiaHardware::InitMPI() {
#ifdef USE_MPI
    // Check if MPI is already initialized (might be called by user or previous instance)
    int mpi_initialized = 0;
    int mpi_error = MPI_Initialized(&mpi_initialized);
    
    // Check if MPI is finalized (should not happen, but be safe)
    int mpi_finalized = 0;
    if (mpi_initialized) {
        MPI_Finalized(&mpi_finalized);
    }
    
    if (!mpi_initialized && !mpi_finalized) {
        // We need to initialize MPI
        // Only initialize if MPI is not already initialized and not finalized
        // NOTE: If running with mpirun, MPI should already be initialized by the MPI runtime
        // If MPI_Initialized() returns false when mpirun is used, it might be a timing issue
        // or configuration problem. We'll try to initialize and catch any errors.
        int provided;
        mpi_error = MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
        if (mpi_error != MPI_SUCCESS) {
            // MPI initialization failed - this can happen if:
            // 1. mpirun is used incorrectly (MPI should already be initialized)
            // 2. There's a configuration issue with Open MPI
            // 3. There's a conflict with how R loads shared libraries
            // Check if MPI is now initialized (might have been initialized by mpirun)
            int mpi_initialized_after = 0;
            MPI_Initialized(&mpi_initialized_after);
            if (mpi_initialized_after) {
                // MPI is now initialized (might have been initialized by mpirun during the failed call)
                mIsMPIInit = true;
                LOGGER("   MPI: Initialized by mpirun (detected after Init_thread attempt)")
            } else {
                // MPI is still not initialized - this is a real error
                throw std::runtime_error("MPI_Init_thread failed with error code: " + 
                                        std::to_string(mpi_error) + 
                                        ". If using mpirun, MPI should already be initialized. "
                                        "Check your MPI configuration and ensure you're using mpirun correctly.");
            }
        } else {
            mIsMPIInit = true;
            LOGGER("   MPI: Initialized by VecchiaGP")
        }
    } else if (mpi_initialized && !mpi_finalized) {
        // MPI is already initialized (either by previous VecchiaHardware instance or externally via mpirun)
        // We still need to track it for reference counting
        mIsMPIInit = true;  // Set this so we can properly track and finalize later
        LOGGER("   MPI: Already initialized (external or previous instance)")
    } else if (mpi_finalized) {
        // MPI was finalized - this shouldn't happen, but handle it gracefully
        throw std::runtime_error("MPI is already finalized. Cannot initialize VecchiaGP with MPI.");
    } else {
        // MPI initialization check failed
        throw std::runtime_error("MPI_Initialized() failed. MPI may not be properly configured.");
    }
    
    // Re-check MPI initialization status after any initialization attempts
    // This is important because when using mpirun, MPI might be initialized
    // by the MPI runtime but MPI_Initialized() might not detect it immediately
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        mIsMPIInit = true;
    }
    
    // Get MPI rank and size (only if MPI is initialized)
    if (mpi_initialized) {
        mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &mMPIRank);
        if (mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Comm_rank failed with error code: " + 
                                    std::to_string(mpi_error));
        }
        
        mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &mMPISize);
        if (mpi_error != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Comm_size failed with error code: " + 
                                    std::to_string(mpi_error));
        }
    } else {
        // MPI is not initialized - this should not happen if we're using scaled_block
        throw std::runtime_error("MPI is not initialized. For scaled_block Vecchia, "
                                "MPI must be initialized. If using mpirun, ensure MPI is properly configured.");
    }
    
    // Determine local GPU assignment
    // Create node-local communicator
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 
                       mMPIRank, MPI_INFO_NULL, &node_comm);
    
    int local_rank, local_size;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    
    // Get number of GPUs on this node
    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_count == 0) {
        throw std::runtime_error("No GPUs found on node for MPI rank " + 
                                std::to_string(mMPIRank));
    }
    
    // Assign GPU: round-robin if more ranks than GPUs
    mLocalGPUId = local_rank % gpu_count;
    cudaSetDevice(mLocalGPUId);
    
    MPI_Comm_free(&node_comm);
    
    if (mMPIRank == 0) {
        LOGGER("   MPI: Size = " + std::to_string(mMPISize))
        if (local_size > gpu_count) {
            LOGGER("   MPI: Warning - " + std::to_string(local_size) + 
                   " ranks sharing " + std::to_string(gpu_count) + " GPUs per node")
        }
    }
    LOGGER("   MPI: Rank " + std::to_string(mMPIRank) + " using GPU " + 
           std::to_string(mLocalGPUId))
    
#else
    throw std::runtime_error("MPI support not compiled. Please rebuild with USE_MPI=ON");
#endif
}

// ========== Static Initialization Method ==========

void VecchiaHardware::InitHardware(const VecchiaType &aVecchiaType, 
                                  const int &aCoreNumber, 
                                  const int &aGpuNumber) {
    // Static initialization (alternative to constructor)
    // Just calls constructor
    VecchiaHardware hw(aVecchiaType, aCoreNumber, aGpuNumber);
}

// ========== Finalization Methods ==========

#ifdef USE_KBLAS
void VecchiaHardware::FinalizeKBLAS() {
    if (mKblasHandles != nullptr) {
        for (int g = 0; g < mNumKblasHandles; g++) {
            cudaSetDevice(g);
            kblasDestroy(&mKblasHandles[g]);
        }
        delete[] mKblasHandles;
        mKblasHandles = nullptr;
        mNumKblasHandles = 0;
        LOGGER("   KBLAS: Finalized")
    }
}
#endif // USE_KBLAS

void VecchiaHardware::FinalizeMAGMA() {
    // Destroy MAGMA queues
    if (mQueues[0] != NULL) {
        magma_queue_destroy(mQueues[0]);
        mQueues[0] = NULL;
    }
    if (mQueues[1] != NULL) {
        magma_queue_destroy(mQueues[1]);
        mQueues[1] = NULL;
    }
    mQueues[2] = NULL;
    
    // Finalize MAGMA
    magma_finalize();
    
    LOGGER("   MAGMA: Finalized")
}

void VecchiaHardware::FinalizeMPI() {
#ifdef USE_MPI
    // Decrement reference counter for MPI-using instances
    if (mMPIInstanceCount > 0) {
        mMPIInstanceCount--;
    }
    
    // Only finalize MPI when the last instance is destroyed
    // NOTE: For R wrapper usage, we should NOT finalize MPI between function calls
    // as MPI is needed across multiple R function calls. MPI will be finalized when R exits.
    if (mIsMPIInit && mMPIInstanceCount == 0) {
        // Check if MPI is still initialized before finalizing
        int mpi_initialized;
        MPI_Initialized(&mpi_initialized);
        
        // Check environment variable to detect R wrapper usage (before any finalization)
        const char* r_home = std::getenv("R_HOME");
        bool is_r_wrapper = (r_home != nullptr);
        
        if (mpi_initialized) {
            // Only finalize if MPI is still initialized
            int mpi_finalized;
            MPI_Finalized(&mpi_finalized);
            
            if (!mpi_finalized) {
                if (is_r_wrapper) {
                    // Running from R - don't finalize MPI, it will be finalized when R exits
                    LOGGER("   MPI: Not finalized (R wrapper detected, will be finalized on R exit)")
                } else {
                    // Not running from R - finalize MPI normally
                    MPI_Finalize();
                    LOGGER("   MPI: Finalized (last instance destroyed)")
                    mIsMPIInit = false;
                }
            } else {
                LOGGER("   MPI: Already finalized")
                mIsMPIInit = false;
            }
        } else {
            LOGGER("   MPI: Not initialized, skipping finalization")
            mIsMPIInit = false;
        }
    } else if (mMPIInstanceCount > 0) {
        LOGGER("   MPI: Not finalized (still " + std::to_string(mMPIInstanceCount) + " active instances)")
    }
#endif
}

void VecchiaHardware::FinalizeHardware() {
    if (!mIsInitialized) return;
    
    LOGGER("** Finalizing VecchiaGP Hardware **")
    
    // Finalize based on what was actually initialized
    // Check in reverse order of dependencies: MPI -> MAGMA -> KBLAS
    
    // Finalize MPI if it was initialized
    if (mIsMPIInit) {
        FinalizeMPI();
    }
    
    // Finalize MAGMA if queues were created
    if (mQueues[0] != NULL) {
        FinalizeMAGMA();
    }
    
#ifdef USE_KBLAS
    // Finalize KBLAS if handles were created
    if (mKblasHandles != nullptr) {
        FinalizeKBLAS();
    }
#endif
    
    // Remove hardware initialization from communicator
    CommunicatorMPI::GetInstance()->RemoveHardwareInitialization();
    
    mIsInitialized = false;
    LOGGER("** Hardware finalization complete **")
}

// ========== Destructor ==========

VecchiaHardware::~VecchiaHardware() {
    FinalizeHardware();
}

// ========== Accessor Methods ==========

#ifdef USE_KBLAS
kblasHandle_t VecchiaHardware::GetKblasHandle(int aGpuId) {
    if (mKblasHandles == nullptr || aGpuId >= mNumKblasHandles) {
        throw std::runtime_error("KBLAS handle not initialized for GPU " + 
                                std::to_string(aGpuId));
    }
    return mKblasHandles[aGpuId];
}
#endif // USE_KBLAS
