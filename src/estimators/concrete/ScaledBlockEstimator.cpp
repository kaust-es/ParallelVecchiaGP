
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file ScaledBlockEstimator.cpp
 * @brief Implementation of Scaled Block Vecchia approximation
 * @details Implements distributed Scaled Block Vecchia with RAC partitioning, block-level NN search,
 *          and GPU-accelerated computations using MAGMA vbatched operations.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <set>
#include <tuple>
#include <numeric>
#include <chrono>

#include <estimators/concrete/ScaledBlockEstimator.hpp>
#include <hardware/VecchiaHardware.hpp>
#include <data-units/VecchiaGBData.hpp>
#include <data-units/BlockInfo.hpp>
#include <configurations/Configurations.hpp>
#include <helpers/DistanceCalculationHelpers.hpp>
#include <utilities/Flops.hpp>
#include <common/GpuData.hpp>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <magma_v2.h>
#include <kblas.h>
#endif

using namespace vecchia::estimators;
using namespace vecchia::configurations;
using namespace vecchia::dataunits;
using namespace vecchia::common;
using namespace vecchia::helpers;

// Helper structure for point metadata
struct PointMetadata {
    std::vector<double> coordinates;
    double observation;
};

// Use shared BlockInfo definition from header
using BlockInfo = vecchia::dataunits::BlockInfo;

#ifdef USE_CUDA
// GpuData struct is now defined in common/GpuData.hpp

// Optimization data structure
struct OptimizationData {
    GpuData *gpuData;
    Configurations *configs;
    int rank;
    MPI_Comm comm;
};

// Helper functions for error checking
inline void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkMagmaError(magma_int_t error) {
    if (error != MAGMA_SUCCESS) {
        std::cerr << "MAGMA Error: " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

// Helper function to generate random double [0,1]
inline double generateRandomDouble() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// Calculate Euclidean distance
inline double calculateDistance(const std::vector<double>& p1, const std::vector<double>& p2) {
    double dist = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double dx = p1[i] - p2[i];
        dist += dx * dx;
    }
    return std::sqrt(dist);
}

// Generate random N-dimensional points
std::vector<PointMetadata> generateRandomPoints(int numPointsPerProcess, int dim, int rank) {
    std::vector<PointMetadata> pointsMetadata(numPointsPerProcess);
    std::srand(rank + 1);  // Fix seed for reproducibility
    
    for (int i = 0; i < numPointsPerProcess; ++i) {
        pointsMetadata[i].coordinates.resize(dim);
        for (int j = 0; j < dim; ++j) {
            pointsMetadata[i].coordinates[j] = generateRandomDouble();
        }
        pointsMetadata[i].observation = generateRandomDouble();
    }
    
    return pointsMetadata;
}

// Calculate centers of gravity
std::vector<std::vector<double>> calculateCentersOfGravity(const std::vector<std::vector<PointMetadata>> &finerPartitions, int dim) {
    int numBlocks = finerPartitions.size();
    std::vector<std::vector<double>> centers(numBlocks, std::vector<double>(dim));
    
    for (size_t i = 0; i < numBlocks; ++i) {
        auto &blockmetadata = finerPartitions[i];
        if (blockmetadata.empty()) continue;
        
        std::vector<double> sum(dim, 0.0);
        for (auto &pointmeta : blockmetadata) {
            for (int j = 0; j < dim; ++j) {
                sum[j] += pointmeta.coordinates[j];
            }
        }
        for (int j = 0; j < dim; ++j) {
            centers[i][j] = sum[j] / blockmetadata.size();
        }
    }
    
    return centers;
}

// AllGather centers (MPI communication)
void AllGatherCenters(const std::vector<std::vector<double>> &centers, std::vector<std::pair<std::vector<double>, int>> &allCenters, int dim, int rank, int size) {
#ifdef USE_MPI
    int numCenters = centers.size();
    std::vector<int> recvCounts(size, 0);
    MPI_Allgather(&numCenters, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    std::vector<int> displacements(size, 0);
    int totalCenters = 0;
    for (int i = 0; i < size; ++i) {
        displacements[i] = totalCenters * (dim + 1);
        totalCenters += recvCounts[i];
        recvCounts[i] *= (dim + 1);
    }
    
    std::vector<double> sendBuffer(numCenters * (dim + 1));
    for (int i = 0; i < numCenters; ++i) {
        for (int j = 0; j < dim; ++j) {
            sendBuffer[i * (dim + 1) + j] = centers[i][j];
        }
        sendBuffer[i * (dim + 1) + dim] = static_cast<double>(rank);
    }
    
    std::vector<double> recvBuffer(totalCenters * (dim + 1));
    MPI_Allgatherv(sendBuffer.data(), numCenters * (dim + 1), MPI_DOUBLE,
                   recvBuffer.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    
    allCenters.clear();
    allCenters.resize(totalCenters);
    for (int i = 0; i < totalCenters; ++i) {
        std::vector<double> centerCoords(dim);
        for (int j = 0; j < dim; ++j) {
            centerCoords[j] = recvBuffer[i * (dim + 1) + j];
        }
        int centerRank = static_cast<int>(recvBuffer[i * (dim + 1) + dim]);
        allCenters[i] = std::make_pair(centerCoords, centerRank);
    }
#else
    allCenters.clear();
    for (size_t i = 0; i < centers.size(); ++i) {
        allCenters.push_back(std::make_pair(centers[i], rank));
    }
#endif
}

// Reorder centers
void reorderCenters(std::vector<std::vector<double>> &centers,
                   std::vector<std::pair<std::vector<double>, int>> &allCenters,
                   std::vector<int> &permutation,
                   std::vector<int> &localPermutation,
                   int seed, int rank, int size) {
#ifdef USE_MPI
    std::mt19937 gen(seed);
    permutation.resize(allCenters.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), gen);
    
    int numLocalCenters = centers.size();
    std::vector<int> centerCounts(size);
    MPI_Allgather(&numLocalCenters, 1, MPI_INT, centerCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    std::vector<int> displacements(size, 0);
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i-1] + centerCounts[i-1];
    }
    
    localPermutation.clear();
    localPermutation.reserve(numLocalCenters);
    for (int i = 0; i < numLocalCenters; ++i) {
        localPermutation.push_back(permutation[displacements[rank] + i]);
    }
#else
    permutation.resize(allCenters.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    localPermutation = permutation;
#endif
}

// Create block information
std::vector<BlockInfo> createBlockInfo(const std::vector<std::vector<PointMetadata>> &finerPartitions,
                                       const std::vector<std::vector<double>> &localCenters,
                                       const std::vector<std::pair<std::vector<double>, int>> &allCenters,
                                       const std::vector<int> &permutation,
                                       const std::vector<int> &localPermutation) {
    std::vector<BlockInfo> blockInfos;
    int numBlocksLocal = localCenters.size();
    blockInfos.resize(numBlocksLocal);
    
    for (int i = 0; i < numBlocksLocal; ++i) {
        BlockInfo blockInfo;
        blockInfo.localOrder = i;
        blockInfo.globalOrder = permutation[localPermutation[i]];
        blockInfo.center = localCenters[i];
        
        for (const auto &pointMetadata : finerPartitions[i]) {
            blockInfo.blocks.push_back(pointMetadata.coordinates);
            blockInfo.observations_blocks.push_back(pointMetadata.observation);
        }
        
        blockInfos[i] = blockInfo;
    }
    
    return blockInfos;
}


// Process and send blocks (simplified version - full MPI communication in production)
std::vector<BlockInfo> processAndSendBlocks(std::vector<BlockInfo> &blockInfos,
    const std::vector<std::pair<std::vector<double>, int>> &CenterRanks,
    double distance_threshold,
    const std::vector<int>& permutation,
    const std::vector<int>& localPermutation,
    int dim, int rank, int size) {
    
    std::vector<BlockInfo> receivedBlocks;
    
    // For simplicity in this stub, just return blocks from this process
    // In production, this would involve MPI_Alltoallv communication
    for (auto& block : blockInfos) {
        receivedBlocks.push_back(block);
    }
    
    return receivedBlocks;
}

// Nearest neighbor search
void nearest_neighbor_search(std::vector<BlockInfo> &blockInfos, std::vector<BlockInfo> &receivedBlocks,
                             int m, double distance_threshold) {
    // Sort received blocks by global order
    std::sort(receivedBlocks.begin(), receivedBlocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
        return a.globalOrder < b.globalOrder;
    });
    
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        auto& block = blockInfos[i];
        std::vector<std::tuple<double, std::vector<double>, double>> distancesMeta;
        
        for (auto& prevBlock : receivedBlocks) {
            if (prevBlock.globalOrder >= block.globalOrder) break;
            
            for (size_t j = 0; j < prevBlock.blocks.size(); ++j) {
                double distance = calculateDistance(block.center, prevBlock.blocks[j]);
                if (distance < distance_threshold || block.globalOrder <= 200) {
                    distancesMeta.emplace_back(distance, prevBlock.blocks[j], prevBlock.observations_blocks[j]);
                }
            }
        }
        
        if (block.globalOrder == 0) continue;
        
        // Sort and keep m nearest neighbors
        std::sort(distancesMeta.begin(), distancesMeta.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        
        for (int k = 0; k < std::min(m, (int)distancesMeta.size()); ++k) {
            block.nearestNeighbors.push_back(std::get<1>(distancesMeta[k]));
            block.observations_nearestNeighbors.push_back(std::get<2>(distancesMeta[k]));
        }
    }
}

#ifdef USE_CUDA

// Calculate total GFLOPS for the computation
double gflopsTotal(const GpuData &gpuData, Configurations &aConfigurations) {
    int rank = VecchiaHardware::GetMPIRank();
    int dim = aConfigurations.GetDimensionSize();
    size_t batchCount = gpuData.ldda_locs.size() - 1;
    double gflops = 0;
    
    for (size_t i = 0; i < batchCount; ++i) {
        // 1. Matrix generation
        gflops += (3 * dim + 11) * (gpuData.lda_locs[i] * gpuData.lda_locs[i] + 
                    gpuData.lda_locs_neighbors[i] * gpuData.lda_locs_neighbors[i] + 
                    gpuData.lda_locs[i] * gpuData.lda_locs_neighbors[i]) / 1e9;
        
        // 2. Conditioning correction
        // Cholesky factorization
        gflops += FLOPS_DPOTRF(gpuData.lda_locs_neighbors[i]) / 1e9;
        // TRSM
        gflops += FLOPS_DTRSM(MagmaLeft, gpuData.lda_locs_neighbors[i], gpuData.lda_locs[i]) / 1e9;
        gflops += FLOPS_DTRSM(MagmaLeft, gpuData.lda_locs_neighbors[i], 1) / 1e9;
        // Matrix multiplication
        gflops += FLOPS_DGEMM(gpuData.lda_locs[i], gpuData.lda_locs[i], gpuData.lda_locs_neighbors[i]) / 1e9;
        gflops += FLOPS_DGEMM(gpuData.lda_locs[i], 1, gpuData.lda_locs_neighbors[i]) / 1e9;
        // Addition
        gflops += FLOPS_DAXPY(gpuData.lda_locs[i] * gpuData.lda_locs[i]) / 1e9;
        gflops += FLOPS_DAXPY(gpuData.lda_locs[i]) / 1e9;
        
        // 3. Log-likelihood calculation
        // Cholesky factorization + TRSV + norm + determinant
        gflops += FLOPS_DPOTRF(gpuData.lda_locs[i]) / 1e9;
        gflops += FLOPS_DTRSM(MagmaLeft, gpuData.lda_locs[i], 1) / 1e9;
        gflops += 4 * FLOPS_DAXPY(gpuData.lda_locs[i]) / 1e9;
    }
    
    double total_gflops = 0;
    // MPI sum for gflops
    MPI_Allreduce(&gflops, &total_gflops, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Total Gflops: " << total_gflops << std::endl;
    }
    
    return total_gflops;
}

// Function to copy data from CPU to GPU
GpuData copyDataToGPU(Configurations &aConfigurations, const std::vector<BlockInfo> &blockInfos, magma_queue_t queue) {
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int gpu_id = VecchiaHardware::GetLocalGPUId();
    int dim = aConfigurations.GetDimensionSize();

    // Set the GPU
    checkCudaError(cudaSetDevice(gpu_id));

    GpuData gpuData;
    // CPU leading dimensions (+1 for magma)
    gpuData.lda_locs.resize(blockInfos.size() + 1);
    gpuData.lda_locs_neighbors.resize(blockInfos.size() + 1);
    // GPU leading dimensions
    gpuData.ldda_locs.resize(blockInfos.size() + 1);
    gpuData.ldda_neighbors.resize(blockInfos.size() + 1);
    gpuData.ldda_cov.resize(blockInfos.size() + 1);
    gpuData.ldda_cross_cov.resize(blockInfos.size() + 1);
    gpuData.ldda_conditioning_cov.resize(blockInfos.size() + 1);
    gpuData.h_const1.resize(blockInfos.size() + 1);

    // Allocate arrays of pointers on the host
    gpuData.h_locs_array = new double *[blockInfos.size() * dim];
    gpuData.h_locs_neighbors_array = new double *[blockInfos.size() * dim];
    gpuData.h_observations_array = new double *[blockInfos.size()];
    gpuData.h_observations_neighbors_array = new double *[blockInfos.size()];
    gpuData.h_cov_array = new double *[blockInfos.size()];
    gpuData.h_cross_cov_array = new double *[blockInfos.size()];
    gpuData.h_conditioning_cov_array = new double *[blockInfos.size()];
    gpuData.h_observations_neighbors_copy_array = new double *[blockInfos.size()];
    gpuData.h_observations_copy_array = new double *[blockInfos.size()];
    gpuData.h_mu_correction_array = new double *[blockInfos.size()];
    gpuData.h_cov_correction_array = new double *[blockInfos.size()];

    // Allocate array of pointers for the device
    checkCudaError(cudaMalloc(&gpuData.d_locs_array, blockInfos.size() * dim * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_locs_neighbors_array, blockInfos.size() * dim * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_points_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cross_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_copy_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_observations_copy_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_mu_correction_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_cov_correction_array, blockInfos.size() * sizeof(double *)));
    checkCudaError(cudaMalloc(&gpuData.d_range_device, dim * sizeof(double)));

    // Calculate the total memory needed for blocks and nearest neighbors
    size_t total_observations_points_size = 0;
    size_t total_observations_nearestNeighbors_size = 0;
    size_t total_cov_size = 0;
    size_t total_cross_cov_size = 0;
    size_t total_conditioning_cov_size = 0;
    size_t total_locs_size_host = 0;
    size_t total_locs_nearestNeighbors_size_host = 0;
    size_t total_locs_size_device = 0;
    size_t total_locs_nearestNeighbors_size_device = 0;

    gpuData.numPointsPerProcess = 0;
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        // Number of clusters and their nearest neighbors
        int m_blocks = blockInfos[i].blocks.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // Number of points per process
        gpuData.numPointsPerProcess += m_blocks;
        // 32 is the aligned 32 threads in a warp in GPU
        gpuData.ldda_locs[i] = magma_roundup(m_blocks, 32); 
        gpuData.ldda_neighbors[i] = magma_roundup(m_nearest_neighbor, 32); 
        gpuData.lda_locs[i] = m_blocks;
        gpuData.lda_locs_neighbors[i] = m_nearest_neighbor;
        gpuData.ldda_cov[i] = gpuData.ldda_locs[i];
        gpuData.ldda_cross_cov[i] = gpuData.ldda_neighbors[i];
        gpuData.ldda_conditioning_cov[i] = gpuData.ldda_neighbors[i];
        gpuData.h_const1[i] = 1;
        // Total size of contiguous memory 
        total_cov_size += gpuData.ldda_cov[i] * m_blocks * sizeof(double);
        total_conditioning_cov_size += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor * sizeof(double);
        total_cross_cov_size += gpuData.ldda_conditioning_cov[i] * m_blocks * sizeof(double);
        total_observations_points_size += gpuData.ldda_locs[i] * sizeof(double);
        total_observations_nearestNeighbors_size += gpuData.ldda_neighbors[i] * sizeof(double);
        total_locs_size_host += m_blocks * sizeof(double) * dim;
        total_locs_nearestNeighbors_size_host += m_nearest_neighbor * sizeof(double) * dim;
        total_locs_size_device += gpuData.ldda_locs[i] * sizeof(double) * dim;
        total_locs_nearestNeighbors_size_device += gpuData.ldda_neighbors[i] * sizeof(double) * dim;
    }

    // Allocate contiguous memory on GPU
    gpuData.total_observations_points_size = total_observations_points_size;
    gpuData.total_observations_neighbors_size = total_observations_nearestNeighbors_size;
    gpuData.total_locs_num_device = total_locs_size_device / sizeof(double) / dim;
    gpuData.total_locs_neighbors_num_device = total_locs_nearestNeighbors_size_device / sizeof(double) / dim;
    
    checkCudaError(cudaMalloc(&gpuData.d_locs_device, total_locs_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_locs_neighbors_device, total_locs_nearestNeighbors_size_device));
    checkCudaError(cudaMalloc(&gpuData.d_observations_device, total_observations_points_size));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_device, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMalloc(&gpuData.d_cov_device, total_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_conditioning_cov_device, total_conditioning_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_cross_cov_device, total_cross_cov_size));
    checkCudaError(cudaMalloc(&gpuData.d_observations_neighbors_copy_device, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMalloc(&gpuData.d_observations_copy_device, total_observations_points_size));
    checkCudaError(cudaMalloc(&gpuData.d_mu_correction_device, total_observations_points_size));
    checkCudaError(cudaMalloc(&gpuData.d_cov_correction_device, total_cov_size));

    // Set device memory to zero
    checkCudaError(cudaMemset(gpuData.d_locs_device, 0, total_locs_size_device));
    checkCudaError(cudaMemset(gpuData.d_locs_neighbors_device, 0, total_locs_nearestNeighbors_size_device));
    checkCudaError(cudaMemset(gpuData.d_observations_device, 0, total_observations_points_size));
    checkCudaError(cudaMemset(gpuData.d_observations_neighbors_device, 0, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMemset(gpuData.d_cov_device, 0, total_cov_size));
    checkCudaError(cudaMemset(gpuData.d_conditioning_cov_device, 0, total_conditioning_cov_size));
    checkCudaError(cudaMemset(gpuData.d_cross_cov_device, 0, total_cross_cov_size));
    checkCudaError(cudaMemset(gpuData.d_observations_neighbors_copy_device, 0, total_observations_nearestNeighbors_size));
    checkCudaError(cudaMemset(gpuData.d_observations_copy_device, 0, total_observations_points_size));
    checkCudaError(cudaMemset(gpuData.d_mu_correction_device, 0, total_observations_points_size));
    checkCudaError(cudaMemset(gpuData.d_cov_correction_device, 0, total_cov_size)); 
    
    // Prepare to store blocks data for coalesced memory access
    double *locs_blocks_data = new double[total_locs_size_host / sizeof(double)];
    double *locs_nearestNeighbors_data = new double[total_locs_nearestNeighbors_size_host / sizeof(double)];

    size_t locs_index = 0;
    size_t locs_nearestNeighbors_index = 0;
    size_t _total_locs_num_host = total_locs_size_host / sizeof(double) / dim;
    size_t _total_locs_nearestNeighbors_num_host = total_locs_nearestNeighbors_size_host / sizeof(double) / dim;
    size_t _total_locs_num_device = total_locs_size_device / sizeof(double) / dim;
    size_t _total_locs_nearestNeighbors_num_device = total_locs_nearestNeighbors_size_device / sizeof(double) / dim;
    
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        int m_blocks = blockInfos[i].blocks.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        // Copy locations (coalesced memory access)
        for (int j = 0; j < m_blocks; ++j) {
            for (int d = 0; d < dim; ++d) {
                locs_blocks_data[locs_index + d * _total_locs_num_host] = blockInfos[i].blocks[j][d];
            }
            locs_index++;
        }
        for (int j = 0; j < m_nearest_neighbor; ++j) {
            for (int d = 0; d < dim; ++d) {
                locs_nearestNeighbors_data[locs_nearestNeighbors_index + d * _total_locs_nearestNeighbors_num_host] = blockInfos[i].nearestNeighbors[j][d];
            }
            locs_nearestNeighbors_index++;
        }
    }

    // Assign pointers to the beginning of each block's memory and copy data
    double *locs_ptr = gpuData.d_locs_device;
    double *locs_nearestNeighbors_ptr = gpuData.d_locs_neighbors_device;
    double *observations_points_ptr = gpuData.d_observations_device;
    double *observations_nearestNeighbors_ptr = gpuData.d_observations_neighbors_device;
    double *cov_ptr = gpuData.d_cov_device;
    double *conditioning_cov_ptr = gpuData.d_conditioning_cov_device;
    double *cross_cov_ptr = gpuData.d_cross_cov_device;
    double *observations_neighbors_copy_ptr = gpuData.d_observations_neighbors_copy_device;
    double *observations_copy_ptr = gpuData.d_observations_copy_device;
    double *mu_correction_ptr = gpuData.d_mu_correction_device;
    double *cov_correction_ptr = gpuData.d_cov_correction_device;

    // Calculate size and GPU pointers array
    size_t index_locs = 0;
    size_t index_locs_nearestNeighbors = 0;
    size_t block_num = blockInfos.size();

    for (size_t i = 0; i < block_num; ++i) {   
        int m_blocks = blockInfos[i].blocks.size();
        int m_nearest_neighbor = blockInfos[i].nearestNeighbors.size();
        
        for (int d = 0; d < dim; ++d) {
            gpuData.h_locs_array[i + block_num * d] = locs_ptr + block_num * d;
            gpuData.h_locs_neighbors_array[i + block_num * d] = locs_nearestNeighbors_ptr + block_num * d;
        }
        
        gpuData.h_observations_array[i] = observations_points_ptr;
        gpuData.h_observations_neighbors_array[i] = observations_nearestNeighbors_ptr;
        gpuData.h_cov_array[i] = cov_ptr;
        gpuData.h_conditioning_cov_array[i] = conditioning_cov_ptr;
        gpuData.h_cross_cov_array[i] = cross_cov_ptr;
        gpuData.h_observations_neighbors_copy_array[i] = observations_neighbors_copy_ptr;
        gpuData.h_observations_copy_array[i] = observations_copy_ptr;
        gpuData.h_mu_correction_array[i] = mu_correction_ptr;
        gpuData.h_cov_correction_array[i] = cov_correction_ptr;
        
        // Copy observations
        checkCudaError(cudaMemcpy(observations_points_ptr, 
                                   blockInfos[i].observations_blocks.data(), 
                                   blockInfos[i].observations_blocks.size() * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(observations_nearestNeighbors_ptr, 
                                   blockInfos[i].observations_nearestNeighbors.data(), 
                                   blockInfos[i].observations_nearestNeighbors.size() * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        
        // Copy locations (coalesced memory access)
        for (int d = 0; d < dim; ++d) {
            checkCudaError(cudaMemcpy(locs_ptr + d * _total_locs_num_device, 
                                   locs_blocks_data + index_locs + d * _total_locs_num_host, 
                                   m_blocks * sizeof(double), 
                                   cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(locs_nearestNeighbors_ptr + d * _total_locs_nearestNeighbors_num_device, 
                                   locs_nearestNeighbors_data + index_locs_nearestNeighbors + d * _total_locs_nearestNeighbors_num_host, 
                                   m_nearest_neighbor * sizeof(double), 
                                   cudaMemcpyHostToDevice));
        }
        
        // Next pointer
        locs_ptr += gpuData.ldda_locs[i];
        locs_nearestNeighbors_ptr += gpuData.ldda_neighbors[i];
        observations_points_ptr += gpuData.ldda_locs[i];
        observations_nearestNeighbors_ptr += gpuData.ldda_neighbors[i];
        cov_ptr += gpuData.ldda_cov[i] * m_blocks;
        conditioning_cov_ptr += gpuData.ldda_conditioning_cov[i] * m_nearest_neighbor;
        cross_cov_ptr += gpuData.ldda_conditioning_cov[i] * m_blocks;
        // Index update
        index_locs += m_blocks;
        index_locs_nearestNeighbors += m_nearest_neighbor;
        // Copy update
        observations_neighbors_copy_ptr += gpuData.ldda_neighbors[i];
        observations_copy_ptr += gpuData.ldda_locs[i];
        mu_correction_ptr += gpuData.ldda_locs[i];
        cov_correction_ptr += gpuData.ldda_cov[i] * m_blocks;
    }

    // Copy data array to the GPU
    checkCudaError(cudaMemcpy(gpuData.d_locs_array, 
               gpuData.h_locs_array, 
               blockInfos.size() * dim * sizeof(double *), 
               cudaMemcpyHostToDevice));   
    checkCudaError(cudaMemcpy(gpuData.d_locs_neighbors_array, 
               gpuData.h_locs_neighbors_array, 
               blockInfos.size() * dim * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_points_array, 
               gpuData.h_observations_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_array, 
               gpuData.h_observations_neighbors_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cov_array, 
               gpuData.h_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_conditioning_cov_array, 
               gpuData.h_conditioning_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cross_cov_array, 
               gpuData.h_cross_cov_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_array, 
               gpuData.h_observations_neighbors_copy_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_copy_array, 
               gpuData.h_observations_copy_array, 
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_mu_correction_array, 
               gpuData.h_mu_correction_array,  
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_cov_correction_array, 
               gpuData.h_cov_correction_array,  
               blockInfos.size() * sizeof(double *), 
               cudaMemcpyHostToDevice));

    // Copy data from host to device
    size_t batchCount = gpuData.ldda_locs.size() - 1;
    // Allocate memory for MAGMA use
    checkMagmaError(magma_imalloc(&gpuData.dinfo_magma, batchCount + 1));
    // Set dinfo_magma to 0
    checkCudaError(cudaMemset(gpuData.dinfo_magma, 0, (batchCount + 1) * sizeof(magma_int_t)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_locs, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_neighbors, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_cov, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_cross_cov, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_ldda_conditioning_cov, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_lda_locs, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_lda_locs_neighbors, (batchCount + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void**)&gpuData.d_const1, (batchCount + 1) * sizeof(int)));

    // Set values
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_locs.data(), 1, gpuData.d_ldda_locs, 1, queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_neighbors.data(), 1, gpuData.d_ldda_neighbors, 1, queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_cov.data(), 1, gpuData.d_ldda_cov, 1, queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_cross_cov.data(), 1, gpuData.d_ldda_cross_cov, 1, queue);
    magma_setvector(batchCount, sizeof(int), gpuData.ldda_conditioning_cov.data(), 1, gpuData.d_ldda_conditioning_cov, 1, queue);
    magma_setvector(batchCount, sizeof(int), gpuData.lda_locs.data(), 1, gpuData.d_lda_locs, 1, queue);
    magma_setvector(batchCount, sizeof(int), gpuData.lda_locs_neighbors.data(), 1, gpuData.d_lda_locs_neighbors, 1, queue);
    magma_setvector(batchCount, sizeof(int), gpuData.h_const1.data(), 1, gpuData.d_const1, 1, queue);

    // Get the max dimensions for the TRSM
    magma_imax_size_2(gpuData.d_lda_locs_neighbors, gpuData.d_lda_locs, batchCount, queue);
    magma_getvector(1, sizeof(magma_int_t), &gpuData.d_lda_locs_neighbors[batchCount], 
                    1, &gpuData.max_m, 1, queue);
    magma_getvector(1, sizeof(magma_int_t), &gpuData.d_lda_locs[batchCount], 
                    1, &gpuData.max_n1, 1, queue);
    magma_imax_size_2(gpuData.d_lda_locs_neighbors, gpuData.d_const1, batchCount, queue);
    magma_getvector(1, sizeof(magma_int_t), &gpuData.d_const1[batchCount], 
                    1, &gpuData.max_n2, 1, queue);
    
    delete[] locs_blocks_data;
    delete[] locs_nearestNeighbors_data;

    return gpuData;
}

// Function to cleanup GPU memory
void cleanupGpuMemory(GpuData &gpuData)
{
    cudaFree(gpuData.d_cov_device);
    cudaFree(gpuData.d_conditioning_cov_device);
    cudaFree(gpuData.d_cross_cov_device);
    cudaFree(gpuData.d_observations_device);
    cudaFree(gpuData.d_observations_neighbors_device);
    cudaFree(gpuData.d_observations_neighbors_copy_device);
    cudaFree(gpuData.d_observations_copy_device);
    cudaFree(gpuData.d_mu_correction_device);
    cudaFree(gpuData.d_cov_correction_device);

    delete[] gpuData.h_cov_array;
    delete[] gpuData.h_conditioning_cov_array;
    delete[] gpuData.h_cross_cov_array;
    delete[] gpuData.h_locs_array;
    delete[] gpuData.h_locs_neighbors_array;
    delete[] gpuData.h_observations_array;
    delete[] gpuData.h_observations_neighbors_array;
    delete[] gpuData.h_observations_copy_array;
    delete[] gpuData.h_mu_correction_array;
    delete[] gpuData.h_cov_correction_array;
    delete[] gpuData.h_observations_neighbors_copy_array;
}
#endif

// Forward declarations for GPU computation functions implemented in scaled_block_kernels.cu
#ifdef USE_CUDA
void compute_covariance_vbatched(
    double **d_locs_A, int *d_lda_A, int inca, size_t total_A,
    double **d_locs_B, int *d_lda_B, int incb, size_t total_B,
    double **d_cov, int *d_ldda, int *d_n,
    size_t batchCount,
    int dim, const std::vector<double> &theta, double *d_range,
    bool add_nugget, cudaStream_t stream, Configurations &opts,
    int max_ldx1, int max_ldx2);

double norm2_batch(int *d_n, double **d_vec, int *d_ldda, size_t batchCount, cudaStream_t stream);
double log_det_batch(int *d_n, double **d_L, int *d_ldda, size_t batchCount, cudaStream_t stream);

// GPU computation function
double performComputationOnGPU(const GpuData &gpuData, const std::vector<double> &theta, 
                               Configurations &aConfigurations, cudaStream_t stream, magma_queue_t queue) {
    int rank = VecchiaHardware::GetMPIRank();
    int gpu_id = VecchiaHardware::GetLocalGPUId();
    int dim = aConfigurations.GetDimensionSize();
    
    // Set the GPU
    checkCudaError(cudaSetDevice(gpu_id));
    
    size_t batchCount = gpuData.ldda_locs.size() - 1;
    magma_int_t *dinfo_magma = gpuData.dinfo_magma;
    int *d_ldda_locs = gpuData.d_ldda_locs;
    int *d_ldda_neighbors = gpuData.d_ldda_neighbors;
    int *d_ldda_cov = gpuData.d_ldda_cov;
    int *d_ldda_cross_cov = gpuData.d_ldda_cross_cov;
    int *d_ldda_conditioning_cov = gpuData.d_ldda_conditioning_cov;
    int *d_lda_locs = gpuData.d_lda_locs;
    int *d_lda_locs_neighbors = gpuData.d_lda_locs_neighbors;
    int *d_const1 = gpuData.d_const1;
    magma_int_t max_m = gpuData.max_m;
    magma_int_t max_n1 = gpuData.max_n1;
    magma_int_t max_n2 = gpuData.max_n2;
    
    // Determine range offset based on kernel type
    // For all kernels in ScaledBlock mode: theta = [sigma2, nugget, range...]
    int range_offset = 2;  // Skip sigma2 and nugget to get to range parameters
    
    // Copy data from device to device (for observations backup)
    checkCudaError(cudaMemcpy(gpuData.d_observations_neighbors_copy_device, 
                               gpuData.d_observations_neighbors_device, 
                               gpuData.total_observations_neighbors_size, 
                               cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(gpuData.d_observations_copy_device, 
                               gpuData.d_observations_device, 
                               gpuData.total_observations_points_size, 
                               cudaMemcpyDeviceToDevice));
    // CRITICAL OPTIMIZATION: Precompute 1/(range²) on CPU so GPU can multiply instead of divide!
    {
        std::vector<double> inv_range2_host(dim);
        for (int i = 0; i < dim; ++i) {
            double r = theta[range_offset + i];
            inv_range2_host[i] = 1.0 / (r * r);
        }
        checkCudaError(cudaMemcpy(gpuData.d_range_device, 
                                   inv_range2_host.data(), 
                                   dim * sizeof(double), 
                                   cudaMemcpyHostToDevice));
    }
    
    // 1. Generate covariance matrices using batched operations
    // CRITICAL OPTIMIZATION: Pass pre-computed max dimensions to avoid expensive thrust::reduce!
    compute_covariance_vbatched(gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cov_array, gpuData.d_ldda_cov, gpuData.d_lda_locs,
                batchCount,
                dim, theta, gpuData.d_range_device, true, stream, aConfigurations,
                max_n1, max_n1);  // main cov: n1 x n1
    compute_covariance_vbatched(gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_array,
                gpuData.d_lda_locs, 1, gpuData.total_locs_num_device,
                gpuData.d_cross_cov_array, gpuData.d_ldda_cross_cov, gpuData.d_lda_locs,
                batchCount,
                dim, theta, gpuData.d_range_device, false, stream, aConfigurations,
                max_m, max_n1);  // cross cov: m x n1
    compute_covariance_vbatched(gpuData.d_locs_neighbors_array,
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_locs_neighbors_array, 
                gpuData.d_lda_locs_neighbors, 1, gpuData.total_locs_neighbors_num_device,
                gpuData.d_conditioning_cov_array, gpuData.d_ldda_conditioning_cov, gpuData.d_lda_locs_neighbors,
                batchCount,
                dim, theta, gpuData.d_range_device, true, stream, aConfigurations,
                max_m, max_m);  // conditioning cov: m x m
    
    // 2. Compute conditioning correction (Schur complement)
    // 2.1 Cholesky factorization of conditioning covariance
    checkMagmaError(magma_dpotrf_vbatched_max_nocheck(
            MagmaLower, d_lda_locs_neighbors,
            gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
            dinfo_magma, batchCount, max_m, queue));
    
    // 2.2 Triangular solve (TRSM)
    magmablas_dtrsm_vbatched_max_nocheck(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                        max_m, max_n1, 
                        d_lda_locs_neighbors, d_lda_locs,
                        1.,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_cross_cov_array, d_ldda_cross_cov,
                        batchCount, queue);
    magmablas_dtrsm_vbatched_max_nocheck(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                        max_m, max_n2, 
                        d_lda_locs_neighbors, d_const1,
                        1.,
                        gpuData.d_conditioning_cov_array, d_ldda_conditioning_cov,
                        gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                        batchCount, queue);
    
    // 2.3 Matrix multiplication (GEMM) for covariance and mean correction
    magmablas_dgemm_vbatched_max_nocheck(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_lda_locs, d_lda_locs_neighbors,
                             1, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                                gpuData.d_cross_cov_array, d_ldda_cross_cov,
                             0, gpuData.d_cov_correction_array, d_ldda_cov,
                             batchCount, 
                             max_n1, max_n1, max_m, 
                             queue);
    magmablas_dgemm_vbatched_max_nocheck(MagmaTrans, MagmaNoTrans,
                             d_lda_locs, d_const1, d_lda_locs_neighbors,
                             1, gpuData.d_cross_cov_array, d_ldda_cross_cov,
                                gpuData.d_observations_neighbors_copy_array, d_ldda_neighbors,
                             0, gpuData.d_mu_correction_array, d_ldda_locs,
                             batchCount, 
                             max_n1, max_n2, max_m,
                             queue);
    
    // 2.4 Compute conditional mean and variance
    for (size_t i = 0; i < batchCount; ++i) {
        // Conditional variance: cov -= cov_correction
        magmablas_dgeadd(gpuData.lda_locs[i], gpuData.lda_locs[i],
                        -1.,
                        gpuData.h_cov_correction_array[i], gpuData.ldda_locs[i], 
                        gpuData.h_cov_array[i], gpuData.ldda_cov[i],
                        queue);
        // Conditional mean: obs -= mu_correction
        magmablas_dgeadd(gpuData.lda_locs[i], 1,
                        -1.,
                        gpuData.h_mu_correction_array[i], gpuData.ldda_locs[i], 
                        gpuData.h_observations_copy_array[i], gpuData.ldda_locs[i],
                        queue);
    }
    
    // 3. Compute log-likelihood
    // 3.1 Cholesky factorization of conditional covariance
    checkMagmaError(magma_dpotrf_vbatched(
            MagmaLower, d_lda_locs,
            gpuData.d_cov_array, d_ldda_cov,
            dinfo_magma, batchCount, queue));
    
    // 3.2 Triangular solve for observations
    magmablas_dtrsm_vbatched(
        MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
        d_lda_locs, d_const1, 1.,
        gpuData.d_cov_array, d_ldda_cov,
        gpuData.d_observations_copy_array, d_ldda_locs,
        batchCount, queue);
    
    // 3.3 Compute norm and determinant
    double norm2_item = norm2_batch(d_lda_locs, gpuData.d_observations_copy_array, d_ldda_locs, batchCount, stream);
    double log_det_item = log_det_batch(d_lda_locs, gpuData.d_cov_array, d_ldda_cov, batchCount, stream);
    
    // 3.4 Compute local log-likelihood
    double log_likelihood = -0.5 * (log_det_item + norm2_item);
    
    // 3.5 MPI reduction to get total log-likelihood
    double log_likelihood_all = 0;
    MPI_Allreduce(&log_likelihood, &log_likelihood_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return log_likelihood_all;
}
#endif

template<typename T>
void ScaledBlockEstimator<T>::InitMemory(Configurations &aConfigurations,
                                          std::unique_ptr<VecchiaGBData<T>> &aData) {
    // No-op for Scaled Block Estimator
    // GPU memory allocation and data transfer happen in Estimate() on first call
}

template<typename T>
T ScaledBlockEstimator<T>::Estimate(Configurations &aConfigurations,
                                     std::unique_ptr<VecchiaGBData<T>> &aData,
                                     const double *apTheta) {
    
#ifdef USE_CUDA
    int rank = VecchiaHardware::GetMPIRank();
    int gpu_id = VecchiaHardware::GetLocalGPUId();
    
    // Static variables for GPU data and initialization state
    static GpuData gpuData;
    static bool gpu_initialized = false;
    static bool warmup_done = false;
    static int call_count = 0;
    static cudaStream_t stream = nullptr;
    static magma_queue_t queue = nullptr;
    
    // Convert theta to vector for GPU functions
    // For ScaledBlock, we need all parameters: sigma2 + nugget + range[dim]
    int dim = aConfigurations.GetDimensionSize();
    int parameters_number = 2 + dim;  // sigma2, nugget, + dim range values
    std::vector<double> theta(apTheta, apTheta + parameters_number);
    
    // Broadcast theta from rank 0 to all processes
    MPI_Bcast(theta.data(), theta.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Initialize GPU data on first call
    if (!gpu_initialized) {
        // Get BlockInfo from VecchiaGBData
        auto& blockInfos = aData->GetBlockInfos();
        
        if (blockInfos.empty()) {
            if (rank == 0) {
                std::cerr << "ERROR: ScaledBlockEstimator::Estimate() - BlockInfo data is empty!" << std::endl;
                std::cerr << "  Make sure clustering was performed before estimation." << std::endl;
            }
            return static_cast<T>(-15000.0);
        }
        
        if (rank == 0) {
            std::cout << "Initializing GPU for Scaled Block Vecchia with " << blockInfos.size() << " blocks" << std::endl;
            
            // Debug: Check for blocks with no nearest neighbors
            int blocks_with_zero_nn = 0;
            int blocks_with_zero_points = 0;
            for (const auto& block : blockInfos) {
                if (block.nearestNeighbors.empty()) {
                    blocks_with_zero_nn++;
                    std::cout << "  WARNING: Block " << block.globalOrder << " has 0 nearest neighbors!" << std::endl;
                }
                if (block.blocks.empty()) {
                    blocks_with_zero_points++;
                    std::cout << "  WARNING: Block " << block.globalOrder << " has 0 points!" << std::endl;
                }
            }
            if (blocks_with_zero_nn > 0) {
                std::cout << "  Total blocks with 0 nearest neighbors: " << blocks_with_zero_nn << std::endl;
            }
            if (blocks_with_zero_points > 0) {
                std::cout << "  Total blocks with 0 points: " << blocks_with_zero_points << std::endl;
            }
        }
        
        // Create MAGMA queue
        magma_queue_create(gpu_id, &queue);
        
        // Get the CUDA stream from the MAGMA queue (they share the same stream)
        stream = magma_queue_get_cuda_stream(queue);

        // Time GPU data copy
        auto start_gpu_copy = std::chrono::high_resolution_clock::now();
        
        // Copy data to GPU
        gpuData = copyDataToGPU(aConfigurations, blockInfos, queue);
        
        // Synchronize to ensure copy is complete before timing
        cudaDeviceSynchronize();
        auto end_gpu_copy = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gpu_copy = end_gpu_copy - start_gpu_copy;
        
        // Store gpu_copy timing
        aData->GetTimingData().gpu_copy = duration_gpu_copy.count();
        
        // Calculate and print total GFLOPS
        double total_gflops = gflopsTotal(gpuData, aConfigurations);
        
        // Store GFLOPS in timing data (first call only)
        aData->GetTimingData().total_gflops = total_gflops;
        
        gpu_initialized = true;
        
        if (rank == 0) {
            std::cout << "GPU initialization complete (copy time: " << duration_gpu_copy.count() << "s)" << std::endl;
        }
    }
    
    // Warmup run (if performance mode is enabled)
    if (aConfigurations.GetIsPerformance() && !warmup_done) {
        MPI_Barrier(MPI_COMM_WORLD);
        performComputationOnGPU(gpuData, theta, aConfigurations, stream, queue);
        MPI_Barrier(MPI_COMM_WORLD);
        warmup_done = true;
        if (rank == 0) {
            std::cout << "Warmup completed" << std::endl;
        }
    }
    
    // Timing for GPU computation
    // Perform GPU computation
    double log_likelihood = performComputationOnGPU(gpuData, theta, aConfigurations, stream, queue);
    call_count++;
    // Print optimization info
    if (rank == 0) {
        std::cout << "Optimization step: " << call_count << ", ";
        std::cout << "f(theta): " << std::fixed << std::setprecision(6) << log_likelihood << ", ";
        std::cout << "Theta: ";
        for (const auto& val : theta) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;
    }

    // Return log-likelihood (positive value for NLOPT maximization)
    return static_cast<T>(log_likelihood);
    
#else
    // Non-CUDA path
    int rank = 0;
#ifdef USE_MPI
    rank = VecchiaHardware::GetMPIRank();
#endif
    
    if (rank == 0) {
        std::cerr << "ERROR: ScaledBlockEstimator requires CUDA support!" << std::endl;
        std::cerr << "  Please recompile with USE_CUDA=ON" << std::endl;
    }
    return static_cast<T>(-0.0);
#endif
}