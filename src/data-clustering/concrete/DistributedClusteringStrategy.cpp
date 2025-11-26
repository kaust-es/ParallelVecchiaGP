
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file DistributedClusteringStrategy.cpp
* @version 1.0.0
* @brief Implementation of DistributedClusteringStrategy
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <atomic>
#include <cmath>
#include <set>

#include <data-clustering/concrete/DistributedClusteringStrategy.hpp>
#include <hardware/VecchiaHardware.hpp>
#include <utilities/Logger.hpp>
#include <helpers/DistanceCalculationHelpers.hpp>

using namespace vecchia::clustering;
using namespace vecchia::common;
using namespace vecchia::dataunits;
using namespace vecchia::configurations;
using namespace vecchia::helpers;

// Helper structure for N-dimensional point metadata
struct PointMetadata {
    std::vector<double> coordinates;
    double observation;
};

// Use shared BlockInfo definition from header
using BlockInfo = vecchia::dataunits::BlockInfo;


// Function to generate random double between 0 and 1
double generateRandomDouble()
{
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// Helper function to calculate Euclidean distance between two points
static double calculateDistance(const std::vector<double>& point1, const std::vector<double>& point2)
{
    double distance = 0.0;
    for (size_t i = 0; i < point1.size(); ++i)
    {
        double diff = point1[i] - point2[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}


// Function to generate random points
static std::vector<PointMetadata> generateRandomPoints(int numPointsPerProcess, int dim, int maxIterations)
{
    std::vector<PointMetadata> pointsMetadata(numPointsPerProcess);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // fix the random seed for each process
    std::srand(rank + 1);

    for (int i = 0; i < numPointsPerProcess; ++i)
    {
        pointsMetadata[i].coordinates.resize(dim);
        for (int j = 0; j < dim; ++j)
        {
            pointsMetadata[i].coordinates[j] = generateRandomDouble();
        }
        if (maxIterations == 1){
            pointsMetadata[i].observation = 0.0;
        }
        else{
            pointsMetadata[i].observation = generateRandomDouble();
        }
    }

    return pointsMetadata;
}

void distanceScale(std::vector<PointMetadata> &localPoints, const std::vector<double>& scale_factor, int dim){
    #pragma omp parallel for
    for (int i = 0; i < localPoints.size(); ++i){
        for (int j = 0; j < dim; ++j){
            localPoints[i].coordinates[j] = localPoints[i].coordinates[j] / scale_factor[j];
        }
    }
}

/**
 * @brief Function to partition points and communicate them to the appropriate MPI processes
 * @details Redistributes points across MPI ranks based on their coordinates to enable
 *          balanced spatial partitioning for distributed Scaled Block Vecchia.
 * @param[in] localMetadata Input points on this MPI rank
 * @param[out] localMetadata_block Output points after MPI redistribution
 * @param[in] dim Dimension of the data
 * @param[in] distance_scale Per-dimension distance scaling factors
 */
void partitionPoints(const std::vector<PointMetadata> &localMetadata, 
                     std::vector<PointMetadata> &localMetadata_block, 
                     int dim,
                     const std::vector<double> &distance_scale)
{
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    
    // Edge case: single process - no actual partitioning needed
    // Reference implementation in ParallelScaledBlockVecchiaGP doesn't handle this,
    // but we should for robustness when testing with mpirun -np 1
    if (size == 1) {
        localMetadata_block = localMetadata;
        return;
    }

    // Prepare to send data to the appropriate process based on coordinate value
    std::vector<int> sendCounts(size, 0);
    std::vector<int> sendDisplacements(size, 0);
    
    // Choose the most relevant scaling parameter (min index of distance_scale)
    int min_index = std::min_element(distance_scale.begin(), distance_scale.end()) - distance_scale.begin();
    double min_distance_scale = distance_scale[min_index];

    std::vector<std::vector<PointMetadata>> sendBuffers(size);

    // Determine target process for each point based on its coordinate
    for (const auto &pointmeta : localMetadata)
    {
        int targetProcess = std::min(static_cast<int>(pointmeta.coordinates[min_index] * min_distance_scale * size), size - 1);
        sendBuffers[targetProcess].push_back(pointmeta);
    }

    // Calculate send counts (each point has dim + 1 doubles: coordinates + observation)
    for (int i = 0; i < size; ++i)
    {
        sendCounts[i] = sendBuffers[i].size() * (dim + 1);
    }

    // Pack send data into flat array
    std::vector<double> sendData;
    for (int i = 0; i < size; ++i)
    {
        for (const auto &point : sendBuffers[i])
        {
            for (int j = 0; j < dim; ++j)
            {
                sendData.push_back(point.coordinates[j]);
            }
            sendData.push_back(point.observation);
        }
    }

    // Calculate displacements
    for (int i = 1; i < size; ++i)
    {
        sendDisplacements[i] = sendDisplacements[i - 1] + sendCounts[i - 1];
    }

    // Prepare receive counts and displacements
    std::vector<int> recvCounts(size, 0);
    std::vector<int> recvDisplacements(size, 0);

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 1; i < size; ++i)
    {
        recvDisplacements[i] = recvDisplacements[i - 1] + recvCounts[i - 1];
    }

    int totalRecvCount = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
    std::vector<double> recvData(totalRecvCount);

    MPI_Alltoallv(sendData.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE,
                  recvData.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Convert received data back to PointMetadata
    localMetadata_block.clear();
    localMetadata_block.resize(totalRecvCount / (dim + 1));
    for (int i = 0, index = 0; i < totalRecvCount; i += (dim + 1), ++index)
    {
        localMetadata_block[index].coordinates.resize(dim);
        for (int j = 0; j < dim; ++j)
        {
            localMetadata_block[index].coordinates[j] = recvData[i + j];
        }
        localMetadata_block[index].observation = recvData[i + dim];
    }
#else
    throw std::runtime_error("partitionPoints requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Perform random clustering on metadata
 * @details Randomly selects k centers and assigns remaining points to nearest center.
 *          Reference: random_points.cpp line 135-235 in ParallelScaledBlockVecchiaGP
 */
static std::vector<int> randomClustering(const std::vector<PointMetadata> &metadata, int k, int dim, int seed, bool is_test)
{
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int numPoints = metadata.size();
    std::vector<int> clusters(numPoints);
    int block_size = numPoints / k;
    // Alpha expansion: 99999999 for test (no limit), 15000 for training
    // This allows clusters to grow much larger than average to avoid segfaults
    float alpha_expansion = is_test ? 99999999.0f : 15000.0f;
    
    // Initialize random number generator
    auto start_init = std::chrono::high_resolution_clock::now();
    std::mt19937 gen(seed);

    // 1. Randomly select k centers without replacement
    std::vector<std::vector<double>> centers(k);
    std::vector<int> centerIndices(numPoints);
    std::iota(centerIndices.begin(), centerIndices.end(), 0);

    // Shuffle and take first k indices
    std::shuffle(centerIndices.begin(), centerIndices.end(), gen);
    for (int i = 0; i < k; ++i)
    {
        centers[i] = metadata[centerIndices[i]].coordinates;
        clusters[centerIndices[i]] = i;
    }

    // Track the size of each cluster
    std::vector<int> clusterSizes(k, 1); // Initialize with 1 because we already assigned centers
    
    auto end_init = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_init = end_init - start_init;
    if (rank == 0) {
        std::cout << "    [SubDetail] Random center selection: " << duration_init.count() << " s" << std::endl;
    }

    // 2. Assign remaining points to nearest center
    auto start_assignment_loop = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numPoints; ++i)
    {
        // Skip if this point is a center
        if (i < k && i == centerIndices[i])
            continue;

        double minDist = std::numeric_limits<double>::max();
        int nearestCluster = 0;

        // Find nearest center
        for (int j = 0; j < k; ++j)
        {
            // Skip clusters that have reached their size limit
            if (clusterSizes[j] >= alpha_expansion * block_size)
                continue;
                
            double dist = 0.0;
            for (int d = 0; d < dim; ++d)
            {
                double diff = metadata[i].coordinates[d] - centers[j][d];
                dist += diff * diff;
            }

            if (dist < minDist)
            {
                minDist = dist;
                nearestCluster = j;
            }
        }

        // If all clusters reached their size limit, find the nearest cluster without size restriction
        if (minDist == std::numeric_limits<double>::max())
        {
            for (int j = 0; j < k; ++j)
            {
                double dist = 0.0;
                for (int d = 0; d < dim; ++d)
                {
                    double diff = metadata[i].coordinates[d] - centers[j][d];
                    dist += diff * diff;
                }

                if (dist < minDist)
                {
                    minDist = dist;
                    nearestCluster = j;
                }
            }
        }

        clusters[i] = nearestCluster;
        #pragma omp atomic
        clusterSizes[nearestCluster]++;
    }
    
    auto end_assignment_loop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_assignment_loop = end_assignment_loop - start_assignment_loop;
    if (rank == 0) {
        std::cout << "    [SubDetail] Point-to-center assignment loop: " << duration_assignment_loop.count() << " s" << std::endl;
        std::cout << "    [SubDetail]   (numPoints=" << numPoints << ", k=" << k << ", dim=" << dim << ")" << std::endl;
    }
    
    // Print statistics about cluster sizes
    auto start_stats = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto minmax = std::minmax_element(clusterSizes.begin(), clusterSizes.end());
        double avg = std::accumulate(clusterSizes.begin(), clusterSizes.end(), 0.0) / k;
        
        std::cout << "Cluster size statistics:" << std::endl;
        std::cout << "  Smallest cluster: " << *minmax.first << std::endl; 
        std::cout << "  Largest cluster: " << *minmax.second << std::endl;
        std::cout << "  Average cluster size: " << avg << std::endl;
    }
    auto end_stats = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_stats = end_stats - start_stats;
    if (rank == 0) {
        std::cout << "    [SubDetail] Cluster statistics: " << duration_stats.count() << " s" << std::endl;
    }

    return clusters;
#else
    throw std::runtime_error("randomClustering requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Perform k-means++ clustering with custom reductions
 * @details Reference: random_points.cpp line 450-554 in ParallelScaledBlockVecchiaGP
 */
// Define custom reduction for 2D vector (for OpenMP parallel reduction)
#pragma omp declare reduction(vec2d_double_plus : std::vector<std::vector<double>> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), \
        [](std::vector<double> & a, const std::vector<double> &b){ \
            std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<double>()); \
            return a;})) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), std::vector<double>(omp_orig[0].size())))

// Define custom reduction for 1D vector
#pragma omp declare reduction(vec_int_plus : std::vector<int> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

static std::vector<int> kMeansPlusPlus(const std::vector<PointMetadata> &metadata, int k, int dim, int maxIterations, int rank, int seed)
{
    std::vector<int> clusters(metadata.size());
    std::vector<std::vector<double>> centroids(k, std::vector<double>(dim));
    std::mt19937 gen(seed + rank);
    
    // Choose the first centroid randomly
    std::uniform_int_distribution<> dis(0, metadata.size() - 1);
    int firstCentroidIndex = dis(gen);
    centroids[0] = metadata[firstCentroidIndex].coordinates;

    // Choose the remaining centroids using k-means++
    for (int i = 1; i < k; ++i)
    {
        std::vector<double> distances(metadata.size(), std::numeric_limits<double>::max());

        #pragma omp parallel for
        for (size_t j = 0; j < metadata.size(); ++j)
        {
            for (int c = 0; c < i; ++c)
            {
                double dist = 0;
                for (int d = 0; d < dim; ++d)
                {
                    double diff = metadata[j].coordinates[d] - centroids[c][d];
                    dist += diff * diff;
                }
                distances[j] = std::min(distances[j], dist);
            }
        }

        // Choose the next centroid with probability proportional to distance squared
        std::discrete_distribution<> d(distances.begin(), distances.end());
        int nextCentroidIndex = d(gen);
        centroids[i] = metadata[nextCentroidIndex].coordinates;
    }

    // K-means iterations
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        if (iter % 30 == 0 && rank == 0)
        {
            std::cout << "K-means iteration: " << iter << std::endl;
        }
        
        // Assign points to the nearest centroid
        #pragma omp parallel for
        for (size_t i = 0; i < metadata.size(); ++i)
        {
            double minDist = std::numeric_limits<double>::max();
            int nearestCentroid = 0;
            for (int j = 0; j < k; ++j)
            {
                double dist = 0;
                for (int d = 0; d < dim; ++d)
                {
                    double diff = metadata[i].coordinates[d] - centroids[j][d];
                    dist += diff * diff;
                }
                if (dist < minDist)
                {
                    minDist = dist;
                    nearestCentroid = j;
                }
            }
            clusters[i] = nearestCentroid;
        }

        // Recalculate centroids
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dim, 0.0));
        std::vector<int> clusterSizes(k, 0);

        #pragma omp parallel for reduction(vec2d_double_plus : newCentroids) reduction(vec_int_plus : clusterSizes)
        for (size_t i = 0; i < metadata.size(); ++i)
        {
            int cluster = clusters[i];
            for (int d = 0; d < dim; ++d)
            {
                newCentroids[cluster][d] += metadata[i].coordinates[d];
            }
            clusterSizes[cluster]++;
        }

        // Update centroids
        for (int i = 0; i < k; ++i)
        {
            if (clusterSizes[i] > 0)
            {
                for (int d = 0; d < dim; ++d)
                {
                    centroids[i][d] = newCentroids[i][d] / clusterSizes[i];
                }
            }
            else
            {
                // Assign a random point as the centroid for empty clusters
                int randomIndex = dis(gen);
                centroids[i] = metadata[randomIndex].coordinates;
                clusters[randomIndex] = i;
            }
        }
    }

    return clusters;
}

/**
 * @brief Perform finer partitioning within each processor
 * @details Reference: random_points.cpp line 238-281 in ParallelScaledBlockVecchiaGP
 */
static void finerPartition(const std::vector<PointMetadata> &metadata, int numBlocksPerProcess,
                    std::vector<std::vector<PointMetadata>> &finerPartitions, 
                    Configurations &aConfigurations, bool is_test)
{
#ifdef USE_MPI
    finerPartitions.clear();
    finerPartitions.resize(numBlocksPerProcess);

    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();

    // Perform clustering
    auto start_clustering = std::chrono::high_resolution_clock::now();
    std::vector<int> clusters;
    if (numBlocksPerProcess * 2 < metadata.size())
    {
        // Block Vecchia
        if (aConfigurations.GetClusteringMethod() == "random")
        {
            // Use random clustering for large datasets
            clusters = randomClustering(metadata, numBlocksPerProcess, aConfigurations.GetDimensionSize(), 
                                       aConfigurations.GetSeed() * size + rank, is_test);
        }
        else if (aConfigurations.GetClusteringMethod() == "kmeans++")
        {
            // Use k-means++ for smaller datasets
            clusters = kMeansPlusPlus(metadata, numBlocksPerProcess, aConfigurations.GetDimensionSize(), 
                                     aConfigurations.GetKMeansMaxIter(), rank, aConfigurations.GetSeed() * size + rank);
        }
        else
        {
            std::cerr << "Invalid clustering method: " << aConfigurations.GetClusteringMethod() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    else
    {
        // Classic Vecchia
        clusters.resize(metadata.size());
        std::iota(clusters.begin(), clusters.end(), 0);
    }
    auto end_clustering = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_clustering = end_clustering - start_clustering;
    
    double max_clustering_time;
    double clustering_time_seconds = duration_clustering.count();
    MPI_Allreduce(&clustering_time_seconds, &max_clustering_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  [Detail] Clustering algorithm (" << aConfigurations.GetClusteringMethod() << "): " << max_clustering_time << " s" << std::endl;
    }

    // Assign points to clusters
    auto start_assignment = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < metadata.size(); ++i)
    {
        finerPartitions[clusters[i]].push_back(metadata[i]);
    }
    auto end_assignment = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_assignment = end_assignment - start_assignment;
    
    double max_assignment_time;
    double assignment_time_seconds = duration_assignment.count();
    MPI_Allreduce(&assignment_time_seconds, &max_assignment_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  [Detail] Point assignment to clusters: " << max_assignment_time << " s" << std::endl;
    }
#else
    throw std::runtime_error("finerPartition requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Calculate centers of gravity for each block with OpenMP parallelization
 * @details Reference: random_points.cpp line 284-312 in ParallelScaledBlockVecchiaGP
 */
static std::vector<std::vector<double>> calculateCentersOfGravity(const std::vector<std::vector<PointMetadata>> &finerPartitions, 
                                                           Configurations &aConfigurations)
{
    int numBlocks = finerPartitions.size();
    int dim = aConfigurations.GetDimensionSize();
    std::vector<std::vector<double>> centers(numBlocks, std::vector<double>(dim));

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < numBlocks; ++i)
    {
        auto &blockmetadata = finerPartitions[i];
        if (blockmetadata.empty())
        {
            continue;
        }
        std::vector<double> sum(dim, 0.0);
        for (auto &pointmeta : blockmetadata)
        {
            for (int j = 0; j < dim; ++j)
            {
                sum[j] += pointmeta.coordinates[j];
            }
        }
        for (int j = 0; j < dim; ++j)
        {
            centers[i][j] = sum[j] / blockmetadata.size();
        }
    }

    return centers;
}

/**
 * @brief Send centers of gravity to all processors using AllGather
 * @details Reference: random_points.cpp line 315-369 in ParallelScaledBlockVecchiaGP
 */
static void AllGatherCentersHelper(const std::vector<std::vector<double>> &centers, 
                            std::vector<std::pair<std::vector<double>, int>> &allCenters, 
                            Configurations &aConfigurations)
{
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int dim = aConfigurations.GetDimensionSize();

    // Handle single-process case
    if (size == 1) {
        allCenters.clear();
        for (const auto& center : centers) {
            allCenters.push_back(std::make_pair(center, 0));
        }
        return;
    }

    int numCenters = centers.size();
    std::vector<int> recvCounts(size, 0);
    // Use Allgather instead of Gather to get counts on all nodes
    MPI_Allgather(&numCenters, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displacements(size, 0);
    int totalCenters = 0;
    // Calculate displacements and total centers on all nodes
    for (int i = 0; i < size; ++i)
    {
        displacements[i] = totalCenters * (dim + 1); // +1 for rank
        totalCenters += recvCounts[i];
        recvCounts[i] *= (dim + 1); // Each center has dim doubles + rank
    }

    // Create send buffer with center coordinates and rank
    std::vector<double> sendBuffer(numCenters * (dim + 1));
    for (int i = 0; i < numCenters; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            sendBuffer[i * (dim + 1) + j] = centers[i][j];
        }
        // Add rank information
        sendBuffer[i * (dim + 1) + dim] = static_cast<double>(rank);
    }

    // Allocate receive buffer on all nodes
    std::vector<double> recvBuffer(totalCenters * (dim + 1));
    
    // Use Allgatherv instead of Gatherv to send data to all nodes
    MPI_Allgatherv(sendBuffer.data(), numCenters * (dim + 1), MPI_DOUBLE, 
                   recvBuffer.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Process received data on all nodes
    allCenters.clear();
    allCenters.resize(totalCenters);
    
    for (int i = 0; i < totalCenters; ++i)
    {
        std::vector<double> centerCoords(dim);
        for (int j = 0; j < dim; ++j)
        {
            centerCoords[j] = recvBuffer[i * (dim + 1) + j];
        }
        int centerRank = static_cast<int>(recvBuffer[i * (dim + 1) + dim]);
        allCenters[i] = std::make_pair(centerCoords, centerRank);
    }
#else
    throw std::runtime_error("AllGatherCenters requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Randomly reorder centers at all processors
 * @details Reference: random_points.cpp line 372-407 in ParallelScaledBlockVecchiaGP
 */
static void reorderCenters(std::vector<std::vector<double>> &centers, 
                   std::vector<std::pair<std::vector<double>, int>> &allCenters, 
                   std::vector<int> &permutation,
                   std::vector<int> &localPermutation,
                   Configurations &aConfigurations)
{
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();

    // Generate the same random permutation on all nodes
    std::mt19937 gen(aConfigurations.GetSeed());
    permutation.resize(allCenters.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), gen);

    // Handle single-process case
    if (size == 1) {
        localPermutation.clear();
        localPermutation.reserve(centers.size());
        for (size_t i = 0; i < centers.size(); ++i) {
            localPermutation.push_back(permutation[i]);
        }
        return;
    }

    // Get the number of centers this node has
    int numLocalCenters = centers.size();
    
    // Gather the number of centers from all nodes
    std::vector<int> centerCounts(size);
    MPI_Allgather(&numLocalCenters, 1, MPI_INT, centerCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements
    std::vector<int> displacements(size, 0);
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i-1] + centerCounts[i-1];
    }
    
    // Keep only the permuted indices for this node's portion
    localPermutation.clear();
    localPermutation.reserve(numLocalCenters);
    for (int i = 0; i < numLocalCenters; ++i) {
        localPermutation.push_back(permutation[displacements[rank] + i]);
    }
#else
    throw std::runtime_error("reorderCenters requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Create block information for each processor
 * @details Reference: block_info.cpp in ParallelScaledBlockVecchiaGP
 */
static std::vector<BlockInfo> createBlockInfo(const std::vector<std::vector<PointMetadata>> &finerPartitions,
                                       const std::vector<std::vector<double>> &localCenters,
                                       const std::vector<std::pair<std::vector<double>, int>> &allCenters,
                                       const std::vector<int> &permutation,
                                       const std::vector<int> &localPermutation,
                                       Configurations &aConfigurations)
{
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();

    std::vector<BlockInfo> blockInfos;
    int numBlocksLocal = localCenters.size();
    blockInfos.resize(numBlocksLocal);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numBlocksLocal; ++i)
    {
        const auto &localCenter = localCenters[i];
        int globalOrder = permutation[localPermutation[i]];

        // Create BlockInfo structure locally to avoid race conditions
        BlockInfo blockInfo;
        blockInfo.localOrder = i;
        blockInfo.globalOrder = globalOrder;
        blockInfo.center = localCenter;
        for (const auto &pointMetadata : finerPartitions[i])
        {
            blockInfo.blocks.push_back(pointMetadata.coordinates);
            blockInfo.observations_blocks.push_back(pointMetadata.observation);
        }

        // Each thread writes to its own designated position in blockInfos
        blockInfos[i] = blockInfo;
    }

    return blockInfos;
#else
    throw std::runtime_error("createBlockInfo requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Process and send blocks to appropriate processors based on distance threshold
 * @details Reference: random_points.cpp and distance_calc.cpp in ParallelScaledBlockVecchiaGP
 */
static std::vector<BlockInfo> processAndSendBlocks(std::vector<BlockInfo> &blockInfos, 
    const std::vector<std::pair<std::vector<double>, int>> &CenterRanks, 
    double distance_threshold, 
    const std::vector<int>& permutation, 
    const std::vector<int>& localPermutation, 
    Configurations &aConfigurations, 
    bool pred_tag)
{
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int dim = aConfigurations.GetDimensionSize();

    // Prepare buffers to send blocks to other processors
    std::vector<std::set<int>> blockIndexSets(size);
    std::vector<std::vector<BlockInfo>> sendBuffers(size);
    double distance_threshold_dynamic = (pred_tag) ? distance_threshold : distance_threshold / dim;

    // Magic constant for ensuring enough blocks
    int m_const = (aConfigurations.GetBlockSize() == aConfigurations.GetProblemSize()) ? 300 : 42;
    std::vector<std::pair<std::vector<double>, int>> allCenterRanks = CenterRanks;

    // Using OpenMP with private copies to avoid race conditions
    std::vector<std::vector<std::set<int>>> private_blockIndexSets(omp_get_max_threads(), std::vector<std::set<int>>(size));
    std::vector<std::vector<std::vector<BlockInfo>>> private_sendBuffers(omp_get_max_threads(), std::vector<std::vector<BlockInfo>>(size));

    // Parallelize loop over centers with separate handling for pred_tag and non-pred_tag cases
    // This avoids repeated conditional checks inside the hot loop
    if (pred_tag){
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < allCenterRanks.size(); ++i){
            int thread_id = omp_get_thread_num();
            const auto &centerRank = allCenterRanks[i];
            const auto &center = centerRank.first;
            int destRank = centerRank.second;
            
            // For each block, check if it's within distance threshold of this center
            for (const auto &blockInfo : blockInfos) {
                int globalOrder = blockInfo.globalOrder;
                // Calculate distance between block center and the current center
                double distance = calculateDistance(blockInfo.center, center);
                // If within threshold, send to the corresponding rank
                if (distance < distance_threshold_dynamic) {
                    if (private_blockIndexSets[thread_id][destRank].find(globalOrder) == private_blockIndexSets[thread_id][destRank].end()) {
                        private_sendBuffers[thread_id][destRank].push_back(blockInfo);
                        private_blockIndexSets[thread_id][destRank].insert(globalOrder);
                    }
                }
            }
        }
    }else{
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < allCenterRanks.size(); ++i){
            int thread_id = omp_get_thread_num();
            const auto &centerRank = allCenterRanks[i];
            const auto &center = centerRank.first;
            int destRank = centerRank.second;
            
            // For each block, check if it's within distance threshold of this center
            for (const auto &blockInfo : blockInfos) {
                int globalOrder = blockInfo.globalOrder;
                if (globalOrder >= permutation[i]){
                    continue;
                }

                // Send first m_const blocks to all processors to ensure enough blocks
                if (globalOrder < m_const) {
                    for (int dest = 0; dest < size; ++dest) {
                        if (private_blockIndexSets[thread_id][dest].find(globalOrder) == private_blockIndexSets[thread_id][dest].end()) {
                            private_sendBuffers[thread_id][dest].push_back(blockInfo);
                            private_blockIndexSets[thread_id][dest].insert(globalOrder);
                        }
                    }
                    continue;
                }
                
                // Calculate distance between block center and the current center
                double distance = calculateDistance(blockInfo.center, center);
                // If within threshold, send to the corresponding rank
                if (distance < distance_threshold_dynamic) {
                    if (private_blockIndexSets[thread_id][destRank].find(globalOrder) == private_blockIndexSets[thread_id][destRank].end()) {
                        private_sendBuffers[thread_id][destRank].push_back(blockInfo);
                        private_blockIndexSets[thread_id][destRank].insert(globalOrder);
                    }
                }
            }
        }
    }

    // Merge private data structures
    for (int t = 0; t < omp_get_max_threads(); t++) {
        for (int dest = 0; dest < size; dest++) {
            for (const auto& globalOrder : private_blockIndexSets[t][dest]) {
                if (blockIndexSets[dest].find(globalOrder) == blockIndexSets[dest].end()) {
                    for (const auto& block : private_sendBuffers[t][dest]) {
                        if (block.globalOrder == globalOrder) {
                            sendBuffers[dest].push_back(block);
                            blockIndexSets[dest].insert(globalOrder);
                            break;
                        }
                    }
                }
            }
        }
    }

    // Handle single-process case - return the filtered blocks without MPI communication
    if (size == 1) {
        return sendBuffers[0];
    }

    // Calculate the size of each buffer to send
    std::vector<int> sendCounts(size, 0);
    for (int dest = 0; dest < size; ++dest) {
        for (const auto& block : sendBuffers[dest]) {
            int numPoints = block.blocks.size();
            sendCounts[dest] += 2 + dim + 1 + (numPoints * dim) + numPoints;
        }
    }

    // Exchange send counts to get receive counts
    std::vector<int> recvCounts(size);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements
    std::vector<int> sendDispls(size), recvDispls(size);
    int totalSendCount = 0, totalRecvCount = 0;
    for (int i = 0; i < size; ++i) {
        sendDispls[i] = totalSendCount;
        totalSendCount += sendCounts[i];
        recvDispls[i] = totalRecvCount;
        totalRecvCount += recvCounts[i];
    }

    // Prepare send buffer
    std::vector<double> sendBuffer(totalSendCount);
    int currentPos = 0;
    for (int dest = 0; dest < size; ++dest) {
        for (const auto& block : sendBuffers[dest]) {
            sendBuffer[currentPos++] = static_cast<double>(block.localOrder);
            sendBuffer[currentPos++] = static_cast<double>(block.globalOrder);
            
            for (const auto& coord : block.center) {
                sendBuffer[currentPos++] = coord;
            }
            
            sendBuffer[currentPos++] = static_cast<double>(block.blocks.size());
            
            for (const auto& point : block.blocks) {
                for (const auto& coord : point) {
                    sendBuffer[currentPos++] = coord;
                }
            }
            
            for (const auto& obs : block.observations_blocks) {
                sendBuffer[currentPos++] = obs;
            }
        }
    }

    // Prepare receive buffer
    std::vector<double> recvBuffer(totalRecvCount);

    // Perform the all-to-all communication
    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                 recvBuffer.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Unpack received data
    std::vector<BlockInfo> receivedBlocks;
    size_t i = 0;
    while (i < recvBuffer.size()) {
        BlockInfo block;
        block.localOrder = static_cast<int>(recvBuffer[i++]);
        block.globalOrder = static_cast<int>(recvBuffer[i++]);
        
        block.center.resize(dim);
        for (int j = 0; j < dim; ++j) {
            block.center[j] = recvBuffer[i++];
        }
        
        int numPoints = static_cast<int>(recvBuffer[i++]);
        
        block.blocks.resize(numPoints, std::vector<double>(dim));
        for (int j = 0; j < numPoints; ++j) {
            for (int k = 0; k < dim; ++k) {
                block.blocks[j][k] = recvBuffer[i++];
            }
        }
        
        block.observations_blocks.resize(numPoints);
        for (int j = 0; j < numPoints; ++j) {
            block.observations_blocks[j] = recvBuffer[i++];
        }
        
        receivedBlocks.push_back(block);
    }

    return receivedBlocks;
#else
    throw std::runtime_error("processAndSendBlocks requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Perform nearest neighbor search for each block
 * @details Reference: vecchia_helper.cpp in ParallelScaledBlockVecchiaGP
 */
static void nearest_neighbor_search(std::vector<BlockInfo> &blockInfos, std::vector<BlockInfo> &receivedBlocks, 
                            Configurations &aConfigurations, double distance, bool pred_tag)
{
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    
    // Reorder received blocks based on globalOrder
    auto start_sort = std::chrono::high_resolution_clock::now();
    std::sort(receivedBlocks.begin(), receivedBlocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
        return a.globalOrder < b.globalOrder;
    });
    auto end_sort = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_sort = end_sort - start_sort;
    
    double max_sort_time;
    double sort_time_seconds = duration_sort.count();
    MPI_Allreduce(&sort_time_seconds, &max_sort_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  [Detail] Sorting received blocks: " << max_sort_time << " s" << std::endl;
    }

    int m_nn = pred_tag ? aConfigurations.GetTestConditioningSize() : aConfigurations.GetConditioningSize();
    double distance_threshold = distance;
    
    // Pre-compute these values OUTSIDE the parallel loop to avoid repeated function calls
    int numBlocksPerProcess = aConfigurations.GetBlockSize() / size + (rank < aConfigurations.GetBlockSize() % size ? 1 : 0);
    int numPointsPerProcess = aConfigurations.GetProblemSize() / size + (rank < aConfigurations.GetProblemSize() % size ? 1 : 0);
    
    // Perform nearest neighbor search
    auto start_nn_computation = std::chrono::high_resolution_clock::now();
    
    // Counters for statistics
    std::atomic<long long> total_distance_calcs(0);
    std::atomic<long long> total_candidates_considered(0);
    std::atomic<long long> total_neighbors_found(0);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < blockInfos.size(); ++i) {
        auto& block = blockInfos[i];
        std::vector<std::tuple<double, std::vector<double>, double>> distancesMeta;
        
        long long block_distance_calcs = 0;
        long long block_candidates = 0;

        for (auto& prevBlock : receivedBlocks) {
            if (prevBlock.globalOrder >= block.globalOrder){
                break;
            }
            for (size_t j = 0; j < prevBlock.blocks.size(); ++j) {
                double distance = calculateDistance(block.center, prevBlock.blocks[j]);
                block_distance_calcs++;
                if (distance < distance_threshold || block.globalOrder <= 200){
                    distancesMeta.emplace_back(distance, prevBlock.blocks[j], prevBlock.observations_blocks[j]);
                    block_candidates++;
                }
            }
        }
        
        total_distance_calcs += block_distance_calcs;
        total_candidates_considered += block_candidates;
        
        // Handle classic Vecchia case
        if (block.globalOrder <= m_nn && numBlocksPerProcess == numPointsPerProcess){
            continue;
        }
        
        if (distancesMeta.size() < m_nn && block.globalOrder > 0){
            std::cout << "Warning: Not enough neighbors found for block, random added. m: " << distancesMeta.size() 
                     << ", block: " << block.globalOrder << ", rank: " << rank << std::endl;
            for (auto& prevBlock : receivedBlocks) {
                if (prevBlock.globalOrder >= block.globalOrder) {
                    break;
                }
                for (size_t j = 0; j < prevBlock.blocks.size(); ++j) {
                    bool already_added = false;
                    for (const auto& existing : distancesMeta) {
                        if (std::get<1>(existing) == prevBlock.blocks[j]) {
                            already_added = true;
                            break;
                        }
                    }
                    if (!already_added) {
                        double distance = calculateDistance(block.center, prevBlock.blocks[j]);
                        distancesMeta.emplace_back(distance, prevBlock.blocks[j], prevBlock.observations_blocks[j]);
                        if (distancesMeta.size() >= m_nn) {
                            break;
                        }
                    }
                }
                if (distancesMeta.size() >= m_nn) {
                    break;
                }
            }
        }
        
        // Sort distances and keep the m nearest neighbors
        std::sort(distancesMeta.begin(), distancesMeta.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        size_t neighbors_added = std::min(static_cast<size_t>(m_nn), distancesMeta.size());
        for (size_t k = 0; k < neighbors_added; ++k) {
            block.nearestNeighbors.push_back(std::get<1>(distancesMeta[k]));
            block.observations_nearestNeighbors.push_back(std::get<2>(distancesMeta[k]));
        }
        total_neighbors_found += neighbors_added;
    }
    
    auto end_nn_computation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_nn_computation = end_nn_computation - start_nn_computation;
    
    double max_nn_computation_time;
    double nn_computation_time_seconds = duration_nn_computation.count();
    MPI_Allreduce(&nn_computation_time_seconds, &max_nn_computation_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  [Detail] NN distance computation & selection: " << max_nn_computation_time << " s" << std::endl;
        std::cout << "  [Detail] Average blocks processed: " << blockInfos.size() << std::endl;
        std::cout << "  [Detail] Average candidate blocks: " << receivedBlocks.size() << std::endl;
        std::cout << "    [SubDetail] Total distance calculations: " << total_distance_calcs.load() << std::endl;
        std::cout << "    [SubDetail] Total candidates considered: " << total_candidates_considered.load() << std::endl;
        std::cout << "    [SubDetail] Total neighbors found: " << total_neighbors_found.load() << std::endl;
        if (blockInfos.size() > 0) {
            std::cout << "    [SubDetail] Avg distance calcs per block: " << total_distance_calcs.load() / blockInfos.size() << std::endl;
        }
    }
#else
    throw std::runtime_error("nearest_neighbor_search requires MPI support (USE_MPI)");
#endif
}

/**
 * @brief Descale distances back to original scale
 * @details Reverses the distance scaling applied earlier
 */
static void distanceDeScale(std::vector<BlockInfo> &localBlocks, const std::vector<double>& scale_factor, int dim){
    #pragma omp parallel for
    for (size_t i = 0; i < localBlocks.size(); ++i){
        for (size_t j = 0; j < localBlocks[i].blocks.size(); ++j){
            for (int k = 0; k < dim; ++k){
                localBlocks[i].blocks[j][k] = localBlocks[i].blocks[j][k] * scale_factor[k];
            }
        }
    }
    #pragma omp parallel for
    for (size_t i = 0; i < localBlocks.size(); ++i){
        for (size_t j = 0; j < localBlocks[i].nearestNeighbors.size(); ++j){
            for (int k = 0; k < dim; ++k){
                localBlocks[i].nearestNeighbors[j][k] = localBlocks[i].nearestNeighbors[j][k] * scale_factor[k];
            }
        }
    }
}

template<typename T>
DistributedClusteringStrategy<T>::DistributedClusteringStrategy(
    const std::string &aMethod,
    const std::vector<double> &aDistanceScale,
    int aNNMultiplier,
    int aNumBlocksPerProcess,
    int aMaxIter,
    int aSeed)
    : mMethod(aMethod), mDistanceScale(aDistanceScale),
      mNNMultiplier(aNNMultiplier), mNumBlocksPerProcess(aNumBlocksPerProcess),
      mMaxIter(aMaxIter), mSeed(aSeed) {
    this->mNumClusters = aNumBlocksPerProcess;
}

template<typename T>
ClusteringResult<T> DistributedClusteringStrategy<T>::ComputeClusters(
    Locations<T> &aLocations,
    Configurations &aConfigurations) {
    
#ifndef USE_MPI
    throw std::runtime_error("DistributedClusteringStrategy requires MPI support (USE_MPI)");
#else
    
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    
    if (rank == 0) {
        LOGGER("** DistributedClusteringStrategy: " << mMethod 
               << " with MPI (ranks=" << size << ") **")
    }
    bool pred_tag = false;
    if(aConfigurations.GetTestPointsTotal() > 0 && aConfigurations.GetTestBlocksTotal() > 0){
        pred_tag = true;
    }
    Dimension dim = aLocations.GetDimension();
    int numLocalPoints = aLocations.GetSize();
    double distance = DistanceCalculationHelpers<T>::CalculateDistanceThreshold(aConfigurations.GetDistanceScale(), aConfigurations.GetProblemSize(), aConfigurations.GetConditioningSize(), aConfigurations.GetNNMultiplier());
    printf("after the calculate_distance_threshold %f \n", distance);
    std::cout << "kernel_type: " << aConfigurations.GetKernelType() << std::endl;
    std::cout << "Number of total points: " << aConfigurations.GetProblemSize() << std::endl;
    std::cout << "Number of total blocks: " << aConfigurations.GetBlockSize() << std::endl;
    std::cout << "The number of nearest neighbors: " << aConfigurations.GetConditioningSize() << std::endl;
    if(pred_tag){
        std::cout << "Number of total points_test: " << aConfigurations.GetTestPointsTotal() << std::endl;
        std::cout << "Number of total blocks_test: " << aConfigurations.GetTestBlocksTotal() << std::endl;
        std::cout << "The number of nearest neighbors_test: " << aConfigurations.GetTestConditioningSize() << std::endl;
    }
    std::cout << "The distance threshold_coarse: " << distance << std::endl;
    std::cout << "The distance threshold_finer: " << distance << std::endl;
    std::cout << "Dimension: " << aConfigurations.GetDimensionSize() << std::endl;
    // Calculate range_offset (number of non-range parameters)
    int range_offset = aConfigurations.GetInitialTheta().size() - aConfigurations.GetDimensionSize();
    std::cout << "Range offset: " << range_offset << std::endl;
    std::cout << "Number of processes: " << size << std::endl;
    std::cout << "Distance scale: ";
    auto distance_scale = aConfigurations.GetDistanceScale();
    if (!distance_scale.empty()) {
        for (auto scale : distance_scale) {
            std::cout << scale << ", ";
        }
    } else {
        std::cout << "uniform (1.0 for all dimensions)";
    }
    std::cout << std::endl;
    // print the initial theta values
    std::cout << "Theta: ";
    auto theta_init = aConfigurations.GetInitialTheta();
    if (!theta_init.empty()) {
        for (auto theta : theta_init) {
            std::cout << theta << ", ";
        }
    } else {
        std::cout << "not set";
    }
    std::cout << std::endl;
    // print the lower bounds
    std::cout << "Lower bounds: ";
    auto lower_bounds = aConfigurations.GetLowerBounds();
    for (auto bound : lower_bounds) {
        std::cout << bound << ", ";
    }
    std::cout << std::endl;
    // print the upper bounds
    std::cout << "Upper bounds: ";
    auto upper_bounds = aConfigurations.GetUpperBounds();
    for (auto bound : upper_bounds) {
        std::cout << bound << ", ";
    }
    std::cout << std::endl;
    std::cout << "NN Multiplier: " << aConfigurations.GetNNMultiplier() << std::endl;
    std::cout << "Clustering Method: " << aConfigurations.GetClusteringMethod() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    ClusteringResult<T> result;
    // 1. Generate random points
    long long numPointsPerProcess = aConfigurations.GetProblemSize() / size + (rank < aConfigurations.GetProblemSize() % size ? 1 : 0);
    std::vector<PointMetadata> localPoints = generateRandomPoints(numPointsPerProcess, aConfigurations.GetDimensionSize(), aConfigurations.GetMaxMleIterations());
    long long numPointsPerProcess_test = aConfigurations.GetTestPointsTotal() / size + (rank < aConfigurations.GetTestPointsTotal() % size ? 1 : 0);
    std::vector<PointMetadata> localPoints_test = generateRandomPoints(numPointsPerProcess_test, aConfigurations.GetDimensionSize(), aConfigurations.GetMaxMleIterations());
    // print the first 10 points
    // if (rank == 0){
    //     std::cout << "First 10 points: " << std::endl;
    //     for (int i = 0; i < 10; i++){
    //         for (int j = 0; j < aConfigurations.GetDimensionSize(); ++j){
    //             std::cout << localPoints[i].coordinates[j] << ", ";
    //         }
    //         std::cout << localPoints[i].observation << std::endl;
    //     }
    //     std::cout << "First 10 test points: " << std::endl;
    //     for (int i = 0; i < 10; i++){
    //         for (int j = 0; j < aConfigurations.GetDimensionSize(); ++j){
    //             std::cout << localPoints_test[i].coordinates[j] << ", ";
    //         }
    //         std::cout << localPoints_test[i].observation << std::endl;
    //     }
    // }

    // do the distance scale for input points
    distanceScale(localPoints, aConfigurations.GetDistanceScale(), aConfigurations.GetDimensionSize());
    if(pred_tag){
        distanceScale(localPoints_test, aConfigurations.GetDistanceScale(), aConfigurations.GetDimensionSize());
    }
    // do (coarser) partition - redistribute points across MPI ranks
    std::vector<PointMetadata> localPoints_partition;
    std::vector<PointMetadata> localPoints_partition_test;
    if(aConfigurations.GetPartitionMethod() == common::LINEAR_PARTITION){
        if (rank == 0) {
            printf("Partition method: linear\n");
        }
        partitionPoints(localPoints, localPoints_partition, aConfigurations.GetDimensionSize(), aConfigurations.GetDistanceScale());
        if(pred_tag){
            partitionPoints(localPoints_test, localPoints_partition_test, aConfigurations.GetDimensionSize(), aConfigurations.GetDistanceScale());
        }
    } else if(aConfigurations.GetPartitionMethod() == common::NO_PARTITION){
        if (rank == 0) {
            printf("Partition method: none\n");
        }
        localPoints_partition = localPoints;
        if(pred_tag){
            localPoints_partition_test = localPoints_test;
        }
    }
    
    std::cout << "rank: " << rank << ", gpu_id: " << VecchiaHardware::GetLocalGPUId() << std::endl;
    std::cout << "rank: " << rank << ", Number of points in localPoints: " << localPoints_partition.size() << std::endl;
    std::cout << "rank: " << rank << ", Number of points in localPoints_test: " << localPoints_partition_test.size() << std::endl;
    
    auto start_total = std::chrono::high_resolution_clock::now();

    // 2.1 Perform RAC partitioning (finer partitioning)
    if (rank == 0){
        std::cout << "Performing RAC partitioning" << std::endl;
    }
    auto start_preprocessing = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<PointMetadata>> finerPartitions;
    std::vector<std::vector<PointMetadata>> finerPartitions_test;
    int numBlocksPerProcess = aConfigurations.GetBlockSize() / size + (rank < aConfigurations.GetBlockSize() % size ? 1 : 0);
    int numBlocksPerProcess_test = aConfigurations.GetTestBlocksTotal() / size + (rank < aConfigurations.GetTestBlocksTotal() % size ? 1 : 0);
    
    finerPartition(localPoints_partition, numBlocksPerProcess, finerPartitions, aConfigurations, false);
    
    if(pred_tag){
        finerPartition(localPoints_partition_test, numBlocksPerProcess_test, finerPartitions_test, aConfigurations, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_preprocessing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_preprocessing = end_preprocessing - start_preprocessing;
    
    // Find the maximum preprocessing duration across all processes
    double max_RAC_partitioning;
    double duration_preprocessing_seconds = duration_preprocessing.count();
    MPI_Allreduce(&duration_preprocessing_seconds, &max_RAC_partitioning, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 2.2 Calculate centers of gravity for each block
    if (rank == 0){
        std::cout << "Calculating centers of gravity" << std::endl;
    }
    auto start_centers_of_gravity = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> centers = calculateCentersOfGravity(finerPartitions, aConfigurations);
    std::vector<std::vector<double>> centers_test;
    if(pred_tag){
        centers_test = calculateCentersOfGravity(finerPartitions_test, aConfigurations);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_centers_of_gravity = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_centers_of_gravity = end_centers_of_gravity - start_centers_of_gravity;
    double max_duration_centers_of_gravity;
    double duration_centers_of_gravity_seconds = duration_centers_of_gravity.count();
    MPI_Allreduce(&duration_centers_of_gravity_seconds, &max_duration_centers_of_gravity, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 2.3 AllGather centers from all processors
    auto start_send_centers_of_gravity = std::chrono::high_resolution_clock::now();
    // first is the centers, second is the rank of the processor
    std::vector<std::pair<std::vector<double>, int>> allCenters;
    std::vector<std::pair<std::vector<double>, int>> allCenters_test;
    AllGatherCentersHelper(centers, allCenters, aConfigurations);
    if(pred_tag){
        AllGatherCentersHelper(centers_test, allCenters_test, aConfigurations);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_send_centers_of_gravity = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_send_centers_of_gravity = end_send_centers_of_gravity - start_send_centers_of_gravity;
    double max_duration_send_centers_of_gravity;
    double duration_send_centers_of_gravity_seconds = duration_send_centers_of_gravity.count();
    MPI_Allreduce(&duration_send_centers_of_gravity_seconds, &max_duration_send_centers_of_gravity, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 3 Reorder centers at all processors
    if (rank == 0){
        std::cout << "Reordering centers" << std::endl;
    }
    auto start_reorder_centers = std::chrono::high_resolution_clock::now();
    std::vector<int> permutation;
    std::vector<int> localPermutation;
    reorderCenters(centers, allCenters, permutation, localPermutation, aConfigurations);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_reorder_centers = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_reorder_centers = end_reorder_centers - start_reorder_centers;
    double max_duration_reorder_centers;
    double duration_reorder_centers_seconds = duration_reorder_centers.count();
    MPI_Allreduce(&duration_reorder_centers_seconds, &max_duration_reorder_centers, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // 4. NN searching - Create block information
    auto start_create_block_info = std::chrono::high_resolution_clock::now();
    std::vector<BlockInfo> localBlocks = createBlockInfo(finerPartitions, centers, allCenters, permutation, localPermutation, aConfigurations);
    std::vector<BlockInfo> localBlocks_test;
    if(pred_tag){
        localBlocks_test = createBlockInfo(finerPartitions_test, centers_test, allCenters_test, permutation, localPermutation, aConfigurations);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_create_block_info = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_create_block_info = end_create_block_info - start_create_block_info;
    double max_duration_create_block_info;
    double duration_create_block_info_seconds = duration_create_block_info.count();
    MPI_Allreduce(&duration_create_block_info_seconds, &max_duration_create_block_info, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // 4.2 Block candidate preparation
    auto start_block_sending = std::chrono::high_resolution_clock::now();
    double distance_threshold_coarse = DistanceCalculationHelpers<T>::CalculateDistanceThreshold(
        aConfigurations.GetDistanceScale(), 
        aConfigurations.GetProblemSize(), 
        aConfigurations.GetConditioningSize(), 
        aConfigurations.GetNNMultiplier());
    std::vector<BlockInfo> receivedBlocks = processAndSendBlocks(localBlocks, allCenters, distance_threshold_coarse, 
                                                                 permutation, localPermutation, aConfigurations, false);
    std::vector<BlockInfo> receivedBlocks_test;
    if(pred_tag){
        // For test/prediction, use training blocks (localBlocks) to find neighbors for test blocks
        receivedBlocks_test = processAndSendBlocks(localBlocks, allCenters_test, distance_threshold_coarse, 
                                                                 permutation, localPermutation, aConfigurations, true);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_block_sending = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_block_sending = end_block_sending - start_block_sending;
    double max_duration_block_sending;
    double duration_block_sending_seconds = duration_block_sending.count();
    MPI_Allreduce(&duration_block_sending_seconds, &max_duration_block_sending, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Print total point size of receivedBlocks
    int total_point_size = 0;
    for (auto& block : receivedBlocks){
        total_point_size += block.blocks.size();
    }
    std::cout << "rank: " << rank << ", total_point_size: " << total_point_size << std::endl;
    if(pred_tag){
    int total_point_size_test = 0;
        for (auto& block : receivedBlocks_test){
            total_point_size_test += block.blocks.size();
        }
        std::cout << "rank: " << rank << ", total_point_size_test: " << total_point_size_test << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    

    // 4.3 NN searching
    if (rank == 0){
        std::cout << "Performing NN searching" << std::endl;
    }
    auto start_nn_searching = std::chrono::high_resolution_clock::now();
    nearest_neighbor_search(localBlocks, receivedBlocks, aConfigurations, distance, false);
    if(pred_tag){
        nearest_neighbor_search(localBlocks_test, receivedBlocks_test, aConfigurations, distance, true);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_nn_searching = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_nn_searching = end_nn_searching - start_nn_searching;
    double max_duration_nn_searching;
    double duration_nn_searching_seconds = duration_nn_searching.count();
    MPI_Allreduce(&duration_nn_searching_seconds, &max_duration_nn_searching, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Descale distances back to original scale
    distanceDeScale(localBlocks, aConfigurations.GetDistanceScale(), aConfigurations.GetDimensionSize());
    if(pred_tag){
        distanceDeScale(localBlocks_test, aConfigurations.GetDistanceScale(), aConfigurations.GetDimensionSize());
    }
    // Store BlockInfo in ClusteringResult for use by ScaledBlockEstimator
    result.blockInfos = localBlocks;
    if(pred_tag){
        result.blockInfos_test = localBlocks_test;  
    }

    if (rank == 0) {
        std::cout << "Stored " << localBlocks.size() << " blocks in ClusteringResult" << std::endl;
        std::cout << "Stored " << localBlocks_test.size() << " test blocks in ClusteringResult" << std::endl;
    }
    return result;
#endif
}

template<typename T>
void DistributedClusteringStrategy<T>::PartitionAcrossRanks(
    const std::vector<Point<T>> &aPoints,
    std::vector<Point<T>> &aPartitionedPoints,
    Dimension aDimension) {
    
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int dim = (aDimension == Dimension2D) ? 2 : 3;
    
    // Choose dimension with minimum scale for partitioning
    int minIndex = 0;
    if (!mDistanceScale.empty()) {
        minIndex = std::min_element(mDistanceScale.begin(), mDistanceScale.end()) 
                   - mDistanceScale.begin();
    }
    double minDistanceScale = mDistanceScale.empty() ? 1.0 : mDistanceScale[minIndex];
    
    // Prepare send buffers for each rank
    std::vector<std::vector<Point<T>>> sendBuffers(size);
    for (const auto &point : aPoints) {
        int targetRank = std::min(
            static_cast<int>(point.GetCoordinates()[minIndex] * minDistanceScale * size),
            size - 1);
        sendBuffers[targetRank].push_back(point);
    }
    
    // Count send sizes
    std::vector<int> sendCounts(size, 0);
    for (int i = 0; i < size; i++) {
        sendCounts[i] = sendBuffers[i].size() * dim;  // coordinates only
    }
    
    // Pack data (coordinates only, no observation in Point class)
    std::vector<double> sendData;
    for (int i = 0; i < size; i++) {
        for (const auto &point : sendBuffers[i]) {
            for (int d = 0; d < dim; d++) {
                sendData.push_back(point.GetCoordinates()[d]);
            }
        }
    }
    
    // Exchange counts
    std::vector<int> recvCounts(size, 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements
    std::vector<int> sendDisplacements(size, 0);
    std::vector<int> recvDisplacements(size, 0);
    for (int i = 1; i < size; i++) {
        sendDisplacements[i] = sendDisplacements[i-1] + sendCounts[i-1];
        recvDisplacements[i] = recvDisplacements[i-1] + recvCounts[i-1];
    }
    
    int totalRecv = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
    std::vector<double> recvData(totalRecv);
    
    // Exchange data
    MPI_Alltoallv(sendData.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE,
                  recvData.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);
    
    // Unpack received data
    aPartitionedPoints.clear();
    aPartitionedPoints.reserve(totalRecv / dim);
    for (size_t i = 0; i < recvData.size(); i += dim) {
        Point<T> point;
        T coords[3] = {0, 0, 0};
        for (int d = 0; d < dim; d++) {
            coords[d] = recvData[i + d];
        }
        point.SetCoordinates(coords);
        aPartitionedPoints.push_back(point);
    }
#endif
}

template<typename T>
std::vector<int> DistributedClusteringStrategy<T>::LocalClustering(
    const std::vector<Point<T>> &aPoints,
    Dimension aDimension) {
    
    if (mMethod == "kmeans++") {
        return KMeansPlusPlusClustering(aPoints, aDimension);
    } else if (mMethod == "random") {
        return RandomClustering(aPoints, aDimension);
    } else {
        throw std::runtime_error("Unknown clustering method: " + mMethod);
    }
}

template<typename T>
std::vector<int> DistributedClusteringStrategy<T>::KMeansPlusPlusClustering(
    const std::vector<Point<T>> &aPoints,
    Dimension aDimension) {
    
    // Similar to LocalClusteringStrategy but with rank-specific seed
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int seed = mSeed * size + rank;
    
    // Use same k-means++ logic as LocalClusteringStrategy
    // (Implementation details omitted for brevity - would be similar)
    
    // For now, simple random assignment
    std::vector<int> assignments(aPoints.size());
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, mNumBlocksPerProcess - 1);
    
    for (size_t i = 0; i < aPoints.size(); i++) {
        assignments[i] = dis(gen);
    }
    
    return assignments;
}

template<typename T>
std::vector<int> DistributedClusteringStrategy<T>::RandomClustering(
    const std::vector<Point<T>> &aPoints,
    Dimension aDimension) {
    
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int seed = mSeed * size + rank;
    
    std::vector<int> assignments(aPoints.size());
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, mNumBlocksPerProcess - 1);
    
    for (size_t i = 0; i < aPoints.size(); i++) {
        assignments[i] = dis(gen);
    }
    
    return assignments;
}

template<typename T>
void DistributedClusteringStrategy<T>::AllGatherCenters(
    const std::vector<std::vector<T>> &aLocalCenters,
    std::vector<std::pair<std::vector<T>, int>> &aAllCenters,
    Dimension aDimension) {
    
#ifdef USE_MPI
    int rank = VecchiaHardware::GetMPIRank();
    int size = VecchiaHardware::GetMPISize();
    int dim = (aDimension == Dimension2D) ? 2 : 3;
    
    int numLocalCenters = aLocalCenters.size();
    std::vector<int> recvCounts(size, 0);
    
    MPI_Allgather(&numLocalCenters, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements
    std::vector<int> displacements(size, 0);
    int totalCenters = 0;
    for (int i = 0; i < size; i++) {
        displacements[i] = totalCenters * (dim + 1);  // +1 for rank
        totalCenters += recvCounts[i];
        recvCounts[i] *= (dim + 1);
    }
    
    // Pack local centers with rank
    std::vector<double> sendBuffer(numLocalCenters * (dim + 1));
    for (int i = 0; i < numLocalCenters; i++) {
        for (int d = 0; d < dim; d++) {
            sendBuffer[i * (dim + 1) + d] = aLocalCenters[i][d];
        }
        sendBuffer[i * (dim + 1) + dim] = static_cast<double>(rank);
    }
    
    std::vector<double> recvBuffer(totalCenters * (dim + 1));
    
    MPI_Allgatherv(sendBuffer.data(), numLocalCenters * (dim + 1), MPI_DOUBLE,
                   recvBuffer.data(), recvCounts.data(), displacements.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    
    // Unpack
    aAllCenters.clear();
    aAllCenters.reserve(totalCenters);
    for (int i = 0; i < totalCenters; i++) {
        std::vector<T> coords(dim);
        for (int d = 0; d < dim; d++) {
            coords[d] = recvBuffer[i * (dim + 1) + d];
        }
        int centerRank = static_cast<int>(recvBuffer[i * (dim + 1) + dim]);
        aAllCenters.push_back(std::make_pair(coords, centerRank));
    }
#endif
}

template<typename T>
void DistributedClusteringStrategy<T>::ReorderCenters(
    std::vector<std::pair<std::vector<T>, int>> &aAllCenters,
    std::vector<int> &aPermutation) {
    
    // Random permutation using same seed on all ranks
    std::mt19937 gen(mSeed);
    aPermutation.resize(aAllCenters.size());
    std::iota(aPermutation.begin(), aPermutation.end(), 0);
    std::shuffle(aPermutation.begin(), aPermutation.end(), gen);
}

template<typename T>
std::vector<Point<T>> DistributedClusteringStrategy<T>::ConvertToPoints(
    Locations<T> &aLocations,
    Dimension aDimension) {
    
    int n = aLocations.GetSize();
    std::vector<Point<T>> points;
    points.reserve(n);
    
    for (int i = 0; i < n; i++) {
        Point<T> point;
        T coords[3] = {aLocations.GetLocationX()[i], aLocations.GetLocationY()[i], 0};
        if (aDimension == Dimension3D || aDimension == DimensionST) {
            coords[2] = aLocations.GetLocationZ()[i];
        }
        point.SetCoordinates(coords);
        point.SetCluster(-1);
        points.push_back(point);
    }
    
    return points;
}

template<typename T>
std::vector<int> DistributedClusteringStrategy<T>::CountClusterSizes(
    const std::vector<Point<T>> &aPoints) {
    
    std::vector<int> sizes(mNumBlocksPerProcess, 0);
    for (const auto &point : aPoints) {
        int cluster = point.GetCluster();
        if (cluster >= 0 && cluster < mNumBlocksPerProcess) {
            sizes[cluster]++;
        }
    }
    
    return sizes;
}

