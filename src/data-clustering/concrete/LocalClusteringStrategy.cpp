
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file LocalClusteringStrategy.cpp
* @version 1.0.0
* @brief Implementation of LocalClusteringStrategy
* @author Mahmoud ElKarargy
* @date 2025-10-20
**/

#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

#include <data-clustering/concrete/LocalClusteringStrategy.hpp>
#include <utilities/Logger.hpp>
#include <helpers/CSVUtils.hpp>

using namespace vecchia::clustering;
using namespace vecchia::common;
using namespace vecchia::dataunits;
using namespace vecchia::configurations;
using namespace vecchia::helpers;

template<typename T>
LocalClusteringStrategy<T>::LocalClusteringStrategy(
    int aMaxIter, int aNumClusters, int aSeed)
    : mMaxIter(aMaxIter), mSeed(aSeed) {
    this->mNumClusters = aNumClusters;
}

template<typename T>
ClusteringResult<T> LocalClusteringStrategy<T>::ComputeClusters(
    Locations<T> &aLocations,
    Configurations &aConfigurations) {
    
    int numPoints = aLocations.GetSize();
    Dimension dim = aLocations.GetDimension();
    
    ClusteringResult<T> result;
    
    // Convert locations to points
    auto points = ConvertToPoints(aLocations, dim);

    // Check if we should use classic Vecchia (point-wise)
    if (numPoints < 2 * this->mNumClusters) {
        LOGGER("------You are using the classic Vecchia!------");

        // Fall back to point-wise
        result.assignments.resize(numPoints);
        std::iota(result.assignments.begin(), result.assignments.end(), 0);
        result.batchSizes.resize(numPoints, 1);
        result.numClusters = numPoints;
        result.isPointWise = true;
        
        // Set cluster assignments
        for (int i = 0; i < numPoints; i++) {
            points[i].SetCluster(i);
        }
    } else {
        // Block Vecchia mode
        LOGGER("------You are using the cluster Vecchia!------");
        
        // Initialize centroids randomly
        std::vector<Point<T>> centroids = RandomInitializer(points);
        
        // Run K-means (default 50 iterations, can be made configurable)
        int numThreads = omp_get_max_threads();
        KMeansParallel(points, centroids, mMaxIter, this->mNumClusters, numThreads);
        
        // After K-means, extract assignments from points
        result.assignments.resize(numPoints);
        for (int i = 0; i < numPoints; i++) {
            result.assignments[i] = points[i].GetCluster();
        }
        
        // Count cluster sizes
        result.batchSizes = CountClusterSizes(points);
        result.numClusters = this->mNumClusters;
        result.isPointWise = false;
    }
    
    result.points = points;
    
    // Calculate centroids for the result
    result.centroids = std::make_unique<Locations<T>>(result.numClusters, dim);
    std::vector<int> clusterCounts(result.numClusters, 0);
    std::vector<T> sumX(result.numClusters, 0.0);
    std::vector<T> sumY(result.numClusters, 0.0);
    std::vector<T> sumZ(result.numClusters, 0.0);
    
    for (const auto &point : points) {
        int cluster = point.GetCluster();
        if (cluster >= 0 && cluster < result.numClusters) {
            sumX[cluster] += point.GetCoordinates()[0];
            sumY[cluster] += point.GetCoordinates()[1];
            if (dim == Dimension3D || dim == DimensionST) {
                sumZ[cluster] += point.GetCoordinates()[2];
            }
            clusterCounts[cluster]++;
        }
    }
    
    for (int i = 0; i < result.numClusters; i++) {
        if (clusterCounts[i] > 0) {
            result.centroids->GetLocationX()[i] = sumX[i] / clusterCounts[i];
            result.centroids->GetLocationY()[i] = sumY[i] / clusterCounts[i];
            if (dim == Dimension3D || dim == DimensionST) {
                result.centroids->GetLocationZ()[i] = sumZ[i] / clusterCounts[i];
            }
        }
    }
    
    LOGGER("--------------Clustering Done-----------------")
    return result;
}

template<typename T>
std::vector<Point<T>> LocalClusteringStrategy<T>::ConvertToPoints(
    Locations<T> &aLocations, Dimension aDimension) {
    
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
        point.SetCluster(-1);  // Not assigned yet
        points.push_back(point);
    }
    
    return points;
}

template<typename T>
std::vector<Point<T>> LocalClusteringStrategy<T>::RandomInitializer(
    const std::vector<Point<T>> &aPoints) {
        
    std::mt19937 gen(mSeed);
    std::uniform_int_distribution<> dis(0, aPoints.size() - 1);
    
    std::vector<Point<T>> centroids;
    int i = 0;
    
    // Lambda to check if a point is already in centroids
    auto contains = [](const std::vector<Point<T>>& vec, const Point<T>& point) {
        for (const auto& p : vec) {
            bool same = true;
            for (int d = 0; d < 3; d++) {
                if (std::abs(p.GetCoordinates()[d] - point.GetCoordinates()[d]) > 1e-9) {
                    same = false;
                    break;
                }
            }
            if (same) return true;
        }
        return false;
    };
    
    // Randomly select k centers without replacement
    while (centroids.size() < this->mNumClusters) {
        Point<T> new_centroid = aPoints[dis(gen)];
        if (!contains(centroids, new_centroid)) {
            new_centroid.SetCluster(i);
            i++;
            centroids.push_back(new_centroid);
        }
    }
    
    return centroids;
}

template<typename T>
void LocalClusteringStrategy<T>::KMeansParallel(std::vector<Point<T>> &aPoints, 
                                         std::vector<Point<T>> &aCentroids,
                                         int aEpochs, int aNumberOfClusters, int aNumThreads) {

    std::vector<Point<T>> newCentroids(aNumberOfClusters, Point<T>());
    std::vector<int> clusterCardinality(aNumberOfClusters, 0);
    
    int pointsPerThread = std::ceil(static_cast<double>(aPoints.size()) / aNumThreads);
    
    for (int epoch = 0; epoch < aEpochs; epoch++) {
        if (epoch % 10 == 0) {
            LOGGER("\tK-means: " + std::to_string(epoch) + " rounds / " + std::to_string(aEpochs) + " finished.");
        }
        
#pragma omp parallel num_threads(aNumThreads)
        {
            std::vector<int> tmpClusterCardinality(aNumberOfClusters, 0);
            std::vector<Point<T>> tmpNewCentroids(aNumberOfClusters, Point<T>());
            
            // Pre-cache centroid coordinates outside the point loop
            std::vector<const T*> centroidCoords(aNumberOfClusters);
            for (int j = 0; j < aNumberOfClusters; j++) {
                centroidCoords[j] = aCentroids[j].GetCoordinates();
            }
            
#pragma omp for nowait schedule(static, pointsPerThread)
            for (Point<T> &p : aPoints) {
                // Get point coordinates once
                const T *pCoords = p.GetCoordinates();
                
                // First centroid - inline distance calculation with SIMD
                T distance = 0;
#pragma omp simd
                for (int d = 0; d < 3; d++) {
                    T diff = pCoords[d] - centroidCoords[0][d];
                    distance += diff * diff;
                }
                T minDistance = distance;
                int nearestCluster = 0;
                
                // Remaining centroids
                for (int j = 1; j < aNumberOfClusters; j++) {
                    distance = 0;
#pragma omp simd
                    for (int d = 0; d < 3; d++) {
                        T diff = pCoords[d] - centroidCoords[j][d];
                        distance += diff * diff;
                    }
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCluster = j;
                    }
                }
                
                p.SetCluster(nearestCluster);
                tmpNewCentroids[nearestCluster] += p;
                tmpClusterCardinality[nearestCluster]++;
            }
            
#pragma omp critical
            {
                for (int i = 0; i < aNumberOfClusters; i++) {
                    newCentroids[i] += tmpNewCentroids[i];
                    clusterCardinality[i] += tmpClusterCardinality[i];
                }
            }
        }
        
        // Update centroids
        for (int i = 0; i < aNumberOfClusters; i++) {
            if (clusterCardinality[i] == 0) {
                // If cluster is empty, randomly reassign a point
                std::mt19937 gen(42);
                std::uniform_int_distribution<> distr(0, aPoints.size() - 1);
                int randomIdx = distr(gen);
                
                newCentroids[i] += aPoints[randomIdx];
                aPoints[randomIdx].SetCluster(i);
                clusterCardinality[i] = 1;
            }
            newCentroids[i] /= clusterCardinality[i];
        }
        
        aCentroids = newCentroids;
        
        // Reset for next iteration
        newCentroids = std::vector<Point<T>>(aNumberOfClusters, Point<T>());
        clusterCardinality = std::vector<int>(aNumberOfClusters, 0);
    }
}


template<typename T>
std::vector<int> LocalClusteringStrategy<T>::RandomClustering(
    const std::vector<Point<T>> &aPoints, Dimension aDimension) {
    
    LOGGER("** Running random clustering **")
    
    int numPoints = aPoints.size();
    int dim = (aDimension == Dimension2D) ? 2 : 3;
    
    std::vector<int> assignments(numPoints);
    std::mt19937 gen(mSeed);
    
    // Randomly select k centers
    std::vector<int> centerIndices(numPoints);
    std::iota(centerIndices.begin(), centerIndices.end(), 0);
    std::shuffle(centerIndices.begin(), centerIndices.end(), gen);
    
    std::vector<std::vector<T>> centers(this->mNumClusters, std::vector<T>(dim));
    for (int k = 0; k < this->mNumClusters; k++) {
        assignments[centerIndices[k]] = k;
        for (int d = 0; d < dim; d++) {
            centers[k][d] = aPoints[centerIndices[k]].GetCoordinates()[d];
        }
    }
    
    // Assign remaining points to nearest center
    #pragma omp parallel for
    for (int i = 0; i < numPoints; i++) {
        // Skip if already assigned (is a center)
        bool isCenter = false;
        for (int k = 0; k < this->mNumClusters; k++) {
            if (i == centerIndices[k]) {
                isCenter = true;
                break;
            }
        }
        if (isCenter) continue;
        
        T minDist = std::numeric_limits<T>::max();
        int nearestCluster = 0;
        
        for (int k = 0; k < this->mNumClusters; k++) {
            T dist = 0;
            for (int d = 0; d < dim; d++) {
                T diff = aPoints[i].GetCoordinates()[d] - centers[k][d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                nearestCluster = k;
            }
        }
        assignments[i] = nearestCluster;
    }
    
    return assignments;
}

template<typename T>
std::vector<int> LocalClusteringStrategy<T>::CountClusterSizes(
    const std::vector<Point<T>> &aPoints) {
    
    std::vector<int> sizes(this->mNumClusters, 0);
    for (const auto &point : aPoints) {
        int cluster = point.GetCluster();
        if (cluster >= 0 && cluster < this->mNumClusters) {
            sizes[cluster]++;
        }
    }
    
    // Print statistics with cluster IDs
    if (!sizes.empty()) {
        // Find max and min cluster IDs with their sizes
        auto maxIt = std::max_element(sizes.begin(), sizes.end());
        auto minIt = std::min_element(sizes.begin(), sizes.end());
        
        int maxClusterId = std::distance(sizes.begin(), maxIt);
        int minClusterId = std::distance(sizes.begin(), minIt);
        
        double avg = std::accumulate(sizes.begin(), sizes.end(), 0.0) / sizes.size();
        
        LOGGER("Cluster with the most points: " << maxClusterId 
               << " (" << *maxIt << " points)")
        LOGGER("Cluster with the least points: " << minClusterId 
               << " (" << *minIt << " points)")
        LOGGER("  Cluster sizes - Min: " << *minIt 
               << ", Max: " << *maxIt 
               << ", Avg: " << avg)
    }
    
    return sizes;
}

template<typename T>
ClusteringResult<T> LocalClusteringStrategy<T>::ComputeClustersForPrediction(
    Configurations &aConfigurations) {
    
    // Get configuration parameters - EXACT CODE FROM PREDICT FUNCTION
    std::string testLocsFile = aConfigurations.GetTestLocationsPath();
    int seed = aConfigurations.GetSeed();
    int n = aConfigurations.GetProblemSize();
    int k = aConfigurations.GetBlockSize();  // Number of clusters
    int dim = aConfigurations.GetDimensionSize();
    int omp_numthreads = aConfigurations.GetCoresNumber();
    int kmeans_iter = aConfigurations.GetKMeansMaxIter();
    
    ClusteringResult<T> result;
    
    // Load test locations from CSV file - EXACT CODE FROM PREDICT FUNCTION
    LOGGER("Loading test locations from CSV file...")
    std::vector<std::vector<T>> testLocs = loadCSV<T>(testLocsFile, dim);
    
    // Update n based on actual test locations size
    n = testLocs.size();
    
    // Perform K-means clustering and find nearest neighbors
    // kmeans - EXACT CODE FROM PREDICT FUNCTION
    std::vector<Point<T>> points;
    std::vector<Point<T>> centroids;
    // transform the locations into points
    points.reserve(n);
    for (int i = 0; i < n; i++) {
        Point<T> point;
        T coords[3] = {testLocs[i][0], testLocs[i][1], 0.0};
        if (dim == 3) {
            coords[2] = testLocs[i][2];
        }
        point.SetCoordinates(coords);
        point.SetCluster(-1);  // Not assigned yet
        points.push_back(point);
    }
    // init the centroids
    centroids = RandomInitializer(points);
    // kmeans_iter, kmeans iterations
    KMeansParallel(points, centroids, kmeans_iter, k, omp_numthreads);
    
    // Store points and centroids in result
    result.points = points;
    result.numClusters = k;
    result.isPointWise = false;
    
    // Extract assignments from points
    result.assignments.resize(n);
    for (int i = 0; i < n; i++) {
        result.assignments[i] = points[i].GetCluster();
    }
    
    // Count cluster sizes
    result.batchSizes = CountClusterSizes(points);
    
    // Store centroids as Locations
    Dimension dimension = (dim == 3) ? Dimension3D : Dimension2D;
    result.centroids = std::make_unique<Locations<T>>(k, dimension);
    for (int i = 0; i < k; i++) {
        const T* coords = centroids[i].GetCoordinates();
        result.centroids->GetLocationX()[i] = coords[0];
        result.centroids->GetLocationY()[i] = coords[1];
        if (dim == 3) {
            result.centroids->GetLocationZ()[i] = coords[2];
        }
    }
    
    LOGGER("--------------Clustering Done for Prediction-----------------")
    return result;
}

