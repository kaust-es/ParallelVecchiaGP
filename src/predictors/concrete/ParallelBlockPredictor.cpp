// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by the STSDS group at KAUST.

/**
 * @file ParallelBlockPredictor.cpp
 * @brief Implementation of Block Vecchia prediction
 * @details Implements Block Vecchia prediction using LocalClusteringStrategy for clustering,
 *          ClusterData for per-cluster operations, and GSL for matrix operations.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <random>
#include <omp.h>

#include <predictors/concrete/ParallelBlockPredictor.hpp>
#include <hardware/VecchiaHardware.hpp>
#include <data-units/VecchiaGBData.hpp>
#include <data-units/Locations.hpp>
#include <data-units/Point.hpp>
#include <data-units/ClusterData.hpp>
#include <configurations/Configurations.hpp>
#include <kernels/Kernel.hpp>
#include <common/PluginRegistry.hpp>
#include <common/Definitions.hpp>
#include <utilities/Logger.hpp>
#include <data-clustering/concrete/LocalClusteringStrategy.hpp>
#include <helpers/CSVUtils.hpp>

using namespace vecchia::predictors;
using namespace vecchia::configurations;
using namespace vecchia::dataunits;
using namespace vecchia::common;
using namespace vecchia::kernels;
using namespace vecchia::plugins;
using namespace vecchia::clustering;
using namespace vecchia::helpers;

template<typename T>
void ParallelBlockPredictor<T>::InitMemory(Configurations &aConfigurations, 
                                         std::unique_ptr<VecchiaGBData<T>> &aData) {
    // No-op for Parallel Block Predictor
    // Memory allocation and data processing happen in Predict() on first call
    LOGGER("ParallelBlockPredictor: Memory will be initialized during prediction")
}

template<typename T>
T ParallelBlockPredictor<T>::Predict(Configurations &aConfigurations, 
                                   std::unique_ptr<VecchiaGBData<T>> &aData, 
                                   const double *apTheta) {
    
    // Get configuration parameters (equivalent to cxxopts parsing)
    std::string trainLocsFile = aConfigurations.GetTrainLocationsPath();
    std::string testLocsFile = aConfigurations.GetTestLocationsPath();
    std::string trainDataFile = aConfigurations.GetTrainDataPath();
    std::string testDataFile = aConfigurations.GetTestDataPath();
    int seed = aConfigurations.GetSeed();
    int n = aConfigurations.GetProblemSize();
    int k = aConfigurations.GetBlockSize();  // Number of clusters
    int m = aConfigurations.GetConditioningSize();  // Number of nearest neighbors
    int dim = aConfigurations.GetDimensionSize();
    int omp_numthreads = aConfigurations.GetCoresNumber();
    int kmeans_iter = aConfigurations.GetKMeansMaxIter();
    
    // Parse theta from initial theta (passed from terminal via --itheta or --theta)
    // Theta should be provided via command line for prediction
    // Match original repo: just use theta as-is (no auto-padding)
    std::vector<T> theta = aConfigurations.GetInitialTheta();
    if (theta.empty()) {
        if (apTheta != nullptr) {
            // Fallback to pointer if not in config
            int theta_size = 2 + dim;
            theta.assign(apTheta, apTheta + theta_size);
        } else {
            // Error if theta not provided
            LOGGER("ERROR: Theta not provided. Please provide theta via --itheta=value or --theta=value (e.g., --theta=1.0,0.5,0.1,0.0 or --itheta=1.0:0.5:0.1:0.0)")
            return static_cast<T>(-15000.0);
        }
    }
    
    // Use theta as-is (matching original repo behavior - no modification)
    // ClusterData expects theta format: [sigma^2, beta, nu, nugget] = 4 values
    
    int distance_metric = static_cast<int>(aConfigurations.GetDistanceMetric());
    double scale_factor = aConfigurations.GetScaleFactor();
    int conditional_sim = aConfigurations.GetConditionalSimulations();
    
    // Print the parameters (matching original code)
    LOGGER("trainLocsFile: " << trainLocsFile)
    LOGGER("testLocsFile: " << testLocsFile)
    LOGGER("trainDataFile: " << trainDataFile)
    LOGGER("testDataFile: " << testDataFile)
    LOGGER("seed: " << seed)
    LOGGER("n: " << n)
    LOGGER("k: " << k)
    LOGGER("m: " << m)
    // Print theta
    LOGGER("theta: ", true)
    for (size_t i = 0; i < theta.size(); i++) {
        LOGGER_PRECISION(theta[i])
        if (i != theta.size() - 1) {
            LOGGER_PRECISION(" ")
        }
    }
    LOGGER("")
    
    omp_set_num_threads(omp_numthreads);
    
    // Check if CSV files are provided - if not, this is normal prediction mode
    if (trainLocsFile.empty() || testLocsFile.empty() || trainDataFile.empty() || testDataFile.empty()) {
        LOGGER("ERROR: CSV file paths not provided. Please provide train_locs, test_locs, train_data, and test_data")
        return static_cast<T>(-15000.0);
    }
    
    // Load the datasets from CSV files (still need train data and test data for prediction)
    LOGGER("Loading training and test data from CSV files...")
    std::vector<T> trainData = loadOneDimensionalData<T>(trainDataFile);
    std::vector<T> testData = loadOneDimensionalData<T>(testDataFile);
    std::vector<std::vector<T>> trainLocs = loadCSV<T>(trainLocsFile, dim);
    
    // Get clustering result from VecchiaGBData (should be pre-clustered in loadData)
    std::vector<Point<T>> points;
    std::vector<Point<T>> centroids;
    
    auto testClusteringResult = aData->GetTestClusteringResult();
    if (testClusteringResult != nullptr) {
        // Use pre-clustered data from VecchiaGBData
        LOGGER("Using pre-clustered test locations from VecchiaGBData...")
        points = testClusteringResult->points;
        k = testClusteringResult->numClusters;
        n = points.size();  // Update n based on actual clustered points size
        
        // Convert centroids from Locations to Points
        if (testClusteringResult->centroids) {
            centroids.reserve(k);
            for (int i = 0; i < k; i++) {
                Point<T> centroid;
                T coords[3] = {
                    testClusteringResult->centroids->GetLocationX()[i],
                    testClusteringResult->centroids->GetLocationY()[i],
                    0.0
                };
                if (dim == 3) {
                    coords[2] = testClusteringResult->centroids->GetLocationZ()[i];
                }
                centroid.SetCoordinates(coords);
                centroid.SetCluster(i);
                centroids.push_back(centroid);
            }
        }
    } else {
        // Fallback: Perform clustering if not pre-clustered (shouldn't happen in normal flow)
        LOGGER("WARNING: Test clustering not found in VecchiaGBData, performing clustering now...")
        // Load test locations for fallback clustering
        std::vector<std::vector<T>> testLocs = loadCSV<T>(testLocsFile, dim);
        n = testLocs.size();
        
        // Perform K-means clustering and find nearest neighbors
        // kmeans
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
        LocalClusteringStrategy<T> clusteringStrategy(kmeans_iter, k, seed);
        centroids = clusteringStrategy.RandomInitializer(points);
        // kmeans_iter, kmeans iterations
        clusteringStrategy.KMeansParallel(points, centroids, kmeans_iter, k, omp_numthreads);
    }

    // Construct data for each cluster
    std::vector<ClusterData<T>> clusters = constructClusterData(points, testData, centroids, trainLocs, trainData, k, m, omp_numthreads, dim);
    saveAllClustersToCSV(clusters, "cluster_data.csv");

    // for each cluster, generate the covariance matrix
    #pragma omp parallel for num_threads(omp_numthreads) schedule(dynamic)
    for (int i = 0; i < int(clusters.size()); i++)
    {
        clusters[i].generateCovarianceMatrix(theta, distance_metric, scale_factor);
        clusters[i].krigingPredict();
    }

    // conditional simulation
    #pragma omp parallel for num_threads(omp_numthreads) schedule(dynamic)
    for (int i = 0; i < int(clusters.size()); i++)
    {
        clusters[i].conditionalSimulate(conditional_sim);
    }

    // calculate the mspe in total
    double mspe = 0.0;
    double mape = 0.0;
    double picp = 0.0;
    double mpiw = 0.0;
    for (int i = 0; i < int(clusters.size()); i++)
    {
        mspe += clusters[i].mspe * clusters[i].numPoints;
        mape += clusters[i].mape * clusters[i].numPoints;
        picp += clusters[i].picp * clusters[i].numPoints;
        mpiw += clusters[i].mpiw * clusters[i].numPoints;
    }
    mspe /= n;
    mape /= n;
    picp /= n;
    mpiw /= n;
    if (mspe <= 0 || std::isnan(mspe) || mape <= 0 || std::isnan(mape)){
        std::cout << "MSPE/MAPE calculation error" << std::endl;
        std::cout << "MSPE: " << mspe << std::endl;
        std::cout << "MAPE: " << mape << std::endl;
    }else{
        std::cout << "MSPE: " << mspe << std::endl;   
        std::cout << "MAPE: " << mape << std::endl;
    }

    // write summary statistics to csv
    writeSummaryStatisticsToCSV(mspe, mape, picp, mpiw, k, m, seed);

    // write full log 
    writeResultsToCSV(clusters, theta, mspe, k, m, seed);
    
    LOGGER("** Block Vecchia Prediction Complete **")
    
    return static_cast<T>(0.0);
}

// Explicit template instantiation
VECCHIAGP_INSTANTIATE_CLASS(ParallelBlockPredictor)
