// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at KAUST.

/**
 * @file ClusterData.cpp
 * @brief Implementation of ClusterData class for Block Vecchia prediction
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#include <data-units/ClusterData.hpp>
#include <data-units/Point.hpp>
#include <common/Definitions.hpp>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <omp.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

using namespace vecchia::dataunits;
using namespace vecchia::common;

// Helper functions
namespace {
    inline double deg2rad(double deg) {
        return deg * PI / 180.0;
    }
    
    inline double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
    double lat1r, lon1r, lat2r, lon2r, u, v;
    lat1r = deg2rad(lat1d);
    lon1r = deg2rad(lon1d);
    lat2r = deg2rad(lat2d);
    lon2r = deg2rad(lon2d);
    u = std::sin((lat2r - lat1r) / 2);
    v = std::sin((lon2r - lon1r) / 2);
        return 2.0 * EARTH_RADIUS * std::asin(std::sqrt(u * u + std::cos(lat1r) * std::cos(lat2r) * v * v));
    }
    
    template<typename T>
    double euclideanDistance(const std::vector<T>& p1, const std::vector<T>& p2) {
    double sum = 0.0;
    for (size_t i = 0; i < p1.size() && i < p2.size(); ++i) {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
        return std::sqrt(sum);
    }
    
    template<typename T>
    void core_dcmg(std::vector<std::vector<T>>& locs1, 
               std::vector<std::vector<T>>& locs2, 
               std::vector<std::vector<T>>& A,
               const std::vector<T>& localtheta, 
               int distance_metric,
               const double scale_factor) {
    double expr = 0.0, con = 0.0;
    double sigma_square = localtheta[0];
    double nugget = localtheta[3];
    int length1 = locs1.size();
    int length2 = locs2.size();

    // Compute constant part of the Matern function
    con = pow(2, (localtheta[2] - 1)) * tgamma(localtheta[2]);
    con = 1.0 / con;
    con = sigma_square * con;

    // Loop through matrix A
    for (int i = 0; i < length1; ++i) {
        for (int j = 0; j < length2; ++j) {
            // Calculate distance between points in l1 and l2
            if (distance_metric == 1) {
                expr = distanceEarth(locs1[i][0], locs1[i][1], locs2[j][0], locs2[j][1]) / localtheta[1];
            } else {
                expr = euclideanDistance(locs1[i], locs2[j]) / localtheta[1];
            }
            expr /= scale_factor;
            if (expr == 0.0) {
                A[i][j] = sigma_square + nugget;  // Diagonal elements
            } else {
                // Matern covariance function
                A[i][j] = con * pow(expr, localtheta[2]) * gsl_sf_bessel_Knu(localtheta[2], expr);
            }
        }
    }

        // Disable GSL error handling
        gsl_set_error_handler_off();
    }
    
    template<typename T>
    std::vector<int> findNearestNeighborsIndices(const std::vector<T>& centroid, 
                                                 const std::vector<std::vector<T>>& trainLocs, 
                                                 int m) {
        std::vector<std::pair<double, int>> distances;
        for (int i = 0; i < static_cast<int>(trainLocs.size()); ++i) {
            double dist = euclideanDistance(centroid, trainLocs[i]);
            distances.push_back({dist, i});
        }
        
        std::sort(distances.begin(), distances.end());
        
        std::vector<int> neighborIndices;
        for (int i = 0; i < std::min(m, static_cast<int>(distances.size())); ++i) {
            neighborIndices.push_back(distances[i].second);
        }
        
        return neighborIndices;
    }
} // anonymous namespace

// Constructor and destructor are defined inline in the header file

template<typename T>
void ClusterData<T>::generateCovarianceMatrix(const std::vector<T>& theta, int distance_metric, const double scale_factor) {
    int size_cluster = clustersLocations.size();
    int size_nearest = nearestNeighborsLocations.size();
    
    // Implementation of covariance matrix generation
    covarianceMat_cluster.resize(size_cluster, std::vector<T>(size_cluster, 0.0));
    covarianceMat_nearestNeighbors.resize(size_nearest, std::vector<T>(size_nearest, 0.0));
    covarianceMat_cross.resize(size_nearest, std::vector<T>(size_cluster, 0.0));
    
    core_dcmg(clustersLocations, clustersLocations, covarianceMat_cluster, theta, distance_metric, scale_factor);
    core_dcmg(nearestNeighborsLocations, nearestNeighborsLocations, covarianceMat_nearestNeighbors, theta, distance_metric, scale_factor);
    core_dcmg(nearestNeighborsLocations, clustersLocations, covarianceMat_cross, theta, distance_metric, scale_factor);
}

template<typename T>
void ClusterData<T>::krigingPredict() {
    // Implementation of kriging prediction
    // 1. cholesky decomposition for covarianceMat_nearestNeighbors
    int size_nearest = nearestNeighborsLocations.size();
    int size_cluster = clustersLocations.size();
    
    // set predictedValues, predictedUncertainties to 0
    predictedValues.resize(size_cluster, 0.0);
    predictedUncertainties.resize(size_cluster, std::vector<T>(size_cluster, 0.0));
    
    gsl_matrix* covmat = gsl_matrix_alloc(size_nearest, size_nearest);
    // initialize the matrix L
    for (int i = 0; i < size_nearest; ++i) {
        for (int j = 0; j < size_nearest; ++j) {
            gsl_matrix_set(covmat, i, j, covarianceMat_nearestNeighbors[i][j]);
        }
    }
    gsl_linalg_cholesky_decomp1(covmat);
    
    // 2. solve the linear equations
    gsl_vector* obs_nearest = gsl_vector_alloc(size_nearest);
    for (int i = 0; i < size_nearest; ++i) {
        gsl_vector_set(obs_nearest, i, nearestNeighborsObservations[i]);
    }
    gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, covmat, obs_nearest);
    
    // solve the linear equations for the cross covariance matrix
    gsl_matrix* b_cross = gsl_matrix_alloc(size_nearest, size_cluster);
    gsl_vector* _b_inter = gsl_vector_alloc(size_nearest);
    for (int i = 0; i < size_nearest; ++i) {
        for (int j = 0; j < size_cluster; ++j) {
            gsl_matrix_set(b_cross, i, j, covarianceMat_cross[i][j]);
        }
    }
    for (int i = 0; i < size_cluster; ++i) {
        // Extract column i of b_cross
        gsl_matrix_get_col(_b_inter, b_cross, i);
        // solve the linear equations
        gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, covmat, _b_inter);
        // store the result
        gsl_matrix_set_col(b_cross, i, _b_inter);
    }
   
    
    // 3. matrix multiplication
    gsl_matrix* cov_offset = gsl_matrix_alloc(size_cluster, size_cluster);
    gsl_vector* obs_mean = gsl_vector_alloc(size_cluster);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, b_cross, b_cross, 0.0, cov_offset);
    gsl_blas_dgemv(CblasTrans, 1.0, b_cross, obs_nearest, 0.0, obs_mean);
    
    // 4. matrix subtraction
    gsl_matrix* cov_predict = gsl_matrix_alloc(size_cluster, size_cluster);
    // copy covarianceMat_cluster to cov_predict
    for (int i = 0; i < size_cluster; ++i) {
        for (int j = 0; j < size_cluster; ++j) {
            gsl_matrix_set(cov_predict, i, j, covarianceMat_cluster[i][j]);
        }
    }
    gsl_matrix_sub(cov_predict, cov_offset);
    
    // 5. copy the result to the observations and uncertainties
    for (int i = 0; i < size_cluster; ++i) {
        predictedValues[i] = gsl_vector_get(obs_mean, i);
        for (int j = 0; j < size_cluster; ++j) {
            predictedUncertainties[i][j] = gsl_matrix_get(cov_predict, i, j);
        }
    }
    
    // 6. calculate the mspe and mape
    for (int i = 0; i < size_cluster; ++i) {
        mspe += pow(observations[i] - predictedValues[i], 2);
        mape += abs(observations[i] - predictedValues[i]);
    }
    mspe /= size_cluster;
    mape /= size_cluster;
    
    if (mspe <= 0 || std::isnan(mspe)) {
        std::cerr << "Warning: mspe calculation issue - mspe: " << mspe << std::endl;
        std::cerr << "  number of cluster: " << size_cluster << std::endl;
        std::cerr << "  number of nearest neighbors: " << size_nearest << std::endl;
    }

    gsl_set_error_handler_off();  // Disable the default error handler

    // 8. free the memory
    gsl_matrix_free(b_cross);
    gsl_matrix_free(covmat);
    gsl_matrix_free(cov_offset);
    gsl_vector_free(obs_mean);
    gsl_vector_free(obs_nearest);
}

template<typename T>
void ClusterData<T>::conditionalSimulate(int m_replicates) {
    // Implementation of conditional simulation
    // simulate the gaussian random field using predictedValues and predictedUncertainties
    // generate m_replicates of the gaussian random field with dimension size_cluster
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0, 1);
    
    int total_points = numPoints * m_replicates;
    std::vector<double> noise(total_points);
    for (int i = 0; i < total_points; ++i) {
        noise[i] = dist(gen);
    }
    
    // cholesky decomposition for predictedUncertainties
    gsl_matrix* covmat = gsl_matrix_alloc(numPoints, numPoints);
    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            gsl_matrix_set(covmat, i, j, predictedUncertainties[i][j]);
        }
    }
    gsl_linalg_cholesky_decomp(covmat);
    
    // matrix to vector multiplication
    std::vector<gsl_vector*> noise_vecs(m_replicates);
    for (int i = 0; i < m_replicates; ++i) {
        noise_vecs[i] = gsl_vector_alloc(numPoints);
    }
    for (int i = 0; i < m_replicates; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            gsl_vector_set(noise_vecs[i], j, noise[i * numPoints + j]);
        }
    }
    for (int i = 0; i < m_replicates; ++i) {
        gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, covmat, noise_vecs[i]);
    }
    
    // reuse the noise to store the result for the conditional simulation
    for (int i = 0; i < m_replicates; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            noise[i * numPoints + j] = gsl_vector_get(noise_vecs[i], j) + predictedValues[j];
        }
    }
    
    // calculate the mean and variance of the conditional simulation
    mean.resize(numPoints, 0.0);
    variance.resize(numPoints, 0.0);
    for (int i = 0; i < m_replicates; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            mean[j] += noise[i * numPoints + j];
        }
    }
    for (int i = 0; i < numPoints; ++i) {
        mean[i] /= m_replicates;
    }
    for (int i = 0; i < m_replicates; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            variance[j] += pow(noise[i * numPoints + j] - mean[j], 2);
        }
    }
    for (int i = 0; i < numPoints; ++i) {
        variance[i] /= m_replicates;
    }
    
    // Calculate the predicted interval coverage percentage (PICP)
    for (int i = 0; i < numPoints; ++i) {
        double lower_bound = mean[i] - 1.96 * sqrt(variance[i]);
        double upper_bound = mean[i] + 1.96 * sqrt(variance[i]);
        if (observations[i] >= lower_bound && observations[i] <= upper_bound) {
            picp += 1.0;
        }
    }
    picp = (picp / numPoints) * 100.0;

    // Calculate the mean predicted interval width (MPIW)
    for (int i = 0; i < numPoints; ++i) {
        double interval_width = 2 * 1.96 * sqrt(variance[i]);
        mpiw += interval_width;
    }
    mpiw /= numPoints;
    
    // free the memory
    for (int i = 0; i < m_replicates; ++i) {
        gsl_vector_free(noise_vecs[i]);
    }
    gsl_matrix_free(covmat);
}

template<typename T>
void ClusterData<T>::saveToCSV(const std::string& filename) const {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    
    // Write cluster locations and observations
    for (size_t i = 0; i < clustersLocations.size(); ++i) {
        file << clusterIndex << ",cluster";
        for (const auto& coord : clustersLocations[i]) {
            file << "," << coord;
        }
        file << "," << observations[i] << "\n";
    }
    
    // Write nearest neighbor locations and observations
    for (size_t i = 0; i < nearestNeighborsLocations.size(); ++i) {
        file << clusterIndex << ",neighbor";
        for (const auto& coord : nearestNeighborsLocations[i]) {
            file << "," << coord;
        }
        file << "," << nearestNeighborsObservations[i] << "\n";
    }
    
    file.close();
}

namespace vecchia {
namespace dataunits {

// Helper function to save all clusters to CSV
template<typename T>
void saveAllClustersToCSV(const std::vector<ClusterData<T>>& clusters, const std::string& filename) {
    // Clear the file if it exists
    std::ofstream file(filename, std::ios::trunc);
    file.close();
    
    if (clusters.empty()) {
        return;
    }
    
    // Write header
    file.open(filename, std::ios::app);
    file << "cluster_index,point_type";
    for (int i = 0; i < clusters[0].dimension; ++i) {
        file << ",coord_" << i;
    }
    file << ",observation\n";
    file.close();
    
    // Save each cluster's data
    for (const auto& cluster : clusters) {
        cluster.saveToCSV(filename);
    }
}

// Function to construct ClusterData from Points and training data
template<typename T>
std::vector<ClusterData<T>> constructClusterData(
    const std::vector<Point<T>>& points, 
    const std::vector<T>& observations, 
    const std::vector<Point<T>>& centroids_points, 
    const std::vector<std::vector<T>>& trainLocs, 
    const std::vector<T>& trainObservations, 
    int k, int m, int num_threads, int dimension) {
    
        std::vector<ClusterData<T>> clusters(k, ClusterData<T>(dimension));

        // Initialize clusters with centroids and find nearest neighbors
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < k; ++i) {
            // reserve the space for the nearest neighbors
            clusters[i].nearestNeighborsObservations.reserve(m);
            clusters[i].nearestNeighborsLocations.reserve(m);
            // assign the centroids
            const T* coords = centroids_points[i].GetCoordinates();
            clusters[i].centroids.assign(coords, coords + dimension);
            std::vector<int> nearestIndices = findNearestNeighborsIndices(clusters[i].centroids, trainLocs, m);
            for (int idx : nearestIndices) {
                clusters[i].nearestNeighborsLocations.push_back(trainLocs[idx]);
                clusters[i].nearestNeighborsObservations.push_back(trainObservations[idx]);
            }
            clusters[i].clusterIndex = i;
        }
    
        // Assign points to clusters based on k-means results
        for (size_t i = 0; i < points.size(); ++i) {
            int clusterIndex = points[i].GetCluster();
            const T* coords = points[i].GetCoordinates();
            std::vector<T> pointLocation(coords, coords + dimension);
            clusters[clusterIndex].clustersLocations.push_back(pointLocation);
            clusters[clusterIndex].observations.push_back(observations[i]);
            clusters[clusterIndex].numPoints++;
        }
    
        // if the number of points in the cluster is 0, then print a warning and remove the cluster
        for (int i = 0; i < k; ++i) {
            if (clusters[i].numPoints == 0) {
                std::cout << "Warning: cluster " << i << " has no points" << std::endl;
                clusters.erase(clusters.begin() + i);
                --i;
                --k;
            }
        }
    
        return clusters;
}

} // namespace dataunits
} // namespace vecchia

// Explicit template instantiation
VECCHIAGP_INSTANTIATE_CLASS(ClusterData)

// Explicit instantiation for helper functions
// Use fully qualified names for explicit instantiation to avoid ambiguity
template void vecchia::dataunits::saveAllClustersToCSV<double>(const std::vector<vecchia::dataunits::ClusterData<double>>& clusters, const std::string& filename);
template std::vector<vecchia::dataunits::ClusterData<double>> vecchia::dataunits::constructClusterData<double>(
    const std::vector<vecchia::dataunits::Point<double>>& points, 
    const std::vector<double>& observations, 
    const std::vector<vecchia::dataunits::Point<double>>& centroids_points, 
    const std::vector<std::vector<double>>& trainLocs, 
    const std::vector<double>& trainObservations, 
    int k, int m, int num_threads, int dimension);

