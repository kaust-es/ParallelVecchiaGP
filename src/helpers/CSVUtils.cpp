// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at KAUST.

/**
 * @file CSVUtils.cpp
 * @brief Implementation of CSV utility functions
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#include <helpers/CSVUtils.hpp>
#include <data-units/ClusterData.hpp>
#include <data-units/Point.hpp>
#include <common/Definitions.hpp>
#include <fstream>
#include <sstream>
#include <limits>
#include <filesystem>
#include <vector>
#include <algorithm>

using namespace vecchia::dataunits;
using namespace vecchia::common;

namespace vecchia {
namespace helpers {

template<typename T>
std::vector<std::vector<T>> loadCSV(const std::string &filename, int dim) {
    std::vector<std::vector<T>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;  // Skip empty lines
        
        std::stringstream ss(line);
        std::vector<T> row(dim);
        for (int i = 0; i < dim; ++i) {
            ss >> row[i];
            if (ss.peek() == ',')
                ss.ignore();
        }
        data.push_back(row);
    }
    return data;
}

template<typename T>
std::vector<T> loadOneDimensionalData(const std::string &filename) {
    std::vector<T> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        T value;
        if (ss >> value) {
            data.push_back(value);
        }
    }
    return data;
}

template<typename T>
void saveClusterInfo(const std::vector<ClusterData<T>>& clusters, 
                    const std::vector<Point<T>>& centroids, 
                    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    
    file << "type,cluster_id,x,y,z\n";
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        // Save centroid
        file << "centroid," << clusters[i].clusterIndex;
        const T* coords = centroids[i].GetCoordinates();
        // Point always has 3 coordinates (x, y, z)
        for (int j = 0; j < 3; ++j) {
            file << "," << coords[j];
        }
        file << "\n";
        
        // Save cluster points
        for (const auto& point : clusters[i].clustersLocations) {
            file << "point," << clusters[i].clusterIndex;
            for (const T& coord : point) {
                file << "," << coord;
            }
            file << "\n";
        }
        
        // Save nearest neighbors
        for (const auto& neighbor : clusters[i].nearestNeighborsLocations) {
            file << "neighbor," << clusters[i].clusterIndex;
            for (const T& coord : neighbor) {
                file << "," << coord;
            }
            file << "\n";
        }
    }
    file.close();
}

template<typename T>
void writeSummaryStatisticsToCSV(T mspe, T mape, T picp, T mpiw, int k, int m, int seed) {
    std::ofstream csvFile;
    std::string filename = "log/summary_statistics_k_" + std::to_string(k) + "_m_" + std::to_string(m) + ".csv";
    bool fileExists = std::filesystem::exists(filename);
    csvFile.open(filename, std::ios_base::app);
    if (!fileExists) {
        csvFile << "k,m,seed,mspe,mape,picp,mpiw\n";
    }
    csvFile << k << "," << m << "," << seed << "," << mspe << "," << mape << "," << picp << "," << mpiw << "\n";
    csvFile.close();
}

template<typename T>
void writeResultsToCSV(const std::vector<ClusterData<T>>& clusters, 
                      const std::vector<T>& theta, 
                      T mspe, int k, int m, int seed) {
    // save the mean and variance of the conditional simulation as a csv file
    // create log folder
    std::string log_folder = "log";
    if (!std::filesystem::exists(log_folder)) {
        std::filesystem::create_directory(log_folder);
    }
    std::ofstream csvFile;
    if (theta.size() == 3) {
        csvFile.open("log/conditional_simulation_k_" + std::to_string(k) + "_m_" + std::to_string(m) + "_theta_" + std::to_string(theta[0]) + "_" + std::to_string(theta[1]) + "_" + std::to_string(theta[2]) + "_seed_" + std::to_string(seed) + ".csv");
    } else {
        csvFile.open("log/conditional_simulation_k_" + std::to_string(k) + "_m_" + std::to_string(m) + "_theta_" + std::to_string(theta[0]) + "_" + std::to_string(theta[1]) + "_" + std::to_string(theta[2]) + "_" + std::to_string(theta[3]) + "_seed_" + std::to_string(seed) + ".csv");
    }
    // header
    csvFile << "smean,svariance,x,y,z,mspe,true_value\n";
    for (const auto& cluster : clusters) {
        for (int i = 0; i < cluster.numPoints; ++i) {
            csvFile << cluster.mean[i] << "," << cluster.variance[i] << "," 
                    << cluster.clustersLocations[i][0] << "," 
                    << cluster.clustersLocations[i][1] << "," 
                    << cluster.clustersLocations[i][2] << "," 
                    << mspe << "," 
                    << cluster.observations[i] << std::endl;
        }
    }
    csvFile.close();
}

} // namespace helpers
} // namespace vecchia

// Explicit template instantiation
// Note: VECCHIAGP_INSTANTIATE_CLASS doesn't work for non-class templates
// so we explicitly instantiate the functions
// Use fully qualified names for explicit instantiation to avoid ambiguity
template std::vector<std::vector<double>> vecchia::helpers::loadCSV<double>(const std::string &filename, int dim);
template std::vector<double> vecchia::helpers::loadOneDimensionalData<double>(const std::string &filename);
template void vecchia::helpers::saveClusterInfo<double>(const std::vector<vecchia::dataunits::ClusterData<double>>& clusters, 
                                     const std::vector<vecchia::dataunits::Point<double>>& centroids, 
                                     const std::string& filename);
template void vecchia::helpers::writeSummaryStatisticsToCSV<double>(double mspe, double mape, double picp, double mpiw, int k, int m, int seed);
template void vecchia::helpers::writeResultsToCSV<double>(const std::vector<vecchia::dataunits::ClusterData<double>>& clusters, 
                                       const std::vector<double>& theta, 
                                       double mspe, int k, int m, int seed);

