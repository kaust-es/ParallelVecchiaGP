// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at KAUST.

/**
 * @file CSVUtils.hpp
 * @brief CSV utility functions for loading and saving data
 * @details Provides functions for loading CSV files, saving cluster information,
 *          and writing summary statistics and results.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#ifndef VECCHIAGP_CSVUTILS_HPP
#define VECCHIAGP_CSVUTILS_HPP

#include <string>
#include <vector>
#include <data-units/ClusterData.hpp>
#include <data-units/Point.hpp>

namespace vecchia::helpers {
    
    /**
     * @brief Load CSV file with multi-dimensional data
     * @param[in] filename Path to CSV file
     * @param[in] dim Dimension of data (number of coordinates per row)
     * @return Vector of vectors containing the loaded data
     */
    template<typename T>
    std::vector<std::vector<T>> loadCSV(const std::string &filename, int dim);
    
    /**
     * @brief Load one-dimensional data from CSV file
     * @param[in] filename Path to CSV file
     * @return Vector containing the loaded data
     */
    template<typename T>
    std::vector<T> loadOneDimensionalData(const std::string &filename);
    
    /**
     * @brief Save cluster information to CSV file
     * @param[in] clusters Vector of ClusterData objects
     * @param[in] centroids Vector of centroid Points
     * @param[in] filename Output CSV filename
     */
    template<typename T>
    void saveClusterInfo(const std::vector<dataunits::ClusterData<T>>& clusters, 
                        const std::vector<dataunits::Point<T>>& centroids, 
                        const std::string& filename);
    
    /**
     * @brief Write summary statistics to CSV file
     * @param[in] mspe Mean squared prediction error
     * @param[in] mape Mean absolute prediction error
     * @param[in] picp Prediction interval coverage percentage
     * @param[in] mpiw Mean prediction interval width
     * @param[in] k Number of clusters
     * @param[in] m Number of nearest neighbors
     * @param[in] seed Random seed
     */
    template<typename T>
    void writeSummaryStatisticsToCSV(T mspe, T mape, T picp, T mpiw, int k, int m, int seed);
    
    /**
     * @brief Write conditional simulation results to CSV file
     * @param[in] clusters Vector of ClusterData objects
     * @param[in] theta Theta parameters
     * @param[in] mspe Mean squared prediction error
     * @param[in] k Number of clusters
     * @param[in] m Number of nearest neighbors
     * @param[in] seed Random seed
     */
    template<typename T>
    void writeResultsToCSV(const std::vector<dataunits::ClusterData<T>>& clusters, 
                          const std::vector<T>& theta, 
                          T mspe, int k, int m, int seed);
    
} // namespace vecchia::helpers

#endif // VECCHIAGP_CSVUTILS_HPP

