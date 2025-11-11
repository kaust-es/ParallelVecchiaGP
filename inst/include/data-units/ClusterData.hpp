// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at KAUST.

/**
 * @file ClusterData.hpp
 * @brief Cluster data structure for Block Vecchia prediction
 * @details Contains cluster locations, observations, nearest neighbors, 
 *          covariance matrices, and prediction results for block Vecchia approximation.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#ifndef VECCHIAGP_CLUSTERDATA_HPP
#define VECCHIAGP_CLUSTERDATA_HPP

#include <vector>
#include <string>
#include <data-units/Point.hpp>

namespace vecchia {
    namespace dataunits {
        
        /**
         * @class ClusterData
         * @brief Class to hold cluster-level data for Block Vecchia prediction
         * @tparam T Data Type: float or double
         * 
         * Contains the spatial locations, observations, nearest neighbors,
         * covariance matrices, and prediction results for a single cluster
         * in the Block Vecchia approximation.
         */
        template<typename T>
        class ClusterData {
        public:
            // Cluster locations and observations
            std::vector<std::vector<T>> clustersLocations;        ///< Cluster point coordinates
            std::vector<T> observations;                           ///< Observations for cluster points
            int numPoints;                                         ///< Number of points in cluster
            std::vector<T> centroids;                             ///< Cluster centroid coordinates
            int dimension;                                         ///< Dimension (2 for 2D, 3 for 3D)
            
            // Nearest neighbors
            std::vector<std::vector<T>> nearestNeighborsLocations; ///< Nearest neighbor coordinates
            std::vector<T> nearestNeighborsObservations;           ///< Observations for nearest neighbors
            
            // Covariance matrices
            std::vector<std::vector<T>> covarianceMat_cluster;    ///< Covariance matrix for cluster
            std::vector<std::vector<T>> covarianceMat_nearestNeighbors; ///< Covariance matrix for neighbors
            std::vector<std::vector<T>> covarianceMat_cross;     ///< Cross-covariance matrix
            
            // Prediction results
            std::vector<T> predictedValues;                        ///< Predicted values
            std::vector<std::vector<T>> predictedUncertainties;    ///< Prediction uncertainty matrix
            T mspe;                                                ///< Mean squared prediction error
            T mape;                                                ///< Mean absolute prediction error
            T picp;                                                ///< Prediction interval coverage percentage
            T mpiw;                                                ///< Mean prediction interval width
            int clusterIndex;                                      ///< Index of this cluster
            
            // Conditional simulation results
            std::vector<double> mean;                              ///< Mean from conditional simulation
            std::vector<double> variance;                          ///< Variance from conditional simulation
            
           

            ClusterData(int dim): numPoints(0), dimension(dim), mspe(0), mape(0), picp(0), mpiw(0), mean(0), variance(0) {};
            ~ClusterData() {};
            
            /**
             * @brief Generate covariance matrices using kernel parameters
             * @param[in] theta Kernel parameters [sigma_square, range, nu, nugget]
             * @param[in] distance_metric Distance metric (1 for earth, 2 for euclidean)
             * @param[in] scale_factor Scale factor for covariance matrix (default 1.0)
             */
            void generateCovarianceMatrix(const std::vector<T>& theta, int distance_metric, const double scale_factor = 1.0);
            
            /**
             * @brief Perform kriging prediction
             * @details Computes predicted values and uncertainties using the covariance matrices
             */
            void krigingPredict();
            
            /**
             * @brief Perform conditional simulation
             * @param[in] m_replicates Number of simulation replicates
             */
            void conditionalSimulate(int m_replicates);
            
            /**
             * @brief Save cluster and neighbor locations to CSV file
             * @param[in] filename Output CSV filename
             */
            void saveToCSV(const std::string& filename) const;
        };
        
        /**
         * @brief Save all clusters to CSV file
         * @param[in] clusters Vector of ClusterData objects
         * @param[in] filename Output CSV filename
         */
        template<typename T>
        void saveAllClustersToCSV(const std::vector<ClusterData<T>>& clusters, const std::string& filename);
        
        /**
         * @brief Construct ClusterData objects from Points and training data
         * @param[in] points Vector of Points with cluster assignments
         * @param[in] observations Observations for each point
         * @param[in] centroids_points Vector of centroid Points
         * @param[in] trainLocs Training location coordinates
         * @param[in] trainObservations Training observations
         * @param[in] k Number of clusters
         * @param[in] m Number of nearest neighbors to find
         * @param[in] num_threads Number of threads for parallel processing
         * @param[in] dimension Dimension of the data (2 or 3)
         * @return Vector of ClusterData objects
         */
        template<typename T>
        std::vector<ClusterData<T>> constructClusterData(
            const std::vector<Point<T>>& points, 
            const std::vector<T>& observations, 
            const std::vector<Point<T>>& centroids_points, 
            const std::vector<std::vector<T>>& trainLocs, 
            const std::vector<T>& trainObservations, 
            int k, int m, int num_threads, int dimension);
        
    } // namespace dataunits
} // namespace vecchia

#endif // VECCHIAGP_CLUSTERDATA_HPP

