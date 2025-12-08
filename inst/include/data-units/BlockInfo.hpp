// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at KAUST.

/**
 * @file BlockInfo.hpp
 * @brief Block information structure for Scaled Block Vecchia
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-28
**/

#ifndef VECCHIAGP_BLOCKINFO_HPP
#define VECCHIAGP_BLOCKINFO_HPP

#include <vector>

namespace vecchia {
    namespace dataunits {
        
        /**
         * @brief Structure to hold block-level information for GPU processing
         * 
         * Contains the spatial locations, observations, and nearest neighbors
         * for a single block in the Scaled Block Vecchia approximation.
         */
        struct BlockInfo {
            int localOrder;                                      ///< Local order within process
            int globalOrder;                                     ///< Global order across all processes
            std::vector<double> center;                          ///< Center of gravity coordinates
            std::vector<std::vector<double>> blocks;            ///< Block point coordinates
            std::vector<std::vector<double>> nearestNeighbors;  ///< Nearest neighbor coordinates
            std::vector<double> observations_blocks;             ///< Observations for block points
            std::vector<double> observations_nearestNeighbors;  ///< Observations for nearest neighbors
            
            /**
             * @brief Default constructor
             */
            BlockInfo() = default;
        };
        
    } // namespace dataunits
} // namespace vecchia

#endif // VECCHIAGP_BLOCKINFO_HPP

