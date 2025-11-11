
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file ConditioningUpdater.cpp
 * @brief Implementation of ConditioningUpdater class
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <fstream>
#include <string>
#include <vector>

#include <conditioning-updater/ConditioningUpdater.hpp>
#include <common/Definitions.hpp>

using namespace vecchia::conditioningupdater;
using namespace vecchia::configurations;
using namespace vecchia::dataunits;
using namespace vecchia::common;

template<typename T>
void ConditioningUpdater<T>::SaveClusterAndNeighborFiles(Configurations &aConfigurations, VecchiaGBData<T> &aData)
{
    namespace fs = std::filesystem;
    // Ensure output directory exists (creates if missing)
    const fs::path outputDir = "./logs";
    fs::create_directories(outputDir);

    // Determine file suffix based on ordering
    std::string suffix;
    switch (aConfigurations.GetPermutation()) {
        case RANDOM:   suffix = "Random";   break;
        case KD_TREE:  suffix = "KDtree";   break;
        case HILBERT:  suffix = "Hilbert";  break;
        case MORTON:   suffix = "Morton";   break;
        case MMD:      suffix = "MMD";      break;
        default:       suffix = "Default";  break;
    }

    // Build file paths
    const fs::path clustersFile  = outputDir / ("points_" + suffix + ".csv");
    const fs::path neighborsFile = outputDir / ("neighbors_" + suffix + ".csv");

    // ---- Write clusters file ----
    {
        std::ofstream outFile(clustersFile, std::ios::out | std::ios::trunc);
        // if (!outFile)
        //     throw std::runtime_error("Failed to open " + clustersFile.string());

        // Header
        outFile << "x,y,cluster\n";
        for (int i = 0; i < aData.GetBatchCount(); i++)
        {

            for (int j = 0; j < aData.GetBatchNum()[i]; j++)
            {
                const int idx = aData.GetBatchNumAccum()[i] + j;
                outFile << aData.GetNewLocations()->GetLocationX()[idx] << ','
                        << aData.GetNewLocations()->GetLocationY()[idx] << ','
                        << i << '\n';
            }
            
        }
        outFile.close();
    }

    // ---- Write neighbors file ----
    {
        std::ofstream outFile(neighborsFile);
        // if (!outFile.is_open())
        //     throw std::runtime_error("Failed to open " + neighborsFile);

        outFile << "x,y,cluster\n";
        for (int i = 0; i < aConfigurations.GetConditioningSize() * aData.GetBatchCount(); ++i)
        {
            outFile << aData.GetConditioningLocations()->GetLocationX()[i] << ','
                    << aData.GetConditioningLocations()->GetLocationY()[i] << ','
                    << i / aConfigurations.GetConditioningSize() << '\n';
        }
    }
}