
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file VecchiaGP.cpp
 * @brief High-Level Wrapper class containing the static API for VecchiaGP operations.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <api/VecchiaGP.hpp>
#include <utilities/Logger.hpp>
#include <kernels/Kernel.hpp>
#include <data-generators/DataGenerator.hpp>
#include <data-clustering/ClusteringFactory.hpp>
#include <data-clustering/ClusteringStrategy.hpp>
#include <data-generators/LocationGenerator.hpp>
#include <data-units/ModelingDataHolders.hpp>
#include <data-units/Point.hpp>
#include <conditioning-updater/ConditioningUpdater.hpp>
#include <conditioning-updater/concrete/KnnConditioningUpdater.hpp>
#include <conditioning-updater/concrete/SequentialConditioningUpdater.hpp>
#include <estimators/EstimatorFactory.hpp>
#include <predictors/PredictorFactory.hpp>
#include <helpers/LocationSorter.hpp>
#include <helpers/BatchPreparationUtility.hpp>
#include <helpers/CSVUtils.hpp>
#include <data-clustering/concrete/LocalClusteringStrategy.hpp>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utilities/ErrorHandler.hpp>

using namespace std;
using namespace nlopt;

using namespace vecchia::api;
using namespace vecchia::configurations;
using namespace vecchia::generators;
using namespace vecchia::clustering;
using namespace vecchia::dataunits;
using namespace vecchia::common;
using namespace vecchia::conditioningupdater;
using namespace vecchia::estimators;
using namespace vecchia::predictors;
using namespace vecchia::helpers;

template<typename T>
void VecchiaGP<T>::VecchiaLoadData(Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData) {
    
    int seed = aConfigurations.GetSeed();
    std::srand(seed);
    aConfigurations.PrintSummary();
    LOGGER("** VecchiaGP data generation/loading **")
    // Register and create a kernel object
    kernels::Kernel<T> *pKernel = plugins::PluginRegistry<kernels::Kernel<T>>::Create(aConfigurations.GetKernelName(),
                                                                                      aConfigurations.GetTimeSlot());
    // Create a unique pointer to a DataGenerator object
    unique_ptr<DataGenerator<T>> data_generator = DataGenerator<T>::CreateGenerator(aConfigurations);
    aData = data_generator->CreateData(aConfigurations, *pKernel);
    delete pKernel;


    // Perform clustering based on Vecchia type
    if(aConfigurations.GetVecchiaType() == common::PARALLEL_VECCHIA_GP) {
        // Scalar Vecchia: No clustering, simple point-wise processing
        int n = aConfigurations.GetProblemSize();
        int cs = aConfigurations.GetConditioningSize();
        int batch_count = n - cs + 1;
        aData->SetBatchCount(batch_count);
        LOGGER("-----------Total batch count: " << batch_count << "------------");
        LOGGER("--------------GPUs tobe used: " << aConfigurations.GetGPUsNumbers() << "---------------");
        // Reorder the main locations and observations together
        LocationSorter<T>::GetInstance().ApplyReordering(
            aConfigurations.GetPermutation(), 
            aConfigurations.GetProblemSize(), 
            aConfigurations.GetDimension(), 
            *aData->GetLocations(), 
            aData->GetHostObservations());
    }
    else if(aConfigurations.GetVecchiaType() == common::PARALLEL_BLOCK_VECCHIA_GP){
        // Create clustering strategy using factory
        auto clusteringStrategy = ClusteringFactory::Create<T>(
            aConfigurations.GetVecchiaType(), 
            aConfigurations);
        
        // Perform clustering on training locations
        auto clusteringResult = clusteringStrategy->ComputeClusters(
            *aData->GetLocations(), 
            aConfigurations);
        
        // Process clustering results and prepare batches using utility class
        BatchPreparationUtility<T>::PrepareBatchesFromClustering(
            aConfigurations, *aData, clusteringResult);

    }
    else if(aConfigurations.GetVecchiaType() == common::PARALLEL_SCALED_BLOCK_VECCHIA_GP){
        // Create clustering strategy using factory
        auto clusteringStrategy = ClusteringFactory::Create<T>(
            aConfigurations.GetVecchiaType(), 
            aConfigurations);
        
        // Perform clustering
        auto clusteringResult = clusteringStrategy->ComputeClusters(
            *aData->GetLocations(), 
            aConfigurations);
        
        // Store BlockInfo in VecchiaGBData for use by ScaledBlockEstimator
        aData->SetBlockInfos(clusteringResult.blockInfos);
        aData->SetBlockInfos_test(clusteringResult.blockInfos_test);
    }
    
    // For Block Vecchia prediction: Load and cluster test locations if provided
    if(aConfigurations.GetVecchiaType() == common::PARALLEL_BLOCK_VECCHIA_GP) {
        std::string testLocsFile = aConfigurations.GetTestLocationsPath();
        if (!testLocsFile.empty()) {
            LOGGER("Loading and clustering test locations from CSV for prediction...")
            
            // Perform clustering on test locations using exact code from Predict
            int k = aConfigurations.GetBlockSize();
            int kmeans_iter = aConfigurations.GetKMeansMaxIter();
            int seed = aConfigurations.GetSeed();
            
            LocalClusteringStrategy<T> clusteringStrategy(kmeans_iter, k, seed);
            ClusteringResult<T> testClusteringResult = clusteringStrategy.ComputeClustersForPrediction(
                aConfigurations);
            
            // Store clustering result in VecchiaGBData
            aData->SetTestClusteringResult(testClusteringResult);
            LOGGER("Test locations clustering completed and stored in VecchiaGBData")
        }
    }
    
    LOGGER("\t*Data generation/loading finished*")
}

template<typename T>
T VecchiaGP<T>::VecchiaDataEstimation(Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData, T *apMeasurementsMatrix) {
    
    LOGGER("** Vecchia Data Estimation **")

    if (aConfigurations.GetVecchiaType() == common::PARALLEL_VECCHIA_GP) {
        LOGGER("------Using the Parallel Vecchia Method------")
    } else if (aConfigurations.GetVecchiaType() == common::PARALLEL_BLOCK_VECCHIA_GP){
        LOGGER("--------Using the Block Vecchia Method--------")
    }
    else {
        LOGGER("------Using the Scaled Block Tile Method------")
    }
    // Create kernel object for covariance computation
    kernels::Kernel<T> *pKernel = plugins::PluginRegistry<kernels::Kernel<T>>::Create(
        aConfigurations.GetKernelName(),
        aConfigurations.GetTimeSlot()
    );

    // Create a unique pointer to a ConditioningUpdater object (skip for Scaled Block)
    if (aConfigurations.GetVecchiaType() != common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
        std::unique_ptr<ConditioningUpdater<T>> updater;
        if (aConfigurations.GetIsKNN()){
            updater = std::make_unique<knn::KnnConditioningUpdater<T>>(aConfigurations, *aData);
        }
        else{
            updater = std::make_unique<sequential::SequentialConditioningUpdater<T>>(aConfigurations, *aData);
        }

#pragma omp parallel for
        for (int i = 1; i < aData->GetBatchCount(); ++i){
            updater->Update(aConfigurations, *aData, i);
        }
        // TODO: this should be enabled by a flag
        // updater->SaveClusterAndNeighborFiles(aConfigurations, *aData);
        LOGGER("-----------Nearest Neighbor Done--------------")
    }
    int parameters_number = pKernel->GetParametersNumbers();
    int max_number_of_iterations = aConfigurations.GetMaxMleIterations();
    auto estimator = EstimatorFactory<T>::CreateEstimator(aConfigurations.GetVecchiaType());
    estimator->InitMemory(aConfigurations, aData);
    
    // Note: For ScaledBlock mode, we keep the full parameter vector
    // including all range parameters (sigma2 + nugget + range[dim])
    // The kernel's GetParametersNumbers() only returns base parameters (sigma2 + nugget)
    // but ScaledBlock needs the full parameter set
    
    // For ScaledBlock, adjust parameters_number to include range parameters
    if (aConfigurations.GetVecchiaType() == common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
        int dim = aConfigurations.GetDimensionSize();
        parameters_number = 2 + dim;  // sigma2 + nugget + range[dim]
        
        // Apply kernel-specific bounds for sigma2 and nugget (first 2 parameters)
        // This sets tighter bounds to match the old code's successful range
        aConfigurations.ApplyKernelSpecificBounds();
    }
    
    // Setting struct of data to pass to the modeling.
    auto modeling_data = new mModelingData(aData, aConfigurations, *apMeasurementsMatrix, *pKernel);
    
    struct timespec start_whole, end_whole;
    double whole_time = 0;
    clock_gettime(CLOCK_MONOTONIC, &start_whole);
    
    // Create nlopt
    double opt_f;
    opt optimizing_function(nlopt::LN_BOBYQA, parameters_number);
    
    // Initialize problem's bound.
    optimizing_function.set_lower_bounds(aConfigurations.GetLowerBounds());
    optimizing_function.set_upper_bounds(aConfigurations.GetUpperBounds());
    optimizing_function.set_ftol_abs(aConfigurations.GetTolerance());
    // Set max iterations value.
    optimizing_function.set_maxeval(max_number_of_iterations);
    optimizing_function.set_max_objective(VecchiaMLETileAPI, (void *) modeling_data);

    // Optimize mle using nlopt.
    optimizing_function.optimize(aConfigurations.GetInitialTheta(), opt_f);
    aConfigurations.SetEstimatedTheta(aConfigurations.GetInitialTheta());
    
    auto theta = aConfigurations.GetInitialTheta();

    LOGGER("--> Final Theta Values (", true)
    for (int i = 0; i < parameters_number; i++) {
        LOGGER_PRECISION(theta[i])
        if (i != parameters_number - 1) {
            LOGGER_PRECISION(", ")
        }
    }
    LOGGER_PRECISION(")")
    LOGGER("")
    
    LOGGER("--> Final Log-Likelihood: ", true)
    LOGGER_PRECISION(opt_f)
    LOGGER("")

    clock_gettime(CLOCK_MONOTONIC, &end_whole);
    whole_time = end_whole.tv_sec - start_whole.tv_sec + (end_whole.tv_nsec - start_whole.tv_nsec) / 1e9;
    LOGGER("Total Optimization Time = ", true);
    LOGGER_PRECISION(whole_time)
    LOGGER_PRECISION(" seconds")
    LOGGER("")
    delete pKernel;
    delete modeling_data;
    return opt_f;
}

template<typename T>
double VecchiaGP<T>::VecchiaMLETileAPI(const std::vector<double> &aTheta, std::vector<double> &aGrad, void *apInfo) {

    auto config = ((mModelingData<T> *) apInfo)->mpConfiguration;
    auto data = ((mModelingData<T> *) apInfo)->mpData;
    auto measurements = ((mModelingData<T> *) apInfo)->mpMeasurementsMatrix;
    auto kernel = ((mModelingData<T> *) apInfo)->mpKernel;

    auto estimator = EstimatorFactory<T>::CreateEstimator(config->GetVecchiaType());
    return estimator->Estimate(*config, *data, aTheta.data());
}

template<typename T>
void VecchiaGP<T>::VecchiaPrediction(Configurations &aConfigurations, std::unique_ptr<VecchiaGBData<T>> &aData, T *apMeasurementsMatrix) {
    LOGGER("** Vecchia Data Prediction **")

    if (aConfigurations.GetVecchiaType() == common::PARALLEL_BLOCK_VECCHIA_GP){
        LOGGER("--------Using the Block Vecchia Method--------")
    }
    else {
        LOGGER("------Using the Scaled Block Tile Method------")
    }
    
    // Create kernel object for covariance computation
    kernels::Kernel<T> *pKernel = plugins::PluginRegistry<kernels::Kernel<T>>::Create(
        aConfigurations.GetKernelName(),
        aConfigurations.GetTimeSlot()
    );

    int parameters_number = pKernel->GetParametersNumbers();
    int max_number_of_iterations = aConfigurations.GetMaxMleIterations();
    auto predictor = PredictorFactory<T>::CreatePredictor(aConfigurations.GetVecchiaType());
    predictor->InitMemory(aConfigurations, aData);
    
    // Note: For ScaledBlock mode, we keep the full parameter vector
    // including all range parameters (sigma2 + nugget + range[dim])
    // The kernel's GetParametersNumbers() only returns base parameters (sigma2 + nugget)
    // but ScaledBlock needs the full parameter set
    
    // For ScaledBlock, adjust parameters_number to include range parameters
    if (aConfigurations.GetVecchiaType() == common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
        int dim = aConfigurations.GetDimensionSize();
        parameters_number = 2 + dim;  // sigma2 + nugget + range[dim]
        
        // Apply kernel-specific bounds for sigma2 and nugget (first 2 parameters)
        // This sets tighter bounds to match the old code's successful range
        aConfigurations.ApplyKernelSpecificBounds();
    }
    
    // Get estimated theta from configurations (use initial if not estimated yet)
    auto theta = aConfigurations.GetEstimatedTheta();
    if (theta.empty()) {
        theta = aConfigurations.GetInitialTheta();
    }

    
    struct timespec start_whole, end_whole;
    double whole_time = 0;
    clock_gettime(CLOCK_MONOTONIC, &start_whole);

    LOGGER("--> Using Theta Values for Prediction (", true)
    for (size_t i = 0; i < theta.size(); i++) {
        LOGGER_PRECISION(theta[i])
        if (i != theta.size() - 1) {
            LOGGER_PRECISION(", ")
        }
    }
    LOGGER_PRECISION(")")
    LOGGER("")
    
    // Perform prediction
    T result = predictor->Predict(aConfigurations, aData, theta.data());
    
    clock_gettime(CLOCK_MONOTONIC, &end_whole);
    whole_time = end_whole.tv_sec - start_whole.tv_sec + (end_whole.tv_nsec - start_whole.tv_nsec) / 1e9;
    LOGGER("Total Prediction Time = ", true);
    LOGGER_PRECISION(whole_time)
    LOGGER_PRECISION(" seconds")
    LOGGER("")
    
    LOGGER("--> Prediction Result: ", true)
    LOGGER_PRECISION(result)
    LOGGER("")
    
    delete pKernel;
}
