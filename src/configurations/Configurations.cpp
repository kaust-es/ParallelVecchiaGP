
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file Configurations.cpp
 * @brief This file defines the Configurations class which stores the configuration parameters for VecchiaGB.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @date 2025-09-29
**/

#include <algorithm>
#include <cstring>
#include <cmath>

#include <configurations/Configurations.hpp>
#include <utilities/Logger.hpp>
#include <kernels/Kernel.hpp>

using namespace std;

using namespace vecchia::configurations;
using namespace vecchia::common;

Verbose Configurations::mVerbosity = Verbose::STANDARD_MODE;
bool Configurations::mIsThetaInit = false;
bool Configurations::mHeapAllocated = false;
bool Configurations::mFirstInit = false;

Configurations::Configurations() {

    // Set default values for arguments!
    SetVecchiaType(PARALLEL_BLOCK_VECCHIA_GP);
    SetProblemSize(2000);
    SetDimension(Dimension2D);  // For Parallel/Block Vecchia
    SetDimensionSize(2);  // For Scaled Block Vecchia (N-dimensional)
    SetKernelType("Matern72");  // For Scaled Block Vecchia kernel specification
    SetIsKNN(true);
    SetPermutation(OrderingMethod::RANDOM);
    SetPartitionMethod(LINEAR_PARTITION);  // Default partition method
    SetCoresNumber(40);
    SetConditioningSize(100);
    SetBlockSize(200);
    SetLowerBounds(ParseTheta("0.01:0.01:0.01"));
    SetUpperBounds(ParseTheta("3:3:3"));

    SetMaxMleIterations(1);
    SetGPUsNumbers(0);
    SetIsKNN(false);
    SetIsPerformance(false);
    SetKernelName("Matern");
    SetProblemSize(0);
    SetTolerance(5);
    SetSeed(static_cast<unsigned int>(time(0)));
    vector<double> theta;
    SetInitialTheta(theta);
    SetEstimatedTheta(theta);
    SetObservationsFilePath("");
    SetPrecision(DOUBLE);
    SetDataPath("");
    SetTimeSlot(1);
    SetDistanceMetric(EUCLIDEAN_DISTANCE);  // Default distance metric
    SetIsNonGaussian(false);
    
    // Block & Scaled Block parameters
    SetNNMultiplier(400);  // Default from ScaledBlock paper
    SetClusteringMethod("random");  // Fast default
    SetKMeansMaxIter(50);  // Quick convergence
    vector<double> empty_scale;
    SetDistanceScale(empty_scale);  // Empty = uniform scaling
    
    // Test/Prediction parameters (for Scaled Block prediction mode)
    SetTestPointsTotal(0); 
    SetTestBlocksTotal(0);  
    SetTestConditioningSize(0);  // Default m_test
    
    // CSV file paths for train/test data
    SetTrainLocationsPath("");
    SetTestLocationsPath("");
    SetTrainDataPath("");
    SetTestDataPath("");
    
    // Prediction parameters
    SetConditionalSimulations(1000);  // Default number of conditional simulations
    SetScaleFactor(1.0);  // Default scale factor
    
    mIsThetaInit = false;
}

void Configurations::InitializeArguments(const int &aArgC, char **apArgV, const bool &aEnableR) {

    this->mArgC = aArgC;
    this->mpArgV = apArgV;
    mHeapAllocated = aEnableR;

    // Get the example name
    string example_name = apArgV[0];
    // Remove the './'
    example_name.erase(0, 2);
    string argument;
    string argument_name;
    string argument_value;
    int equal_sign_Idx;

    // Loop through the arguments
    for (int i = 1; i < aArgC; ++i) {
        argument = apArgV[i];
        equal_sign_Idx = static_cast<int>(argument.find('='));
        argument_name = argument.substr(0, equal_sign_Idx);

        // Check if argument has an equal sign.
        if (equal_sign_Idx != string::npos) {
            argument_value = argument.substr(equal_sign_Idx + 1);

            // Check the argument name and set the corresponding value
            if (argument_name == "--N" || argument_name == "--n") {
                SetProblemSize(CheckNumericalValue(argument_value));
            } else if (argument_name == "--Kernel" || argument_name == "--kernel") {
                CheckKernelValue(argument_value);
            } else if (argument_name == "--VecchiaType" || argument_name == "--vecchiaType") {
                SetVecchiaType(CheckVecchiaTypeValue(argument_value));
            } else if (argument_name == "--dimension" || argument_name == "--Dimension") {
                // For Parallel/Block Vecchia: 2D, 3D, ST
                SetDimension(CheckDimensionValue(argument_value));
                // Also set integer dimension for consistency
                if (GetDimension() == Dimension2D) SetDimensionSize(2);
                else if (GetDimension() == Dimension3D) SetDimensionSize(3);
                else if (GetDimension() == DimensionST) SetDimensionSize(3);
            } else if (argument_name == "--dim" || argument_name == "--Dim") {
                // For Scaled Block Vecchia: integer dimension (e.g., 10 for 10D)
                int dim_value = CheckNumericalValue(argument_value);
                SetDimensionSize(dim_value);
                // Map to enum for compatibility (Scaled Block will use DimensionSize)
                if (dim_value == 2) SetDimension(Dimension2D);
                else if (dim_value == 3) SetDimension(Dimension3D);
                else SetDimension(Dimension2D);  // Default fallback
            } else if (argument_name == "--precision" || argument_name == "--Precision") {
                SetPrecision(CheckPrecisionValue(argument_value));
            } else if (argument_name == "--cores" || argument_name == "--coresNumber" ||
                       argument_name == "--cores_number" || argument_name == "--ncores") {
                SetCoresNumber(CheckNumericalValue(argument_value));
            } else if (argument_name == "--gpus" || argument_name == "--GPUsNumbers" ||
                       argument_name == "--gpu_number" || argument_name == "--ngpus") {
                SetGPUsNumbers(CheckNumericalValue(argument_value));
            } else if (argument_name == "--initial_theta" || argument_name == "--itheta" ||
                       argument_name == "--iTheta" || argument_name == "--theta") {
                vector<double> theta = ParseTheta(argument_value);
                SetInitialTheta(theta);
            } else if (argument_name == "--observations_path" || argument_name == "--observationsPath" ||
                       argument_name == "--observationspath") {
                SetObservationsFilePath(argument_value);
            } else if (argument_name == "--Seed" || argument_name == "--seed") {
                SetSeed(CheckNumericalValue(argument_value));
            } else if (argument_name == "--verbose" || argument_name == "--Verbose") {
                ParseVerbose(argument_value);
            } else if (argument_name == "--conditioning_size") {
                SetConditioningSize(CheckNumericalValue(argument_value));
            } else if (argument_name == "--block_size") {
                SetBlockSize(CheckNumericalValue(argument_value));
            } else if (argument_name == "--tolerance") {
                SetTolerance(CheckNumericalValue(argument_value));
            } else if (argument_name == "--DataPath" || argument_name == "--dataPath" ||
                argument_name == "--data_path") {
                SetDataPath(argument_value);
            } else if (argument_name == "--permutation") {
                ParseOrderingMethod(argument_value);
            } else if (argument_name == "--partition") {
                ParsePartitionMethod(argument_value);
            } else if (argument_name == "--distance_metric") {
                ParseDistanceMetric(argument_value);
            } else if (argument_name == "--max_mle_iterations") {
                SetMaxMleIterations(CheckNumericalValue(argument_value));
            } else if (argument_name == "--accuracy") {
                SetAccuracy(CheckNumericalValue(argument_value));
            } else if (argument_name == "--time_slot") {
                SetTimeSlot(CheckNumericalValue(argument_value));
            } else if (argument_name == "--lower_bounds" || argument_name == "--lowerBounds" ||
                       argument_name == "--lower_bounds") {
                vector<double> theta = ParseTheta(argument_value);
                SetLowerBounds(theta);
                SetStartingTheta(theta);
            } else if (argument_name == "--upper_bounds") {
                SetUpperBounds(ParseTheta(argument_value));
            } else if (argument_name == "--upper_bounds" || argument_name == "--upperBounds" ||
                       argument_name == "--upper_bounds") {
                SetUpperBounds(ParseTheta(argument_value));
            } else if (argument_name == "--estimated_theta" || argument_name == "--etheta" ||
                argument_name == "--eTheta") {
                vector<double> theta = ParseTheta(argument_value);
                SetEstimatedTheta(theta);
            } else if (argument_name == "--nn_multiplier" || argument_name == "--NNMultiplier") {
                SetNNMultiplier(CheckNumericalValue(argument_value));
            } else if (argument_name == "--distance_scale" || argument_name == "--distanceScale") {
                vector<double> scale = ParseTheta(argument_value);
                SetDistanceScale(scale);
            } else if (argument_name == "--clustering_method" || argument_name == "--clusteringMethod") {
                SetClusteringMethod(argument_value);
            } else if (argument_name == "--kmeans_max_iter" || argument_name == "--kmeansMaxIter") {
                SetKMeansMaxIter(CheckNumericalValue(argument_value));
            } else if (argument_name == "--num_total_points_test" || argument_name == "--test_points") {
                SetTestPointsTotal(CheckNumericalValue(argument_value));
            } else if (argument_name == "--num_total_blocks_test" || argument_name == "--test_blocks") {
                SetTestBlocksTotal(CheckNumericalValue(argument_value));
            } else if (argument_name == "--m_test" || argument_name == "--test_conditioning_size") {
                SetTestConditioningSize(CheckNumericalValue(argument_value));
            } else if (argument_name == "--kernel_type" || argument_name == "--kernelType") {
                // For Scaled Block: Matern12, Matern32, Matern52, Matern72, PowerExponential
                SetKernelType(argument_value);
            } else if (argument_name == "--train_locs" || argument_name == "--trainLocationsPath") {
                SetTrainLocationsPath(argument_value);
            } else if (argument_name == "--test_locs" || argument_name == "--testLocationsPath") {
                SetTestLocationsPath(argument_value);
            } else if (argument_name == "--train_data" || argument_name == "--trainDataPath") {
                SetTrainDataPath(argument_value);
            } else if (argument_name == "--test_data" || argument_name == "--testDataPath") {
                SetTestDataPath(argument_value);
            } else if (argument_name == "--conditional_sim" || argument_name == "--conditionalSimulations") {
                SetConditionalSimulations(CheckNumericalValue(argument_value));
            } else if (argument_name == "--scale_factor" || argument_name == "--scaleFactor") {
                // Parse as double
                try {
                    SetScaleFactor(stod(argument_value));
                } catch (...) {
                    throw range_error("Invalid value for scale_factor. Please use a numerical value.");
                }
            } else {
                throw invalid_argument(
                        "This argument is undefined, Please use --help to print all available arguments");
            }
        } else {
            if (argument_name == "--help") {
                PrintUsage();
            } else if (argument_name == "--performance") {
                SetIsPerformance(true);
            } else if (argument_name == "--knn") {
                SetIsKNN(true);
            } else {
                LOGGER("!! " << argument_name << " !!")
                throw invalid_argument(
                        "This argument is undefined, Please use --help to print all available arguments");
            }
        }
    }

    // Throw Errors if any of these arguments aren't given by the user.
    if (GetProblemSize() == 0) {
        throw domain_error("You need to set the problem size, before starting");
    }
    // Throw Errors if any of these arguments aren't given by the user.
    if (GetKernelName().empty()) {
        throw domain_error("You need to set the Kernel, before starting");
    }

    size_t found = GetKernelName().find("NonGaussian");
    // Check if the substring was found
    if (found != std::string::npos) {
        SetIsNonGaussian(true);
    }

    // For Scaled Block Vecchia: Generate theta and bounds from kernel_type
    // This must be done AFTER all arguments are parsed, BEFORE InitializeAllTheta()
    if (GetVecchiaType() == PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
        GenerateThetaAndBoundsFromKernelType();
    }

    this->InitializeAllTheta();
}


void Configurations::PrintUsage() {
    LOGGER("\n\t*** Available Arguments For VecchiaGP Configurations ***")
    LOGGER("--help : Display this help message and exit")
    LOGGER("--N=value : [int] The number of locations, e.g., 2000")
    LOGGER("--itheta=value : The initial values of parameters in kernel, sigma^2:range:smooth, e.g., 1.5:0.1:0.5")
    LOGGER("--kernel=value : The name of kernels, such as matern kernel, e.g., univariate_matern_stationary")
    LOGGER("--block_size=value : [int] The block count in Vecchia method (number of clusters), e.g., 300")
    LOGGER("--conditioning_size=value : [int] The conditioning size in Vecchia method, e.g., 1500")
    LOGGER("--knn : nearest neighbors searching, default to use.")
    LOGGER("--performance : Only calculate the one iteraion of block/classic Vecchia and obs=0.")
    LOGGER("--seed=value : [int] random generation for locations and observations.")
    LOGGER("--dimension=value : [2D, 3D, ST] Dimension for Parallel/Block Vecchia.")
    LOGGER("--dim=value : [int] Integer dimension for Scaled Block Vecchia, e.g., 10 for 10D data.")
    LOGGER("--distance_metric : [eg, gcd] Used to set the distance metric.")
    LOGGER("--data_path : [string] locations path.")
    LOGGER("--observations_path : [string] observations path.")
    LOGGER("--tolerance : [int] tolerance of BOBYQA, 5 -> 1e-5.")
    LOGGER("--time_slot : [int] The slot of time, e.g., 10, 20, satisfying num_loc % t_slots = 0.")
    LOGGER("--cores=value : [int] number openmp threads, default 40.")
    LOGGER("--permutation : [string] reordering method, default as random, (optional) kdtree, morton, hilber, mmd.")
    LOGGER("--partition : [string] partition type (linear or none), default as linear.")
    LOGGER("--vecchiaType=value : [parallel, block, scaled_block] Used Vecchia Type.")
    LOGGER("--precision=value : [single, double, mixed] Used precision.")
    LOGGER("--verbose=value : [quiet, standard, detailed] Run mode with different verbosity.")
    LOGGER("--lower_bounds : [sigma^2:range:smooth] Used to set the lower bounds, default as 0.01:0.01:0.01.")
    LOGGER("--upper_bounds : [sigma^2:range:smooth] Used to set the upper bounds, default as 3:3:3.")
    LOGGER("--nn_multiplier=value : [int] NN multiplier for coarse-to-fine search (Scaled Block), default 400.")
    LOGGER("--distance_scale=value : [d1:d2:d3:...] Per-dimension distance scaling (Scaled Block), e.g., 0.05:0.05:0.1:1.0")
    LOGGER("--clustering_method=value : [random, kmeans++] Clustering method for Block/Scaled Block Vecchia, default random.")
    LOGGER("--kmeans_max_iter=value : [int] Max iterations for k-means clustering, default 10.")
    LOGGER("--num_total_points_test=value : [int] Total number of test points (for prediction mode), default 0.")
    LOGGER("--num_total_blocks_test=value : [int] Total number of test blocks (for prediction mode), default 0.")
    LOGGER("--m_test=value : [int] Number of nearest neighbors for test data, default 120.")
    LOGGER("--kernel_type=value : [Matern12, Matern32, Matern52, Matern72, PowerExponential] Kernel type for Scaled Block, default Matern72.")
    LOGGER("  Note: kernel_type determines parameter structure and auto-generates bounds:")
    LOGGER("    Matern12/32/52/72: [variance, nugget] + [distance_scale per dimension]")
    LOGGER("    PowerExponential: [variance, smoothness, nugget] + [distance_scale per dimension]")
    LOGGER("--train_locs=value : [string] Path to training locations CSV file.")
    LOGGER("--test_locs=value : [string] Path to test locations CSV file.")
    LOGGER("--train_data=value : [string] Path to training data CSV file.")
    LOGGER("--test_data=value : [string] Path to test data CSV file.")
    LOGGER("--conditional_sim=value : [int] Number of conditional simulations, default 1000.")
    LOGGER("--scale_factor=value : [double] Scale factor for covariance matrix, default 1.0.")
    // TODO: Add more arguments if needed, and is these needed?
    LOGGER("--gpus=value : Used to set the number of GPUs.")
    LOGGER("--max_mle_iterations : Used to set the maximum number of MLE iterations.")
    LOGGER("--accuracy : Used to set the accuracy.")
    LOGGER("\n\n")

    exit(0);
}

void Configurations::InitializeAllTheta() {

    if (!mIsThetaInit) {

        int parameters_number = kernels::KernelsConfigurations::GetParametersNumberKernelMap()[this->GetKernelName()];
        InitTheta(GetInitialTheta(), parameters_number);
        SetInitialTheta(GetInitialTheta());

        if (this->GetIsNonGaussian()) {
            GetInitialTheta()[GetInitialTheta().size() - 1] = 0.2;
            GetInitialTheta()[GetInitialTheta().size() - 2] = 0.2;
        }

        InitTheta(GetLowerBounds(), parameters_number);
        SetLowerBounds(GetLowerBounds());
        InitTheta(GetUpperBounds(), parameters_number);
        SetUpperBounds(GetUpperBounds());
        SetStartingTheta(GetLowerBounds());
        InitTheta(GetEstimatedTheta(), parameters_number);
        SetEstimatedTheta(GetEstimatedTheta());

        for (int i = 0; i < parameters_number; i++) {
            if (GetEstimatedTheta()[i] != -1) {
                GetLowerBounds()[i] = GetEstimatedTheta()[i];
                GetUpperBounds()[i] = GetEstimatedTheta()[i];
                GetStartingTheta()[i] = GetEstimatedTheta()[i];
            }
        }
        mIsThetaInit = true;
    }
}


Verbose Configurations::GetVerbosity() {
    return Configurations::mVerbosity;
}

void Configurations::SetVerbosity(const Verbose &aVerbose) {
    Configurations::mVerbosity = aVerbose;
}

int Configurations::CheckNumericalValue(const string &aValue) {

    int numericalValue;
    try {
        numericalValue = stoi(aValue);
    }
    catch (...) {
        throw range_error("Invalid value. Please use Numerical values only.");
    }

    if (numericalValue < 0) {
        throw range_error("Invalid value. Please use positive values");
    }
    return numericalValue;
}

Precision Configurations::CheckPrecisionValue(const std::string &aValue) {

    if (aValue != "single" and aValue != "Single" and aValue != "double" and aValue != "Double" and aValue != "mix" and
        aValue != "Mix" and aValue != "Mixed" and aValue != "mixed") {
        throw range_error("Invalid value for Computation. Please use Single, Double or Mixed.");
    }
    if (aValue == "single" or aValue == "Single") {
        return SINGLE;
    } else if (aValue == "double" or aValue == "Double") {
        return DOUBLE;
    }
    return MIXED;
}

void Configurations::ParseVerbose(const std::string &aVerbosity) {
    if (aVerbosity == "quiet" || aVerbosity == "Quiet") {
        mVerbosity = Verbose::QUIET_MODE;
    } else if (aVerbosity == "standard" || aVerbosity == "Standard") {
        mVerbosity = Verbose::STANDARD_MODE;
    } else if (aVerbosity == "detailed" || aVerbosity == "Detailed" || aVerbosity == "detail") {
        mVerbosity = Verbose::DETAILED_MODE;
    } else {
        LOGGER("Error: " << aVerbosity << " is not valid ")
        throw range_error("Invalid value. Please use verbose or standard values only.");
    }
}

VecchiaType Configurations::CheckVecchiaTypeValue(const string &aVecchiaType) {

    if (aVecchiaType != "parallel" && aVecchiaType != "block" && aVecchiaType != "scaled_block") {
        throw range_error("Invalid value for VecchiaType. Please check manual.");
    }
    if (aVecchiaType == "parallel") {
        return VecchiaType::PARALLEL_VECCHIA_GP;
    }
    if (aVecchiaType == "block") {
        return VecchiaType::PARALLEL_BLOCK_VECCHIA_GP;
    }
    if (aVecchiaType == "scaled_block") {
        return VecchiaType::PARALLEL_SCALED_BLOCK_VECCHIA_GP;
    }
}

void Configurations::ParseDistanceMetric(const std::string &aDistanceMetric) {
    if (aDistanceMetric == "eg" || aDistanceMetric == "EG" || aDistanceMetric == "euclidean") {
        SetDistanceMetric(EUCLIDEAN_DISTANCE);
    } else if (aDistanceMetric == "gcd" || aDistanceMetric == "GCD" || aDistanceMetric == "great_circle") {
        SetDistanceMetric(GREAT_CIRCLE_DISTANCE);
    } else {
        throw range_error("Invalid value. Please use eg or gcd values only.");
    }
}

void Configurations::ParseOrderingMethod(const std::string &aOrderingMethod) {
    if (aOrderingMethod == "random" || aOrderingMethod == "Random") {
        SetPermutation(OrderingMethod::RANDOM);
    } else if (aOrderingMethod == "morton" || aOrderingMethod == "Morton") {
        SetPermutation(OrderingMethod::MORTON);
    }
    else if (aOrderingMethod == "kdtree" || aOrderingMethod == "KDTREE") {
        SetPermutation(OrderingMethod::KD_TREE);
    } else if (aOrderingMethod == "hilbert" || aOrderingMethod == "Hilbert") {
        SetPermutation(OrderingMethod::HILBERT);
    } else if (aOrderingMethod == "mmd" || aOrderingMethod == "MMD") {
        SetPermutation(OrderingMethod::MMD);
    }
}

void Configurations::ParsePartitionMethod(const std::string &aPartitionMethod) {
    if (aPartitionMethod == "linear" || aPartitionMethod == "Linear" || aPartitionMethod == "LINEAR") {
        SetPartitionMethod(LINEAR_PARTITION);
    } else if (aPartitionMethod == "none" || aPartitionMethod == "None" || aPartitionMethod == "NONE") {
        SetPartitionMethod(NO_PARTITION);
    } else {
        throw range_error("Invalid value for partition method. Please use 'linear' or 'none'.");
    }
}

void Configurations::CheckKernelValue(const string &aKernel) {

    // Check if the kernel name exists in the availableKernels set.
    if (availableKernels.count(aKernel) <= 0) {
        throw range_error("Invalid value for Kernel. Please check manual.");
    }
    // Check if the string is already in CamelCase format
    if (IsCamelCase(aKernel)) {
        this->SetKernelName(aKernel);
        return;
    }
    string str = aKernel;
    // Replace underscores with spaces and split the string into words
    std::replace(str.begin(), str.end(), '_', ' ');
    std::istringstream iss(str);
    std::string word, result;
    while (iss >> word) {
        // Capitalize the first letter of each word and append it to the result
        word[0] = static_cast<char>(toupper(word[0]));
        result += word;
    }
    this->SetKernelName(result);
}

bool Configurations::IsCamelCase(const std::string &aString) {
    // If the string contains an underscore, it is not in CamelCase format
    if (aString.find('_') != std::string::npos) {
        return false;
    }
    // If the string starts with a lowercase letter, it is not in CamelCase format
    if (islower(aString[0])) {
        return false;
    }
    // If none of the above conditions hold, the string is in CamelCase format
    return true;
}

vector<double> Configurations::ParseTheta(const std::string &aInputValues) {
    // Count the number of values in the string (support both : and , delimiters)
    // Check if comma is present (original code format) - prefer comma over colon
    bool uses_comma = (aInputValues.find(',') != std::string::npos);
    
    int num_values = 1;
    char delimiter = uses_comma ? ',' : ':';
    for (char aInputValue: aInputValues) {
        if (aInputValue == delimiter) {
            num_values++;
        }
    }
    // Allocate memory for the array of doubles
    vector<double> theta;

    // Split the string into tokens using strtok()
    // Support both colon (:) and comma (,) delimiters for compatibility
    // Prefer comma if present (matches original code format)
    const char *delim = uses_comma ? "," : ":";
    char *token = strtok((char *) aInputValues.c_str(), delim);
    int i = 0;
    while (token != nullptr) {
        // Check if the token is a valid double or "?"
        if (!strcmp(token, "?")) {
            theta.push_back(-1);
        } else {
            try {
                theta.push_back(stod(token));
            }
            catch (...) {
                LOGGER("Error: " << token << " is not a valid double or '?' ")
                throw range_error("Invalid value. Please use Numerical values only.");
            }
        }

        // Get the next token
        token = strtok(nullptr, delim);
        i++;
    }

    // Check if the number of values in the array is correct
    if (i != num_values) {
        throw range_error(
                "Error: the number of values in the input string is invalid, please use this example format as a reference 1:?:0.1 or 1,?,0.1");
    }

    return theta;
}

Dimension Configurations::CheckDimensionValue(const string &aDimension) {

    if (aDimension != "2D" and aDimension != "2d" and aDimension != "3D" and aDimension != "3d" and
        aDimension != "st" and aDimension != "ST") {
        throw range_error("Invalid value for Dimension. Please use 2D, 3D or ST.");
    }
    if (aDimension == "2D" or aDimension == "2d") {
        return Dimension2D;
    } else if (aDimension == "3D" or aDimension == "3d") {
        return Dimension3D;
    }
    return DimensionST;
}

void Configurations::InitTheta(vector<double> &aTheta, const int &size) {

    // If null, this mean user have not passed the values arguments, Make values equal -1
    if (aTheta.empty()) {
        for (int i = 0; i < size; i++) {
            aTheta.push_back(-1);
        }
    } else if (aTheta.size() < size) {
        // Also allocate new memory as maybe they are not the same size.
        for (size_t i = aTheta.size(); i < size; i++) {
            aTheta.push_back(0);
        }
    }
}

void Configurations::PrintSummary() {

#ifndef USE_R
    Verbose temp = this->GetVerbosity();
    mVerbosity = STANDARD_MODE;

    if (!mFirstInit) {

        LOGGER("********************SUMMARY**********************")
        if (this->GetDataPath().empty()) {
            LOGGER("#Synthetic Data generation")
        } else {
            LOGGER("#Real Data loader")
        }
        LOGGER("#Number of Locations: " << this->GetProblemSize())
        LOGGER("#Threads per node: " << this->GetCoresNumber())
        LOGGER("#GPUs: " << this->GetGPUsNumbers())
        LOGGER("#Precision: " << this->GetPrecision())
        LOGGER("#Vecchia Type: " << this->GetVecchiaType())
        LOGGER("#Conditioning Size: " << this->GetConditioningSize())
        LOGGER("#Tolerance: " << this->GetTolerance())
        LOGGER("#Observations File: " << this->GetObservationsFilePath())

        if (this->GetDimension() == Dimension2D) {
            LOGGER("#Dimension: 2D")
        } else if (this->GetDimension() == Dimension3D) {
            LOGGER("#Dimension: 3D")
        } else if (this->GetDimension() == DimensionST) {
            LOGGER("#Dimension: ST")
        }
        LOGGER("#Kernel: " << this->GetKernelName())
        LOGGER("#KNN: " << this->GetIsKNN())
        LOGGER("#Performance: " << this->GetIsPerformance())
        LOGGER("*************************************************")
        mFirstInit = true;
    }
    mVerbosity = temp;
#endif
}

Configurations::~Configurations() {

    if (mHeapAllocated) {
        for (size_t i = 0; i < this->mArgC; ++i) {
            delete[] this->mpArgV[i];  // Delete each string
        }
        delete[] this->mpArgV;  // Delete the array of pointers
    }
    this->mpArgV = nullptr;
    mFirstInit = false;
}

void Configurations::SetTolerance(double aTolerance) {
    mDictionary["Tolerance"] = pow(10, -1 * aTolerance);
}

void Configurations::GenerateThetaAndBoundsFromKernelType() {
    // This function generates initial theta and bounds based on kernel_type
    // Matches old Scaled Block Vecchia code: input_parser.h lines 137-173
    
    std::string kernel_type = GetKernelType();
    int dim = GetDimensionSize();
    auto distance_scale = GetDistanceScale();
    auto user_theta = GetInitialTheta();
    
    // Ensure distance_scale has correct size (lines 127-131 in old code)
    if (distance_scale.size() != dim) {
        distance_scale = std::vector<double>(dim, 1.0);
        SetDistanceScale(distance_scale);
    }
    
    // Generate kernel parameters based on kernel type (lines 137-166 in old code)
    std::vector<double> theta_init;
    int range_offset = 0;
    
    if (!user_theta.empty()) {
        // User provided theta_init (line 138-139 in old code)
        // For Scaled Block, we still need to format it correctly
        // Take only the variance (and maybe nugget) from user input
        if (kernel_type == "PowerExponential") {
            // variance, smoothness, nugget
            theta_init = {user_theta[0], 
                         user_theta.size() > 1 ? user_theta[1] : 0.5, 
                         user_theta.size() > 2 ? user_theta[2] : 0.0};
            range_offset = 3;
        } else {  // Matern kernels
            // variance, nugget
            theta_init = {user_theta[0], 
                         user_theta.size() > 1 ? user_theta[1] : 0.0};
            range_offset = 2;
        }
    } else {
        // Generate default theta based on kernel type (lines 141-166)
        if (kernel_type == "PowerExponential") {
            theta_init = {1.0, 0.5, 0.00};  // variance, smoothness, nugget
            range_offset = 3;
        } else {  // Matern12/32/52/72
            theta_init = {1.0, 0.00};  // variance, nugget
            range_offset = 2;
        }
    }
    
    // Append distance_scale to theta_init (line 168 in old code)
    theta_init.insert(theta_init.end(), distance_scale.begin(), distance_scale.end());
    SetInitialTheta(theta_init);
    
    // Generate automatic bounds (lines 170-173 in old code)
    // Note: Kernel-specific overrides are NOT applied here
    // They are applied later when setting up the optimizer (in ScaledBlockEstimator)
    std::vector<double> lower_bounds, upper_bounds;
    for (size_t i = 0; i < theta_init.size(); i++) {
        lower_bounds.push_back(theta_init[i] * 0.001);
        upper_bounds.push_back(theta_init[i] * 10.0);
    }
    
    SetLowerBounds(lower_bounds);
    SetUpperBounds(upper_bounds);
}

void Configurations::ApplyKernelSpecificBounds() {
    // This function applies kernel-specific bounds overrides
    // Called by ScaledBlockEstimator when setting up the optimizer
    // (Matches the optimizer setup code in old ParallelScaledBlockVecchiaGP)
    
    std::string kernel_type = GetKernelType();
    auto lower_bounds = GetLowerBounds();
    auto upper_bounds = GetUpperBounds();
    
    if (kernel_type == "PowerExponential") {
        lower_bounds[0] = 0.01;  upper_bounds[0] = 3.0;   // sigma2
        lower_bounds[1] = 0.01;  upper_bounds[1] = 2.0;   // smoothness
        lower_bounds[2] = 0.0;   upper_bounds[2] = 0.1;   // nugget
    } else if (kernel_type == "Matern72") {
        lower_bounds[0] = 0.01;  upper_bounds[0] = 2.0;   // sigma2
        lower_bounds[1] = 0.0;   upper_bounds[1] = 0.1;   // nugget
    } else {  // Matern12/32/52
        lower_bounds[0] = 0.01;  upper_bounds[0] = 3.0;   // sigma2
        lower_bounds[1] = 0.0;   upper_bounds[1] = 0.1;   // nugget
    }
    
    SetLowerBounds(lower_bounds);
    SetUpperBounds(upper_bounds);
}