// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
* @file Configurations.hpp
* @version 1.0.0
* @brief Contains the declaration of the Configurations class and its member functions.
* @author Mahmoud ElKarargy
* @author Sohayla Abouzeid
* @author Sameh Abdulah
* @date 2025-27-05
**/

#ifndef VECCHIA_CPP_CONFIGURATIONS_HPP
#define VECCHIA_CPP_CONFIGURATIONS_HPP

#include <vector>
#include <unordered_map>
#include <any>

#include <common/Definitions.hpp>

/**
 * @brief Macro that generates a setter function for a member variable.
 * @details This macro generates a function named Set##name that takes an argument of
 * the specified type and sets the member variable with the specified name
 * to the value of the argument. The name of the member variable is used as
 * the key to set the corresponding value in the specified dictionary.
 * @param[in] name The name of the member variable to be set.
 * @param[in] type The data type of the member variable.
 * @param[in] argument_name The name of the argument to the generated function.
 * @param[in] dictionary_name The name of the dictionary to set the value in.
 *
 */
#define CREATE_SETTER_FUNCTION(name, type, argument_name, dictionary_name)  \
void Set##name(type argument_name)                                          \
{                                                                           \
    mDictionary[dictionary_name] = argument_name;                           \
}

/**
 * @brief Macro that generates a getter function for a member variable.
 * @details This macro generates a function named Get##name that returns the value of
 * the member variable with the specified name from the specified dictionary.
 * @param[in] name The name of the member variable to be retrieved.
 * @param[in] type The data type of the member variable.
 * @param[in] dictionary_name The name of the dictionary to retrieve the value from.
 *
 */
#define CREATE_GETTER_FUNCTION(name, type, dictionary_name)                                                 \
type Get##name()                                                                                            \
{                                                                                                           \
    if (mDictionary.find(dictionary_name) == mDictionary.end()) {                                           \
        throw std::range_error(std::string("Argument ").append(dictionary_name).append(" is not set!"));    \
    }                                                                                                       \
    return std::any_cast<type>(mDictionary[dictionary_name]);                                               \
}

namespace vecchia::configurations {
    /**
     * @class Configurations
     * @brief Contains methods to set and get.
     *
     */
    class Configurations{
    public:

        /**
         * @brief Constructor initializing a Configuration object with default values.
         *
         */
        Configurations();

        /**
         * @brief destructor to allow calls to the correct concrete destructor.
         *
         */
        ~Configurations();

        /**
         * @brief Initialize the module arguments.
         * @param[in] aArgC The number of arguments being passed into the program from the command line.
         * @param[in] apArgV The array of arguments.
         * @param[in] aEnableR check if R is enabled
         * @details This method initializes the command line arguments and set default values for unused args.
         * @return void
         *
         */
        void InitializeArguments(const int &aArgC, char **apArgV, const bool &aEnableR = false);


        /**
         * @brief Initialize the all theta arguments.
         * @return void
         *
         */
         void InitializeAllTheta();

        CREATE_SETTER_FUNCTION(VecchiaType, common::VecchiaType, aVecchiaType, "VecchiaType")

        CREATE_GETTER_FUNCTION(VecchiaType, common::VecchiaType, "VecchiaType")

        CREATE_SETTER_FUNCTION(ProblemSize, int, aProblemSize, "ProblemSize")

        CREATE_GETTER_FUNCTION(ProblemSize, int, "ProblemSize")

        CREATE_SETTER_FUNCTION(KernelName, const std::string&, aKernel, "Kernel")

        CREATE_GETTER_FUNCTION(KernelName, const std::string&, "Kernel")

        CREATE_SETTER_FUNCTION(Precision, common::Precision, aPrecision, "Precision")

        CREATE_GETTER_FUNCTION(Precision, common::Precision, "Precision")

        CREATE_SETTER_FUNCTION(CoresNumber, int, aCoresNumbers, "CoresNumbers")

        CREATE_GETTER_FUNCTION(CoresNumber, int, "CoresNumbers")

        CREATE_SETTER_FUNCTION(GPUsNumbers, int, aGPUsNumber, "GPUsNumbers")

        CREATE_GETTER_FUNCTION(GPUsNumbers, int, "GPUsNumbers")

        CREATE_SETTER_FUNCTION(Seed, int, aSeed, "Seed")

        CREATE_GETTER_FUNCTION(Seed, int, "Seed")

        CREATE_SETTER_FUNCTION(InitialTheta, const std::vector<double> &, apTheta, "InitialTheta")

        CREATE_GETTER_FUNCTION(InitialTheta, std::vector<double> &, "InitialTheta")

        CREATE_SETTER_FUNCTION(Dimension, vecchia::common::Dimension, aDimension, "Dimension")

        CREATE_GETTER_FUNCTION(Dimension, vecchia::common::Dimension, "Dimension")

        CREATE_SETTER_FUNCTION(DimensionSize, int, aDimensionSize, "DimensionSize")

        CREATE_GETTER_FUNCTION(DimensionSize, int, "DimensionSize")

        CREATE_SETTER_FUNCTION(KernelType, const std::string&, aKernelType, "KernelType")

        CREATE_GETTER_FUNCTION(KernelType, std::string, "KernelType")

        CREATE_SETTER_FUNCTION(DataPath, const std::string&, aDataPath, "DataPath")

        CREATE_GETTER_FUNCTION(DataPath, std::string, "DataPath")

        CREATE_SETTER_FUNCTION(IsPerformance, bool, aIsPerformance, "IsPerformance")

        CREATE_GETTER_FUNCTION(IsPerformance, bool, "IsPerformance")

        CREATE_SETTER_FUNCTION(IsKNN, bool, aIsKNN, "IsKNN")

        CREATE_GETTER_FUNCTION(IsKNN, bool, "IsKNN")

        CREATE_SETTER_FUNCTION(ConditioningSize, int, aConditioningSize, "ConditioningSize")

        CREATE_GETTER_FUNCTION(ConditioningSize, int, "ConditioningSize")

        CREATE_SETTER_FUNCTION(BlockSize, int, aBlockSize, "BlockSize")

        CREATE_GETTER_FUNCTION(BlockSize, int, "BlockSize")

        CREATE_SETTER_FUNCTION(ObservationsFilePath, const std::string &, aObservationsFilePath, "ObservationsFilePath")

        CREATE_GETTER_FUNCTION(ObservationsFilePath, std::string, "ObservationsFilePath")

        CREATE_SETTER_FUNCTION(TimeSlot, int, aTimeSlot, "TimeSlot")

        CREATE_GETTER_FUNCTION(TimeSlot, int, "TimeSlot")

        CREATE_SETTER_FUNCTION(LowerBounds, const std::vector<double> &, apTheta, "LowerBounds")

        CREATE_GETTER_FUNCTION(LowerBounds, std::vector<double> &, "LowerBounds")

        CREATE_SETTER_FUNCTION(UpperBounds, const std::vector<double> &, apTheta, "UpperBounds")

        CREATE_GETTER_FUNCTION(UpperBounds, std::vector<double> &, "UpperBounds")

        CREATE_SETTER_FUNCTION(EstimatedTheta, const std::vector<double> &, apTheta, "EstimatedTheta")

        CREATE_GETTER_FUNCTION(EstimatedTheta, std::vector<double> &, "EstimatedTheta")

        CREATE_SETTER_FUNCTION(StartingTheta, const std::vector<double> &, apTheta, "StartingTheta")

        CREATE_GETTER_FUNCTION(StartingTheta, std::vector<double> &, "StartingTheta")

        CREATE_SETTER_FUNCTION(IsNonGaussian, bool, aIsNonGaussian, "IsNonGaussian")

        CREATE_GETTER_FUNCTION(IsNonGaussian, bool, "IsNonGaussian")

        CREATE_SETTER_FUNCTION(DistanceMetric, common::DistanceMetric, aDistanceMetric, "DistanceMetric")

        CREATE_GETTER_FUNCTION(DistanceMetric, common::DistanceMetric, "DistanceMetric")

        CREATE_SETTER_FUNCTION(MaxMleIterations, int, aMaxMleIterations, "MaxMleIterations")

        CREATE_GETTER_FUNCTION(MaxMleIterations, int, "MaxMleIterations")

        CREATE_SETTER_FUNCTION(Accuracy, int, aAccuracy, "Accuracy")

        CREATE_GETTER_FUNCTION(Accuracy, int, "Accuracy")

        CREATE_SETTER_FUNCTION(Permutation, common::OrderingMethod, aPermutation, "Permutation")

        CREATE_GETTER_FUNCTION(Permutation, common::OrderingMethod, "Permutation")

        CREATE_SETTER_FUNCTION(PartitionMethod, common::PartitionMethod, aPartitionMethod, "PartitionMethod")

        CREATE_GETTER_FUNCTION(PartitionMethod, common::PartitionMethod, "PartitionMethod")

        CREATE_SETTER_FUNCTION(NNMultiplier, int, aNNMultiplier, "NNMultiplier")

        CREATE_GETTER_FUNCTION(NNMultiplier, int, "NNMultiplier")

        CREATE_SETTER_FUNCTION(DistanceScale, const std::vector<double>&, aDistanceScale, "DistanceScale")

        CREATE_GETTER_FUNCTION(DistanceScale, std::vector<double>&, "DistanceScale")

        CREATE_SETTER_FUNCTION(ClusteringMethod, const std::string&, aClusteringMethod, "ClusteringMethod")

        CREATE_GETTER_FUNCTION(ClusteringMethod, std::string, "ClusteringMethod")

        CREATE_SETTER_FUNCTION(KMeansMaxIter, int, aKMeansMaxIter, "KMeansMaxIter")

        CREATE_GETTER_FUNCTION(KMeansMaxIter, int, "KMeansMaxIter")

        CREATE_SETTER_FUNCTION(TestPointsTotal, int, aTestPointsTotal, "TestPointsTotal")

        CREATE_GETTER_FUNCTION(TestPointsTotal, int, "TestPointsTotal")

        CREATE_SETTER_FUNCTION(TestBlocksTotal, int, aTestBlocksTotal, "TestBlocksTotal")

        CREATE_GETTER_FUNCTION(TestBlocksTotal, int, "TestBlocksTotal")

        CREATE_SETTER_FUNCTION(TestConditioningSize, int, aTestConditioningSize, "TestConditioningSize")

        CREATE_GETTER_FUNCTION(TestConditioningSize, int, "TestConditioningSize")

        CREATE_SETTER_FUNCTION(TrainLocationsPath, const std::string&, aTrainLocationsPath, "TrainLocationsPath")

        CREATE_GETTER_FUNCTION(TrainLocationsPath, std::string, "TrainLocationsPath")

        CREATE_SETTER_FUNCTION(TestLocationsPath, const std::string&, aTestLocationsPath, "TestLocationsPath")

        CREATE_GETTER_FUNCTION(TestLocationsPath, std::string, "TestLocationsPath")

        CREATE_SETTER_FUNCTION(TrainDataPath, const std::string&, aTrainDataPath, "TrainDataPath")

        CREATE_GETTER_FUNCTION(TrainDataPath, std::string, "TrainDataPath")

        CREATE_SETTER_FUNCTION(TestDataPath, const std::string&, aTestDataPath, "TestDataPath")

        CREATE_GETTER_FUNCTION(TestDataPath, std::string, "TestDataPath")

        CREATE_SETTER_FUNCTION(ConditionalSimulations, int, aConditionalSimulations, "ConditionalSimulations")

        CREATE_GETTER_FUNCTION(ConditionalSimulations, int, "ConditionalSimulations")

        CREATE_SETTER_FUNCTION(ScaleFactor, double, aScaleFactor, "ScaleFactor")

        CREATE_GETTER_FUNCTION(ScaleFactor, double, "ScaleFactor")

        void SetTolerance(double aTolerance);

        CREATE_GETTER_FUNCTION(Tolerance, double, "Tolerance")
        
        /**
         * @brief Generate initial theta and bounds based on kernel_type.
         * This function mimics the old Scaled Block Vecchia behavior where kernel type
         * determines parameter structure and automatically generates optimization bounds.
         * @return void
         *
         */
        void GenerateThetaAndBoundsFromKernelType();
        
        /**
         * @brief Apply kernel-specific bounds overrides.
         * This is called by ScaledBlockEstimator when setting up the optimizer.
         * Applies kernel-specific lower/upper bound constraints on the first few parameters.
         * @return void
         *
         */
        void ApplyKernelSpecificBounds();
        
        /**
         * @brief Getter for the verbosity.
         * @return The verbosity mode.
         *
         */
        static vecchia::common::Verbose GetVerbosity();

        /**
         * @brief Setter for the verbosity.
         * @param[in] aVerbose The verbosity mode.
         * @return void
         *
         */
        static void SetVerbosity(const common::Verbose &aVerbose);

        /**
         * @brief Check if input value is numerical.
         * @param[in] aValue The input from the user side.
         * @return The int casted value.
         *
         */
        static int CheckNumericalValue(const std::string &aValue);

        /**
         * @brief Check if input value is a valid Vecchia type.
         * @param[in] aVecchiaType The input from the user side.
         * @return The Vecchia type.
         *
         */
        common::VecchiaType CheckVecchiaTypeValue(const std::string &aVecchiaType);

        /**
         * @brief Checks the value of the dimension parameter.
         * @param[in] aDimension A string represents the dimension.
         * @return The corresponding dimension value.
         *
         */
        static vecchia::common::Dimension CheckDimensionValue(const std::string &aDimension);

        /**
         * @brief Checks if the kernel value is valid.
         * @param[in] aKernel The kernel to check.
         * @return void
         *
         */
        void CheckKernelValue(const std::string &aKernel);

        /**
         * @brief Check input precision value.
         * @param[in] aValue The input from the user side.
         * @return Enum with the selected Precision, Error if not exist.
         *
         */
        static common::Precision CheckPrecisionValue(const std::string &aValue);

        /**
         * @brief Initialize a vector with a given size to contain zeros.
         * @param[in, out] aTheta A reference to the vector to initialize.
         * @param[in] aSize The size of the vector to initialize.
         * @return void.
         *
         */
        static void InitTheta(std::vector<double> &aTheta, const int &aSize);

        /**
         * @brief print the summary of MLE inputs.
         * @return void
         *
         */
        void PrintSummary();

        /**
         * @brief Print the usage and accepted Arguments.
         * @return void
         *
         */
        static void PrintUsage();
        /**
         * @brief Parses a string of theta values and returns an array of doubles.
         * @param[in] aInputValues The input string of theta values.
         * @return A vector of parsed theta values.
         *
         */
        static std::vector<double> ParseTheta(const std::string &aInputValues);

        /**
         * @brief parse user's input to distance metric.
         * @param[in] aDistanceMetric string specifying the used distance metric.
         * @return void
         *
         */
        void ParseDistanceMetric(const std::string &aDistanceMetric);

        /**
         * @brief parse user's input to ordering method.
         * @param[in] aOrderingMethod string specifying the used ordering method.
         * @return void
         *
         */
        void ParseOrderingMethod(const std::string &aOrderingMethod);

        /**
         * @brief parse user's input to partition method.
         * @param[in] aPartitionMethod string specifying the used partition method.
         * @return void
         *
         */
        void ParsePartitionMethod(const std::string &aPartitionMethod);

    private:

        /**
         * @brief Checks the run mode and sets the verbosity level.
         * @param[in] aVerbosity A string represents the desired run mode ("verbose" or "standard").
         * @throws std::range_error if the input string is not "verbose" or "standard".
         * @return void
         *
         */
        static void ParseVerbose(const std::string &aVerbosity);

        /**
         * @brief Checks if a given string is in camel case format.
         * @param[in] aString The string to check.
         * @return true if the string is in camel case format, false otherwise.
         *
         */
        static bool IsCamelCase(const std::string &aString);

        /// Used Dictionary
        std::unordered_map<std::string, std::any> mDictionary;
        /// Used Argument counter
        int mArgC = 0;
        /// Used Argument vectors
        char **mpArgV = nullptr;
        //// Used run mode
        static vecchia::common::Verbose mVerbosity;
        //// Used bool for init theta
        static bool mIsThetaInit;
        //// Used bool for R allocated memory on heap
        static bool mHeapAllocated;
        //// Used bool for indicating the first init of configurations for printing summary
        static bool mFirstInit;
    };
}//namespace vecchia

#endif //VECCHIA_CPP_CONFIGURATIONS_HPP