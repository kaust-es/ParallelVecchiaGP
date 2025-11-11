// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file MLEOptimizer.hpp
 * @brief Handles Maximum Likelihood Estimation optimization workflow
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @date 2025-10-04
**/

#ifndef VECCHIAGP_MLEOPTIMIZER_HPP
#define VECCHIAGP_MLEOPTIMIZER_HPP

#include <vector>
#include <nlopt.hpp>
#include <configurations/Configurations.hpp>
#include <data-units/VecchiaGBData.hpp>
#include <kernels/Kernel.hpp>

namespace vecchia::helpers {

    /**
     * @struct MLEResult
     * @brief Holds the results of MLE optimization
     */
    struct MLEResult {
        std::vector<double> optimalTheta;
        double logLikelihood;
        int iterations;
        bool converged;
        double timeElapsed;
    };

    /**
     * @class MLEOptimizer
     * @brief Performs Maximum Likelihood Estimation using NLOPT
     * @tparam T Data Type: float or double
     */
    template<typename T>
    class MLEOptimizer {
    public:
        /**
         * @brief Constructor
         * @param[in] aConfigurations Configuration parameters
         * @param[in] aData VecchiaGB data structure
         * @param[in] aKernel Covariance kernel
         */
        MLEOptimizer(configurations::Configurations &aConfigurations,
                    ::VecchiaGBData<T> &aData,
                    kernels::Kernel<T> &aKernel);

        /**
         * @brief Destructor
         */
        ~MLEOptimizer();

        /**
         * @brief Performs the MLE optimization
         * @return MLEResult structure containing optimization results
         */
        MLEResult Optimize();

        /**
         * @brief Sets the initial theta values
         * @param[in] aInitialTheta Initial parameter values
         */
        void SetInitialTheta(const std::vector<double> &aInitialTheta);

        /**
         * @brief Sets optimization bounds
         * @param[in] aLowerBounds Lower bounds for parameters
         * @param[in] aUpperBounds Upper bounds for parameters
         */
        void SetBounds(const std::vector<double> &aLowerBounds,
                      const std::vector<double> &aUpperBounds);

        /**
         * @brief Sets optimization tolerance
         * @param[in] aTolerance Relative tolerance for convergence
         */
        void SetTolerance(double aTolerance);

        /**
         * @brief Sets maximum iterations
         * @param[in] aMaxIterations Maximum number of iterations
         */
        void SetMaxIterations(int aMaxIterations);

    private:
        /**
         * @brief Objective function for NLOPT (static wrapper)
         * @param[in] aTheta Current parameter values
         * @param[in] aGrad Gradient vector (if needed)
         * @param[in] apData Pointer to optimizer data
         * @return Log-likelihood value
         */
        static double ObjectiveFunction(const std::vector<double> &aTheta,
                                       std::vector<double> &aGrad,
                                       void *apData);

        /**
         * @brief Computes log-likelihood (instance method)
         * @param[in] aTheta Current parameter values
         * @return Log-likelihood value
         */
        double ComputeLogLikelihood(const std::vector<double> &aTheta);

        /**
         * @brief Initializes MAGMA memory and batch structures
         */
        void InitializeBatchMemory();

        /**
         * @brief Cleans up MAGMA memory
         */
        void CleanupBatchMemory();

        /**
         * @brief Prepares batch operations for current iteration
         * @param[in] aTheta Current parameter values
         */
        void PrepareBatchOperations(const std::vector<double> &aTheta);

        /**
         * @brief Computes covariance matrices for all batches
         * @param[in] aTheta Current parameter values
         */
        void ComputeCovarianceMatrices(const std::vector<double> &aTheta);

        /**
         * @brief Performs Cholesky factorization on batches
         * @return true if successful, false otherwise
         */
        bool PerformCholeskyFactorization();

        /**
         * @brief Computes log-determinant from Cholesky factors
         * @return Sum of log-determinants
         */
        double ComputeLogDeterminant();

        /**
         * @brief Computes quadratic form for likelihood
         * @return Quadratic form value
         */
        double ComputeQuadraticForm();

        // Configuration and data references
        configurations::Configurations &mConfigurations;
        ::VecchiaGBData<T> &mData;
        kernels::Kernel<T> &mKernel;

        // Optimization parameters
        std::vector<double> mInitialTheta;
        std::vector<double> mLowerBounds;
        std::vector<double> mUpperBounds;
        double mTolerance;
        int mMaxIterations;
        int mCurrentIteration;

        // Batch information
        int mBatchCount;
        int mConditioningSize;
        int* mpBatchNum;
        int* mpBatchNumAccum;

        // MAGMA queue
        magma_queue_t mQueue;

        // Memory management flags
        bool mMemoryInitialized;
        bool mMagmaInitialized;
    };

    /**
     * @brief Instantiates the MLEOptimizer class for float and double types.
     * @tparam T Data Type: float or double
     */
    VECCHIAGP_INSTANTIATE_CLASS(MLEOptimizer)

}//namespace vecchia::helpers

#endif //VECCHIAGP_MLEOPTIMIZER_HPP

