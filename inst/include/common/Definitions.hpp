
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGP is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file Definitions.hpp
 * @version 1.0.0
 * @brief This file contains common definitions used in VecchiaGP software package.
 * @details These definitions include enums for dimension, computation, precision, and floating point arithmetic;
 * A macro for instantiating template classes with supported types; and a set of available kernels.
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#ifndef VECCHIAGP_DEFINITIONS_HPP
#define VECCHIAGP_DEFINITIONS_HPP

#include <set>
#include <filesystem>

/**
 * @def VECCHIAGP_INSTANTIATE_CLASS
 * @brief Macro definition to instantiate the VECCHIAGP template classes with supported types.
 *
**/
#define VECCHIAGP_INSTANTIATE_CLASS(TEMPLATE_CLASS)   template class TEMPLATE_CLASS<double>; /*template class TEMPLATE_CLASS<float>;  \*/
                                                    

// Variables sizes.
#define SIZE_OF_FLOAT 4
#define SIZE_OF_DOUBLE 8

/**
 * Pi value.
 */
#define PI (3.141592653589793)

/**
 * Earth Radius value.
 */
#define EARTH_RADIUS 6371.0

/**
 * Q Norm value.
 */
#define Q_NORM 1.959964

/**
 * Logging Path Definition
 */
#define LOG_PATH PROJECT_SOURCE_DIR "/logs/"

namespace vecchia::common {


    /** Enum describing the cuda stream behavior (async,sync),
     * not used in the case of CPU Context.  **/
    enum class RunMode {
        SYNC,
        ASYNC
    };

    /** Int Enum to describe whether a RunContext is on GPU or CPU **/
    enum OperationPlacement : int {
        GPU = 0,
        CPU = 1
    };

    /**
     * @enum VerbosityLevel
     * @brief Enum denoting the run mode
     *
     */
    enum Verbose {
        QUIET_MODE = 0,
        STANDARD_MODE = 1,
        DETAILED_MODE = 2
    };

    /**
     * @enum Dimension
     * @brief Enum denoting the dimension of generated data.
     *
     */
    enum Dimension {
        Dimension2D = 0,
        Dimension3D = 1,
        DimensionST = 2,
    };

    /**
     * @enum VecchiaType
     * @brief Enum denoting the type of Vecchia approximation.
     * @details Three approximation methods:
     *          - PARALLEL_VECCHIA_GP: Scalar/point-wise with KBLAS
     *          - PARALLEL_BLOCK_VECCHIA_GP: Block/cluster-based with MAGMA
     *          - PARALLEL_SCALED_BLOCK_VECCHIA_GP: Distributed scaled block with MAGMA+MPI
     */
    enum VecchiaType {
        PARALLEL_VECCHIA_GP = 0,
        PARALLEL_BLOCK_VECCHIA_GP = 1,
        PARALLEL_SCALED_BLOCK_VECCHIA_GP = 2,
    };

    /**
     * @enum Precision
     * @brief Enum denoting the precision of the data.
     *
     */
    enum Precision {
        SINGLE = 0,
        DOUBLE = 1,
        MIXED = 2,
    };

    /**
     * @enum DistanceMetric
     * @brief Enum denoting distance metric type.
     *
     */
    enum DistanceMetric {
        EUCLIDEAN_DISTANCE = 0,
        GREAT_CIRCLE_DISTANCE = 1
    };

    /**
     * @enum OrderingMethod
     * @brief Enum denoting the ordering method for the data.
     *
     */
    enum OrderingMethod {
        RANDOM = 0,
        MORTON = 1,
        KD_TREE = 2,
        HILBERT = 3,
        MMD = 4,
    };

    /**
     * @enum PartitionMethod
     * @brief Enum denoting the partition method for distributed clustering.
     *
     */
    enum PartitionMethod {
        LINEAR_PARTITION = 0,
        NO_PARTITION = 1,
    };

    /**
     * @enum Descriptor Type
     * @brief Enum denoting the Descriptor Type.
     *
     */
    enum DescriptorType {
        CHAMELEON_DESCRIPTOR = 0,
        HICMA_DESCRIPTOR = 1
    };


    /**
     * @enum FloatPoint
     * @brief Enum denoting the floating point arithmetic of the matrix.
     *
     */
    enum FloatPoint : int {
        VECCHIA_BYTE = 0,
        VECCHIA_INTEGER = 1,
        VECCHIA_REAL_FLOAT = 2,
        VECCHIA_REAL_DOUBLE = 3,
        VECCHIA_COMPLEX_FLOAT = 4,
        VECCHIA_COMPLEX_DOUBLE = 5,
    };

    /**
     * @enum UpperLower
     * @brief Enum denoting the Upper/Lower part
     *
     */
    enum UpperLower : int {
        VECCHIA_UPPER = 121, /**< Use lower triangle of A */
        VECCHIA_LOWER = 122, /**< Use upper triangle of A */
        VECCHIA_UPPER_LOWER = 123  /**< Use the full A */
    };

    /**
     * @enum Descriptor Name
     * @brief Enum denoting all Descriptors Names.
     *
     */
    enum DescriptorName : int {
        DESCRIPTOR_C = 0,
        DESCRIPTOR_Z = 1,
        DESCRIPTOR_Z_COPY = 2,
        DESCRIPTOR_PRODUCT = 3,
        DESCRIPTOR_DETERMINANT = 4,
        DESCRIPTOR_CD = 5,
        DESCRIPTOR_CUV = 6,
        DESCRIPTOR_CRK = 7,
        DESCRIPTOR_Z_OBSERVATIONS = 8,
        DESCRIPTOR_Z_Actual = 9,
        DESCRIPTOR_Z_MISS = 10,
        DESCRIPTOR_MSPE = 11,
        DESCRIPTOR_Z_1 = 12,
        DESCRIPTOR_Z_2 = 13,
        DESCRIPTOR_Z_3 = 14,
        DESCRIPTOR_PRODUCT_1 = 15,
        DESCRIPTOR_PRODUCT_2 = 16,
        DESCRIPTOR_PRODUCT_3 = 17,
        DESCRIPTOR_C11 = 18,
        DESCRIPTOR_C12 = 19,
        DESCRIPTOR_C22 = 20,
        DESCRIPTOR_C12D = 21,
        DESCRIPTOR_C12UV = 22,
        DESCRIPTOR_C12RK = 23,
        DESCRIPTOR_C22D = 24,
        DESCRIPTOR_C22UV = 25,
        DESCRIPTOR_C22RK = 26,
        DESCRIPTOR_MSPE_1 = 27,
        DESCRIPTOR_MSPE_2 = 28,
        DESCRIPTOR_k_T = 29,
        DESCRIPTOR_k_A = 30,
        DESCRIPTOR_k_A_TMP = 31,
        DESCRIPTOR_k_T_TMP = 32,
        DESCRIPTOR_K_T = 33,
        DESCRIPTOR_K_T_TMP = 34,
        DESCRIPTOR_K_A = 35,
        DESCRIPTOR_EXPR_1 = 36,
        DESCRIPTOR_EXPR_2 = 37,
        DESCRIPTOR_EXPR_3 = 38,
        DESCRIPTOR_EXPR_4 = 39,
        DESCRIPTOR_MLOE = 40,
        DESCRIPTOR_MMOM = 41,
        DESCRIPTOR_MLOE_MMOM = 42,
        DESCRIPTOR_ALPHA = 43,
        DESCRIPTOR_TRUTH_ALPHA = 44,
        DESCRIPTOR_TIMATED_ALPHA = 45,
        DESCRIPTOR_CK = 46,
        DESCRIPTOR_CJ = 47,
        DESCRIPTOR_C_TRACE = 48,
        DESCRIPTOR_C_DIAG = 49,
        DESCRIPTOR_A = 50,
        DESCRIPTOR_RESULTS = 51,
        DESCRIPTOR_SUM = 52,
        DESCRIPTOR_R = 53,
        DESCRIPTOR_R_COPY = 54,
    };

    /**
     * @var availableKernels
     * @brief Set denoting the available kernels supported in matrix generation.
     * @details This set is updated automatically to add new kernels.
     * The set is initialized with a lambda function that iterates through a directory
     * and extracts the kernel names from the filenames. It also adds lowercase versions
     * of the kernel names with underscores before each capital letter.
     * @return set of all available kernels names
     *
     */
    const static std::set<std::string> availableKernels = []() {
        // This set stores the kernel names.
        std::set<std::string> kernelNames;
        // This string stores the directory path where the kernel files are located.
        const std::string directoryPath = KERNELS_PATH;
        // This loop iterates through all the files in the directory and extracts the kernel names.
        for (const auto &entry: std::filesystem::directory_iterator(directoryPath)) {
            // This checks if the current entry is a regular file.
            if (entry.is_regular_file()) {
                // This string stores the filename of the current entry.
                const std::string filename = entry.path().filename().string();
                // This string stores the file extension of the current entry.
                const std::string extension = std::filesystem::path(filename).extension().string();
                // This string stores the kernel name extracted from the filename.
                const std::string kernelName = filename.substr(0, filename.size() - extension.size());
                // This adds the kernel name to the kernelNames set.
                kernelNames.insert(kernelName);
                // This blo/ck of code converts the kernel name to lowercase     and adds underscores before each capital letter.
                std::string lowercaseName;
                for (std::size_t i = 0; i < kernelName.size(); ++i) {
                    if (std::isupper(kernelName[i])) {
                        // Avoid adding _ in the beginning of the name.
                        if (i != 0) {
                            lowercaseName += '_';
                        }
                        lowercaseName += static_cast<char>(std::tolower(kernelName[i]));
                    } else {
                        lowercaseName += kernelName[i];
                    }
                }
                // This adds the lowercase kernel name to the kernelNames set.
                kernelNames.insert(lowercaseName);
            }
        }
        return kernelNames;
    }();
}//namespace vecchia

#endif //VECCHIAGP_DEFINITIONS_HPP
