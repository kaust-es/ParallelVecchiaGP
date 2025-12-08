// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file FunctionsAdapter.cpp
 * @brief Implementation of function adapters for Vecchia R wrapper
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Khaled
 * @date 2025-01-01
**/

#include <Rcpp-adapters/FunctionsAdapter.hpp>
#include <api/VecchiaGP.hpp>
#include <configurations/Configurations.hpp>
#include <data-units/VecchiaGBData.hpp>
#include <data-units/Locations.hpp>
#include <common/Definitions.hpp>
#include <utilities/Logger.hpp>
#include <hardware/VecchiaHardware.hpp>
#include <helpers/CSVUtils.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>
#include <memory>

using namespace Rcpp;
using namespace vecchia::api;
using namespace vecchia::configurations;
using namespace vecchia::common;
using namespace vecchia::dataunits;

namespace vecchia::adapters {

std::string writeDataToTempCSV(const std::vector<double>& x_vec, 
                               const std::vector<double>& y_vec,
                               const std::vector<double>& m_vec) {
    char temp_file[] = "/tmp/vecchia_data_XXXXXX.csv";
    int fd = mkstemps(temp_file, 4);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary CSV file");
    }
    close(fd);
    
    std::ofstream file(temp_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open temporary CSV file");
    }
    
    for (size_t i = 0; i < x_vec.size(); i++) {
        file << x_vec[i] << "," << y_vec[i] << "," << m_vec[i] << "\n";
    }
    file.close();
    return std::string(temp_file);
}

std::string writeLocationsToTempCSV(const std::vector<double>& x_vec, 
                                    const std::vector<double>& y_vec) {
    char temp_file[] = "/tmp/vecchia_locs_XXXXXX.csv";
    int fd = mkstemps(temp_file, 4);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary locations CSV file");
    }
    close(fd);
    
    std::ofstream file(temp_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open temporary locations CSV file");
    }
    
    for (size_t i = 0; i < x_vec.size(); i++) {
        file << x_vec[i] << "," << y_vec[i] << "\n";
    }
    file.close();
    return std::string(temp_file);
}

std::vector<double> readPredictionsFromCSV(const std::string& csv_file, int expected_size) {
    std::vector<double> predictions;
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open predictions CSV file: " + csv_file);
    }
    
    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string token;
        if (std::getline(iss, token, ',')) {
            try {
                double mean_val = std::stod(token);
                predictions.push_back(mean_val);
            } catch (...) {
                continue;
            }
        }
    }
    file.close();
    
    while (predictions.size() < static_cast<size_t>(expected_size)) {
        predictions.push_back(0.0);
    }
    
    return predictions;
}

VecchiaType stringToVecchiaType(const std::string& vecchia_type) {
    if (vecchia_type == "parallel") {
        return PARALLEL_VECCHIA_GP;
    } else if (vecchia_type == "block") {
        return PARALLEL_BLOCK_VECCHIA_GP;
    } else if (vecchia_type == "scaled_block") {
        return PARALLEL_SCALED_BLOCK_VECCHIA_GP;
    } else {
        throw std::runtime_error("Invalid vecchia_type. Must be 'parallel', 'block', or 'scaled_block'");
    }
}

vecchia::common::Dimension stringToDimension(const std::string& dimension) {
    if (dimension == "2D") {
        return vecchia::common::Dimension2D;
    } else if (dimension == "3D") {
        return vecchia::common::Dimension3D;
    } else if (dimension == "ST") {
        return vecchia::common::DimensionST;
    } else {
        throw std::runtime_error("Invalid dimension. Must be '2D', '3D', or 'ST'");
    }
}

DistanceMetric stringToDistanceMetric(const std::string& distance_matrix) {
    if (distance_matrix == "euclidean") {
        return EUCLIDEAN_DISTANCE;
    } else if (distance_matrix == "great_circle") {
        return GREAT_CIRCLE_DISTANCE;
    } else {
        throw std::runtime_error("Invalid distance_matrix. Must be 'euclidean' or 'great_circle'");
    }
}

vecchia::common::OrderingMethod stringToOrderingMethod(const std::string& permutation) {
    std::string lower = permutation;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "random") {
        return vecchia::common::OrderingMethod::RANDOM;
    } else if (lower == "morton") {
        return vecchia::common::OrderingMethod::MORTON;
    } else if (lower == "kdtree" || lower == "kd_tree") {
        return vecchia::common::OrderingMethod::KD_TREE;
    } else if (lower == "hilbert") {
        return vecchia::common::OrderingMethod::HILBERT;
    } else if (lower == "mmd") {
        return vecchia::common::OrderingMethod::MMD;
    } else {
        throw std::runtime_error("Invalid permutation. Must be 'random', 'morton', 'kdtree', 'hilbert', or 'mmd'");
    }
}

List
R_VecchiaLoadData(const std::string &vecchia_type, const std::string &kernel,
                  const std::vector<double> &initial_theta, const std::string &distance_matrix,
                  const int &problem_size, const int &seed, const int &block_size,
                  const std::string &dimension, const std::string &data_path,
                  Nullable<NumericVector> distance_scale, Nullable<int> nn_multiplier,
                  Nullable<int> conditioning_size, Nullable<int> ncores,
                  Nullable<std::string> permutation, Nullable<std::string> kernel_type) {
    try {
        Configurations config;
        
        config.SetVecchiaType(stringToVecchiaType(vecchia_type));
        config.CheckKernelValue(kernel);
        
        config.SetInitialTheta(initial_theta);
        config.SetDistanceMetric(stringToDistanceMetric(distance_matrix));
        config.SetProblemSize(problem_size);
        config.SetSeed(seed);
        config.SetBlockSize(block_size);
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            try {
                int dim_size = std::stoi(dimension);
                if (dim_size > 0) {
                    config.SetDimensionSize(dim_size);
                    config.SetDimension(vecchia::common::Dimension2D);
                } else {
                    vecchia::common::Dimension dim = stringToDimension(dimension);
                    config.SetDimension(dim);
                    if (dim == vecchia::common::Dimension2D) {
                        config.SetDimensionSize(2);
                    } else if (dim == vecchia::common::Dimension3D) {
                        config.SetDimensionSize(3);
                    } else if (dim == vecchia::common::DimensionST) {
                        config.SetDimensionSize(3);
                    }
                }
            } catch (...) {
                vecchia::common::Dimension dim = stringToDimension(dimension);
                config.SetDimension(dim);
                if (dim == vecchia::common::Dimension2D) {
                    config.SetDimensionSize(2);
                } else if (dim == vecchia::common::Dimension3D) {
                    config.SetDimensionSize(3);
                } else if (dim == vecchia::common::DimensionST) {
                    config.SetDimensionSize(3);
                }
            }
            
            if (kernel_type.isNotNull()) {
                config.SetKernelType(as<std::string>(kernel_type));
            } else {
                config.SetKernelType("Matern72");
            }
            if (permutation.isNotNull()) {
                config.SetPermutation(stringToOrderingMethod(as<std::string>(permutation)));
            } else {
                config.SetPermutation(vecchia::common::OrderingMethod::RANDOM);
            }
            if (nn_multiplier.isNotNull()) {
                config.SetNNMultiplier(as<int>(nn_multiplier));
            } else {
                config.SetNNMultiplier(400);
            }
            if (distance_scale.isNotNull()) {
                NumericVector ds = as<NumericVector>(distance_scale);
                std::vector<double> ds_vec(ds.begin(), ds.end());
                config.SetDistanceScale(ds_vec);
            } else {
                std::vector<double> ds_default(config.GetDimensionSize(), 1.0);
                config.SetDistanceScale(ds_default);
            }
        } else {
            vecchia::common::Dimension dim = stringToDimension(dimension);
            config.SetDimension(dim);
            if (dim == vecchia::common::Dimension2D) {
                config.SetDimensionSize(2);
            } else if (dim == vecchia::common::Dimension3D) {
                config.SetDimensionSize(3);
            } else if (dim == vecchia::common::DimensionST) {
                config.SetDimensionSize(3);
            }
        }
        
        config.SetDataPath(data_path);
        
        if (conditioning_size.isNotNull()) {
            config.SetConditioningSize(as<int>(conditioning_size));
        } else if (config.GetConditioningSize() == 0) {
            config.SetConditioningSize(100);
        }
        
        if (ncores.isNotNull()) {
            config.SetCoresNumber(as<int>(ncores));
        } else {
            config.SetCoresNumber(omp_get_max_threads());
        }
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_VECCHIA_GP) {
            config.SetGPUsNumbers(1);
        } else {
            config.SetGPUsNumbers(0);
        }
        config.SetIsKNN(true);
        config.SetPrecision(DOUBLE);
        
        if (config.GetProblemSize() == 0) {
            throw std::runtime_error("You need to set the problem size, before starting");
        }
        if (config.GetKernelName().empty()) {
            throw std::runtime_error("You need to set the Kernel, before starting");
        }
        
        size_t found = config.GetKernelName().find("NonGaussian");
        if (found != std::string::npos) {
            config.SetIsNonGaussian(true);
        }
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            config.GenerateThetaAndBoundsFromKernelType();
        }
        
        config.InitializeAllTheta();
        
        std::unique_ptr<VecchiaHardware> hardware = std::make_unique<VecchiaHardware>(
            config.GetVecchiaType(), 
            config.GetCoresNumber(),
            config.GetGPUsNumbers());
        
        std::unique_ptr<VecchiaGBData<double>> vecchia_data;
        VecchiaGP<double>::VecchiaLoadData(config, vecchia_data);
        
        auto* locations = vecchia_data->GetLocations();
        if (locations == nullptr) {
            throw std::runtime_error("Failed to get locations from VecchiaGBData");
        }
        
        int size = locations->GetSize();
        double* x_ptr = locations->GetLocationX();
        double* y_ptr = locations->GetLocationY();
        double* measurements_ptr = vecchia_data->GetHostObservations();
        
        if (x_ptr == nullptr || y_ptr == nullptr || measurements_ptr == nullptr) {
            throw std::runtime_error("Failed to get location or measurement data");
        }
        
        NumericVector x(size);
        NumericVector y(size);
        NumericVector m(size);
        
        for (int i = 0; i < size; i++) {
            x[i] = x_ptr[i];
            y[i] = y_ptr[i];
            m[i] = measurements_ptr[i];
        }
        
        std::string data_file = config.GetDataPath();
        if (data_file.empty() && config.GetVecchiaType() != vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            std::vector<double> x_vec_data(size), y_vec_data(size), m_vec_data(size);
            for (int i = 0; i < size; i++) {
                x_vec_data[i] = x_ptr[i];
                y_vec_data[i] = y_ptr[i];
                m_vec_data[i] = measurements_ptr[i];
            }
            data_file = writeDataToTempCSV(x_vec_data, y_vec_data, m_vec_data);
        }
        
        XPtr<VecchiaGBData<double>> vecchia_data_ptr(vecchia_data.release(), true);
        XPtr<VecchiaHardware> hardware_ptr(hardware.release(), true);
        
        return List::create(
            Named("x") = x,
            Named("y") = y,
            Named("m") = m,
            Named("data_path") = data_file,
            Named("seed") = config.GetSeed(),
            Named("vecchia_type") = vecchia_type,
            Named("vecchia_data") = vecchia_data_ptr,
            Named("vecchia_hardware") = hardware_ptr
        );
        
    } catch (const std::exception& e) {
        stop("Error in load_data: %s", e.what());
    }
}

List
R_VecchiaModelData(const std::string &vecchia_type, const std::string &kernel,
                  const std::string &distance_matrix, const std::vector<double> &lb,
                  const std::vector<double> &ub, const double &tol, const int &mle_itr,
                  const int &block_size, const std::string &dimension,
                  SEXP data, Nullable<NumericVector> matrix,
                  Nullable<NumericVector> x, Nullable<NumericVector> y,
                  Nullable<NumericVector> initial_theta, Nullable<NumericVector> distance_scale,
                  Nullable<int> nn_multiplier, Nullable<int> conditioning_size, Nullable<int> ncores,
                  Nullable<std::string> permutation, Nullable<std::string> kernel_type,
                  Nullable<int> seed) {
    try {
        Configurations config;
        
        std::unique_ptr<VecchiaGBData<double>> vecchia_data;
        SEXP vecchia_data_sexp = R_NilValue;
        bool vecchia_data_xptr_valid = false;
        bool data_already_loaded = false;
        std::string data_file_path;
        int load_data_seed = -1;
        std::string load_data_vecchia_type;
        List data_list;
        SEXP hardware_sexp = R_NilValue;
        bool hardware_ptr_valid = false;
        if (data != R_NilValue) {
            data_list = as<List>(data);
            if (data_list.containsElementNamed("vecchia_data")) {
                try {
                    vecchia_data_sexp = data_list["vecchia_data"];
                    if (vecchia_data_sexp != R_NilValue) {
                        XPtr<VecchiaGBData<double>> vecchia_data_xptr_check(vecchia_data_sexp);
                        if (vecchia_data_xptr_check.get() != nullptr) {
                            vecchia_data_xptr_valid = true;
                            data_already_loaded = true;
                        }
                    }
                } catch (...) {
                    data_already_loaded = false;
                    vecchia_data_xptr_valid = false;
                }
            }
            
            if (data_list.containsElementNamed("vecchia_hardware")) {
                try {
                    hardware_sexp = data_list["vecchia_hardware"];
                    if (hardware_sexp != R_NilValue) {
                        XPtr<VecchiaHardware> hardware_ptr_check(hardware_sexp);
                        if (hardware_ptr_check.get() != nullptr) {
                            hardware_ptr_valid = true;
                        }
                    }
                } catch (...) {
                    hardware_ptr_valid = false;
                }
            }
            
            if (!data_already_loaded && data_list.containsElementNamed("data_path")) {
                data_file_path = as<std::string>(data_list["data_path"]);
                if (data_list.containsElementNamed("seed")) {
                    load_data_seed = as<int>(data_list["seed"]);
                }
                if (data_list.containsElementNamed("vecchia_type")) {
                    load_data_vecchia_type = as<std::string>(data_list["vecchia_type"]);
                }
                
                if (!data_file_path.empty()) {
                    data_already_loaded = true;
                } else if (load_data_seed >= 0 && load_data_vecchia_type == vecchia_type) {
                    data_already_loaded = true;
                }
            }
        }
        
        config.SetVecchiaType(stringToVecchiaType(vecchia_type));
        config.CheckKernelValue(kernel);
        config.SetDistanceMetric(stringToDistanceMetric(distance_matrix));
        
        double abs_tol = std::pow(10.0, -tol);
        config.SetTolerance(abs_tol);
        config.SetMaxMleIterations(mle_itr);
        config.SetBlockSize(block_size);
        config.SetLowerBounds(lb);
        config.SetUpperBounds(ub);
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            try {
                int dim_size = std::stoi(dimension);
                if (dim_size > 0) {
                    config.SetDimensionSize(dim_size);
                    config.SetDimension(vecchia::common::Dimension2D);
                } else {
                    vecchia::common::Dimension dim = stringToDimension(dimension);
                    config.SetDimension(dim);
                    if (dim == vecchia::common::Dimension2D) {
                        config.SetDimensionSize(2);
                    } else if (dim == vecchia::common::Dimension3D) {
                        config.SetDimensionSize(3);
                    } else if (dim == vecchia::common::DimensionST) {
                        config.SetDimensionSize(3);
                    }
                }
            } catch (...) {
                vecchia::common::Dimension dim = stringToDimension(dimension);
                config.SetDimension(dim);
                if (dim == vecchia::common::Dimension2D) {
                    config.SetDimensionSize(2);
                } else if (dim == vecchia::common::Dimension3D) {
                    config.SetDimensionSize(3);
                } else if (dim == vecchia::common::DimensionST) {
                    config.SetDimensionSize(3);
                }
            }
            
            if (kernel_type.isNotNull()) {
                config.SetKernelType(as<std::string>(kernel_type));
            } else {
                config.SetKernelType("Matern72");
            }
            if (permutation.isNotNull()) {
                config.SetPermutation(stringToOrderingMethod(as<std::string>(permutation)));
            } else {
                config.SetPermutation(vecchia::common::OrderingMethod::RANDOM);
            }
            if (nn_multiplier.isNotNull()) {
                config.SetNNMultiplier(as<int>(nn_multiplier));
            } else {
                config.SetNNMultiplier(400);
            }
            if (distance_scale.isNotNull()) {
                NumericVector ds = as<NumericVector>(distance_scale);
                std::vector<double> ds_vec(ds.begin(), ds.end());
                config.SetDistanceScale(ds_vec);
            } else {
                std::vector<double> ds_default(config.GetDimensionSize(), 1.0);
                config.SetDistanceScale(ds_default);
            }
        } else {
            vecchia::common::Dimension dim = stringToDimension(dimension);
            config.SetDimension(dim);
            if (dim == vecchia::common::Dimension2D) {
                config.SetDimensionSize(2);
            } else if (dim == vecchia::common::Dimension3D) {
                config.SetDimensionSize(3);
            } else if (dim == vecchia::common::DimensionST) {
                config.SetDimensionSize(3);
            }
        }
        
        if (conditioning_size.isNotNull()) {
            config.SetConditioningSize(as<int>(conditioning_size));
        } else {
            config.SetConditioningSize(100);
        }
        
        if (ncores.isNotNull()) {
            config.SetCoresNumber(as<int>(ncores));
        } else {
            config.SetCoresNumber(omp_get_max_threads());
        }
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_VECCHIA_GP) {
            config.SetGPUsNumbers(1);
        } else {
            config.SetGPUsNumbers(0);
        }
        config.SetIsKNN(true);
        config.SetPrecision(DOUBLE);
        
        if (seed.isNotNull()) {
            config.SetSeed(as<int>(seed));
        } else {
            config.SetSeed(0);  // Default seed
        }
        
        int problem_size = 0;
        std::vector<double> x_vec, y_vec, measurements_vec;
        bool use_provided_data = false;
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP && seed.isNotNull()) {
            if (data != R_NilValue) {
                List data_list = as<List>(data);
                NumericVector x_data = as<NumericVector>(data_list["x"]);
                problem_size = x_data.size();
            } else if (matrix.isNotNull()) {
                NumericVector m_vec_r = as<NumericVector>(matrix);
                problem_size = m_vec_r.size();
            } else if (x.isNotNull()) {
                NumericVector x_vec_r = as<NumericVector>(x);
                problem_size = x_vec_r.size();
            }
            if (problem_size == 0) {
                problem_size = config.GetProblemSize();
                if (problem_size == 0) {
                    throw std::runtime_error("For scaled_block with seed, problem_size must be provided via data parameters or set in config");
                }
            }
        } else {
        if (data != R_NilValue) {
            List data_list = as<List>(data);
            NumericVector x_data = as<NumericVector>(data_list["x"]);
            NumericVector y_data = as<NumericVector>(data_list["y"]);
            NumericVector m_data = as<NumericVector>(data_list["m"]);
            
            problem_size = x_data.size();
            x_vec.assign(x_data.begin(), x_data.end());
            y_vec.assign(y_data.begin(), y_data.end());
            measurements_vec.assign(m_data.begin(), m_data.end());
                use_provided_data = true;
        } else if (matrix.isNotNull() && x.isNotNull() && y.isNotNull()) {
            NumericVector x_vec_r = as<NumericVector>(x);
            NumericVector y_vec_r = as<NumericVector>(y);
            NumericVector m_vec_r = as<NumericVector>(matrix);
            
            problem_size = m_vec_r.size();
            x_vec.assign(x_vec_r.begin(), x_vec_r.end());
            y_vec.assign(y_vec_r.begin(), y_vec_r.end());
            measurements_vec.assign(m_vec_r.begin(), m_vec_r.end());
                use_provided_data = true;
        } else {
            throw std::runtime_error("Either 'data' parameter or 'matrix', 'x', 'y' parameters must be provided");
            }
        }
        
        config.SetProblemSize(problem_size);
        
        if (config.GetProblemSize() == 0) {
            throw std::runtime_error("You need to set the problem size, before starting");
        }
        if (config.GetKernelName().empty()) {
            throw std::runtime_error("You need to set the Kernel, before starting");
        }
        
        size_t found = config.GetKernelName().find("NonGaussian");
        if (found != std::string::npos) {
            config.SetIsNonGaussian(true);
        }
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            if (initial_theta.isNotNull()) {
                NumericVector it = as<NumericVector>(initial_theta);
                std::vector<double> it_vec(it.begin(), it.end());
                if (it_vec.size() >= 2) {
                    config.SetInitialTheta(std::vector<double>{it_vec[0], it_vec[1]});
                } else {
                    config.SetInitialTheta(std::vector<double>{1.0, 0.001});
                }
            } else {
                config.SetInitialTheta(std::vector<double>{1.0, 0.001});
            }
        } else {
            if (initial_theta.isNotNull()) {
                NumericVector it = as<NumericVector>(initial_theta);
                std::vector<double> it_vec(it.begin(), it.end());
                if (!it_vec.empty()) {
                    config.SetInitialTheta(it_vec);
                }
            }
        }
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            config.GenerateThetaAndBoundsFromKernelType();
        }
        if (config.GetVecchiaType() != vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            auto current_theta = config.GetInitialTheta();
            if (current_theta.empty() && initial_theta.isNotNull()) {
                NumericVector it = as<NumericVector>(initial_theta);
                std::vector<double> it_vec(it.begin(), it.end());
                if (!it_vec.empty()) {
                    config.SetInitialTheta(it_vec);
                }
            }
        }
        
        config.InitializeAllTheta();
        if (!data_already_loaded && data != R_NilValue) {
            if (data_list.containsElementNamed("data_path")) {
                data_file_path = as<std::string>(data_list["data_path"]);
                if (data_list.containsElementNamed("seed")) {
                    load_data_seed = as<int>(data_list["seed"]);
                }
                if (data_list.containsElementNamed("vecchia_type")) {
                    load_data_vecchia_type = as<std::string>(data_list["vecchia_type"]);
                }
                
                if (!data_file_path.empty()) {
                    data_already_loaded = true;
                    config.SetDataPath(data_file_path);
                } else if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP && 
                          load_data_seed >= 0 && load_data_vecchia_type == vecchia_type) {
                    data_already_loaded = true;
                    config.SetDataPath("");
                    if (seed.isNotNull()) {
                        int provided_seed = as<int>(seed);
                        if (provided_seed != load_data_seed) {
                            Rcpp::warning("Seed mismatch: load_data used seed %d, model_data provided seed %d. Using load_data seed.",
                                         load_data_seed, provided_seed);
                        }
                    }
                    config.SetSeed(load_data_seed);
                }
            }
        }
        
        std::string temp_csv;
        bool cleanup_temp_file = false;
        
        if (!data_already_loaded) {
            if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP && seed.isNotNull()) {
                config.SetDataPath("");
            } else if (use_provided_data) {
                temp_csv = writeDataToTempCSV(x_vec, y_vec, measurements_vec);
                config.SetDataPath(temp_csv);
                cleanup_temp_file = true;
            } else {
                throw std::runtime_error("Either 'data'/'matrix'/'x'/'y' parameters or 'seed' parameter must be provided");
            }
        }
        
        std::unique_ptr<VecchiaHardware> hardware;
        if (hardware_ptr_valid && hardware_sexp != R_NilValue) {
            XPtr<VecchiaHardware> hardware_ptr(hardware_sexp);
        }
        
        bool hardware_already_initialized = false;
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_BLOCK_VECCHIA_GP || 
            config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            if (VecchiaHardware::GetQueue() != nullptr) {
                hardware_already_initialized = true;
            }
        } else if (config.GetVecchiaType() == vecchia::common::PARALLEL_VECCHIA_GP) {
            if (VecchiaHardware::GetNumKblasHandles() > 0) {
                hardware_already_initialized = true;
            }
        }
        
        if (!data_already_loaded || !hardware_ptr_valid) {
            if (!hardware_already_initialized) {
                hardware = std::make_unique<VecchiaHardware>(config.GetVecchiaType(), 
                                 config.GetCoresNumber(),
                                 config.GetGPUsNumbers());
            }
        }
        
        if (!data_already_loaded) {
            vecchia_data = std::unique_ptr<VecchiaGBData<double>>();
            VecchiaGP<double>::VecchiaLoadData(config, vecchia_data);
        }
        
        double log_likelihood;
        if (data_already_loaded && vecchia_data_xptr_valid && vecchia_data_sexp != R_NilValue) {
            XPtr<VecchiaGBData<double>> vecchia_data_xptr(vecchia_data_sexp);
            VecchiaGBData<double>* raw_ptr = vecchia_data_xptr.get();
            std::unique_ptr<VecchiaGBData<double>> vecchia_data_for_estimation(raw_ptr);
            log_likelihood = VecchiaGP<double>::VecchiaDataEstimation(config, vecchia_data_for_estimation, nullptr);
            vecchia_data_for_estimation.release();
        } else {
            log_likelihood = VecchiaGP<double>::VecchiaDataEstimation(config, vecchia_data, nullptr);
        }
        
        std::vector<double> estimated_theta = config.GetEstimatedTheta();
        if (estimated_theta.empty()) {
            estimated_theta = config.GetInitialTheta();
        }
        
        if (cleanup_temp_file) {
            std::remove(temp_csv.c_str());
        }
        return List::create(
            _["log_likelihood"] = log_likelihood,
            _["estimated_theta"] = NumericVector(estimated_theta.begin(), estimated_theta.end())
        );
        
    } catch (const std::exception& e) {
        stop("Error in model_data: %s", e.what());
    }
}

NumericVector
R_VecchiaPredictData(const std::string &vecchia_type, const std::string &kernel,
                    const std::string &distance_matrix, const std::vector<double> &estimated_theta,
                    const int &block_size, const std::string &dimension,
                    SEXP train_data, SEXP test_data,
                    Nullable<std::string> train_locs, Nullable<std::string> test_locs,
                    Nullable<NumericVector> distance_scale, Nullable<int> nn_multiplier,
                    Nullable<int> conditioning_size, Nullable<int> ncores,
                    Nullable<std::string> permutation, Nullable<std::string> kernel_type,
                    Nullable<int> seed, Nullable<int> problem_size) {
    try {
        Configurations config;
        
        config.SetVecchiaType(stringToVecchiaType(vecchia_type));
        config.CheckKernelValue(kernel);
        config.SetDistanceMetric(stringToDistanceMetric(distance_matrix));
        config.SetBlockSize(block_size);
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            try {
                int dim_size = std::stoi(dimension);
                if (dim_size > 0) {
                    config.SetDimensionSize(dim_size);
                    config.SetDimension(vecchia::common::Dimension2D);
                } else {
                    vecchia::common::Dimension dim = stringToDimension(dimension);
                    config.SetDimension(dim);
                    if (dim == vecchia::common::Dimension2D) {
                        config.SetDimensionSize(2);
                    } else if (dim == vecchia::common::Dimension3D) {
                        config.SetDimensionSize(3);
                    } else if (dim == vecchia::common::DimensionST) {
                        config.SetDimensionSize(3);
                    }
                }
            } catch (...) {
                vecchia::common::Dimension dim = stringToDimension(dimension);
                config.SetDimension(dim);
                if (dim == vecchia::common::Dimension2D) {
                    config.SetDimensionSize(2);
                } else if (dim == vecchia::common::Dimension3D) {
                    config.SetDimensionSize(3);
                } else if (dim == vecchia::common::DimensionST) {
                    config.SetDimensionSize(3);
                }
            }
            
            if (kernel_type.isNotNull()) {
                config.SetKernelType(as<std::string>(kernel_type));
            } else {
                config.SetKernelType("Matern72");
            }
            if (permutation.isNotNull()) {
                config.SetPermutation(stringToOrderingMethod(as<std::string>(permutation)));
            } else {
                config.SetPermutation(vecchia::common::OrderingMethod::RANDOM);
            }
            if (nn_multiplier.isNotNull()) {
                config.SetNNMultiplier(as<int>(nn_multiplier));
            } else {
                config.SetNNMultiplier(400);
            }
            if (distance_scale.isNotNull()) {
                NumericVector ds = as<NumericVector>(distance_scale);
                std::vector<double> ds_vec(ds.begin(), ds.end());
                config.SetDistanceScale(ds_vec);
            } else {
                std::vector<double> ds_default(config.GetDimensionSize(), 1.0);
                config.SetDistanceScale(ds_default);
            }
        } else {
            vecchia::common::Dimension dim = stringToDimension(dimension);
            config.SetDimension(dim);
            if (dim == vecchia::common::Dimension2D) {
                config.SetDimensionSize(2);
            } else if (dim == vecchia::common::Dimension3D) {
                config.SetDimensionSize(3);
            } else if (dim == vecchia::common::DimensionST) {
                config.SetDimensionSize(3);
            }
        }
        
        if (conditioning_size.isNotNull()) {
            config.SetConditioningSize(as<int>(conditioning_size));
        } else {
            config.SetConditioningSize(100);
        }
        
        if (ncores.isNotNull()) {
            config.SetCoresNumber(as<int>(ncores));
        } else {
            config.SetCoresNumber(omp_get_max_threads());
        }
        config.SetGPUsNumbers(0);
        config.SetIsKNN(true);
        config.SetPrecision(DOUBLE);
        
        SEXP vecchia_data_sexp = R_NilValue;
        bool vecchia_data_xptr_valid = false;
        bool data_already_loaded = false;
        List train_data_list;
        SEXP hardware_sexp = R_NilValue;
        bool hardware_ptr_valid = false;
        int load_data_seed = -1;
        
        if (train_data != R_NilValue && Rf_isNewList(train_data) && Rf_length(train_data) > 0) {
            try {
                train_data_list = as<List>(train_data);
                if (train_data_list.containsElementNamed("vecchia_data")) {
                    vecchia_data_sexp = train_data_list["vecchia_data"];
                    if (vecchia_data_sexp != R_NilValue) {
                        XPtr<VecchiaGBData<double>> vecchia_data_xptr_check(vecchia_data_sexp);
                        if (vecchia_data_xptr_check.get() != nullptr) {
                            vecchia_data_xptr_valid = true;
                            data_already_loaded = true;
                        }
                    }
                }
                
                if (train_data_list.containsElementNamed("vecchia_hardware")) {
                    hardware_sexp = train_data_list["vecchia_hardware"];
                    if (hardware_sexp != R_NilValue) {
                        XPtr<VecchiaHardware> hardware_ptr_check(hardware_sexp);
                        if (hardware_ptr_check.get() != nullptr) {
                            hardware_ptr_valid = true;
                        }
                    }
                }
                
                if (train_data_list.containsElementNamed("seed")) {
                    load_data_seed = as<int>(train_data_list["seed"]);
                }
            } catch (...) {
                data_already_loaded = false;
                vecchia_data_xptr_valid = false;
                hardware_ptr_valid = false;
            }
        }
        
        if (seed.isNotNull()) {
            config.SetSeed(as<int>(seed));
        } else if (data_already_loaded && load_data_seed >= 0) {
            config.SetSeed(load_data_seed);
        } else {
            config.SetSeed(123);
        }
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            int dim = std::stoi(dimension);
            
            if (estimated_theta.size() >= 2) {
                config.SetInitialTheta(std::vector<double>{estimated_theta[0], estimated_theta[1]});
            } else {
                config.SetInitialTheta(std::vector<double>{1.0, 0.001});
            }
            
            config.GenerateThetaAndBoundsFromKernelType();
            
            if (estimated_theta.size() >= 2 + dim) {
                config.SetInitialTheta(estimated_theta);
                config.SetEstimatedTheta(estimated_theta);
            } else if (estimated_theta.size() >= 2) {
                    config.SetEstimatedTheta(config.GetInitialTheta());
            } else {
                config.SetEstimatedTheta(config.GetInitialTheta());
            }
        } else {
            config.SetInitialTheta(estimated_theta);
            config.SetEstimatedTheta(estimated_theta);
        }
        
        config.InitializeAllTheta();
        
        std::unique_ptr<VecchiaHardware> hardware;
        if (hardware_ptr_valid && hardware_sexp != R_NilValue) {
            XPtr<VecchiaHardware> hardware_ptr(hardware_sexp);
        }
        
        bool hardware_already_initialized = false;
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_BLOCK_VECCHIA_GP || 
            config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            if (VecchiaHardware::GetQueue() != nullptr) {
                hardware_already_initialized = true;
            }
        } else if (config.GetVecchiaType() == vecchia::common::PARALLEL_VECCHIA_GP) {
            if (VecchiaHardware::GetNumKblasHandles() > 0) {
                hardware_already_initialized = true;
            }
        }
        
        if (!data_already_loaded || !hardware_ptr_valid) {
            if (!hardware_already_initialized) {
                hardware = std::make_unique<VecchiaHardware>(config.GetVecchiaType(), 
                                 config.GetCoresNumber(),
                                 config.GetGPUsNumbers());
            }
        }
        
        bool use_file_paths = train_locs.isNotNull() && test_locs.isNotNull();
        
        std::string train_data_csv, train_locs_csv, test_locs_csv, test_data_csv;
        bool cleanup_temp_files = false;
        int test_size = 0;
        
        if (use_file_paths) {
            train_locs_csv = as<std::string>(train_locs);
            test_locs_csv = as<std::string>(test_locs);
            
            if (Rf_isString(train_data)) {
                train_data_csv = CHAR(STRING_ELT(train_data, 0));
            } else if (Rf_isNewList(train_data)) {
                throw std::runtime_error("When train_locs is provided, train_data must be a file path string (not a list). Pass data_result from load_data() separately if you want hardware reuse, or use train_data as file path string.");
            } else {
                throw std::runtime_error("When train_locs is provided, train_data must be a file path string");
            }
            
            if (Rf_isString(test_data)) {
                test_data_csv = CHAR(STRING_ELT(test_data, 0));
            } else {
                throw std::runtime_error("When test_locs is provided, test_data must be a file path string");
            }
            
            std::ifstream test_file(test_locs_csv);
            if (!test_file.is_open()) {
                throw std::runtime_error("Cannot open test locations file: " + test_locs_csv);
            }
            test_size = std::count(std::istreambuf_iterator<char>(test_file),
                                  std::istreambuf_iterator<char>(), '\n');
            test_file.close();
            
            test_file.open(test_locs_csv);
            if (test_file.is_open()) {
                test_file.seekg(-1, std::ios::end);
                char last_char;
                test_file.get(last_char);
                if (last_char != '\n' && test_size == 0) {
                    test_size = 1;
                } else if (last_char != '\n' && test_size > 0) {
                    test_size++;
                }
                test_file.close();
            }
            
            config.SetTrainLocationsPath(train_locs_csv);
            config.SetTestLocationsPath(test_locs_csv);
            config.SetTrainDataPath(train_data_csv);
            config.SetTestDataPath(test_data_csv);
            
            config.SetDataPath("");
            if (problem_size.isNotNull()) {
                config.SetProblemSize(as<int>(problem_size));
            } else if (config.GetProblemSize() == 0) {
                config.SetProblemSize(50);
            }
        } else {
            if (data_already_loaded && config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
                if (problem_size.isNotNull()) {
                    config.SetProblemSize(as<int>(problem_size));
                }
            } else {
            List train_data_list = as<List>(train_data);
            if (train_data_list.size() < 3) {
                throw std::runtime_error("train_data must contain [x, y, measurements] or be a file path string");
            }
            NumericVector train_x = as<NumericVector>(train_data_list[0]);
            NumericVector train_y = as<NumericVector>(train_data_list[1]);
            NumericVector train_m = as<NumericVector>(train_data_list[2]);
            
            int train_size = train_x.size();
            if (train_y.size() != train_size || train_m.size() != train_size) {
                throw std::runtime_error("train_data vectors must have same length");
            }
            
                if (test_data == R_NilValue) {
                    throw std::runtime_error("test_data must contain [x, y] or be a file path string");
                }
                List test_data_list = as<List>(test_data);
                if (test_data_list.size() < 2) {
                    throw std::runtime_error("test_data must contain [x, y] or be a file path string");
                }
                NumericVector test_x = as<NumericVector>(test_data_list[0]);
                NumericVector test_y = as<NumericVector>(test_data_list[1]);
                
                int test_size = test_x.size();
                if (test_y.size() != test_size) {
                    throw std::runtime_error("test_data vectors must have same length");
                }
                
                config.SetProblemSize(train_size);
                
                std::vector<double> train_x_vec(train_x.begin(), train_x.end());
                std::vector<double> train_y_vec(train_y.begin(), train_y.end());
                std::vector<double> train_m_vec(train_m.begin(), train_m.end());
                std::vector<double> test_x_vec(test_x.begin(), test_x.end());
                std::vector<double> test_y_vec(test_y.begin(), test_y.end());
                
                train_data_csv = writeDataToTempCSV(train_x_vec, train_y_vec, train_m_vec);
                train_locs_csv = writeLocationsToTempCSV(train_x_vec, train_y_vec);
                test_locs_csv = writeLocationsToTempCSV(test_x_vec, test_y_vec);
                
                std::vector<double> test_dummy_data(test_size, 0.0);
                test_data_csv = writeDataToTempCSV(test_x_vec, test_y_vec, test_dummy_data);
            config.SetTrainLocationsPath(train_locs_csv);
            config.SetTestLocationsPath(test_locs_csv);
            config.SetTrainDataPath(train_data_csv);
            config.SetTestDataPath(test_data_csv);
            
            cleanup_temp_files = true;
            }
        }
        
        if (config.GetProblemSize() == 0 && !use_file_paths && 
            !(data_already_loaded && config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP)) {
            throw std::runtime_error("You need to set the problem size, before starting");
        }
        if (config.GetKernelName().empty()) {
            throw std::runtime_error("You need to set the Kernel, before starting");
        }
        
        size_t found = config.GetKernelName().find("NonGaussian");
        if (found != std::string::npos) {
            config.SetIsNonGaussian(true);
        }
        
        config.SetConditionalSimulations(1000);
        config.SetScaleFactor(1.0);
        config.SetKMeansMaxIter(50);
        
        std::unique_ptr<VecchiaGBData<double>> vecchia_data;
        if (!data_already_loaded) {
            vecchia_data = std::unique_ptr<VecchiaGBData<double>>();
        VecchiaGP<double>::VecchiaLoadData(config, vecchia_data);
        } else {
            XPtr<VecchiaGBData<double>> vecchia_data_xptr(vecchia_data_sexp);
            VecchiaGBData<double>* raw_ptr = vecchia_data_xptr.get();
            vecchia_data = std::unique_ptr<VecchiaGBData<double>>(raw_ptr);
        }
        
        double* measurements_ptr = vecchia_data->GetHostObservations();
        if (!use_file_paths && measurements_ptr == nullptr && 
            !(data_already_loaded && config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP)) {
            if (cleanup_temp_files) {
                std::remove(train_data_csv.c_str());
                std::remove(train_locs_csv.c_str());
                std::remove(test_locs_csv.c_str());
                std::remove(test_data_csv.c_str());
            }
            throw std::runtime_error("Failed to get measurements from VecchiaGBData");
        }
        
        if (use_file_paths) {
            measurements_ptr = nullptr;
        }
        
        if (data_already_loaded && config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            measurements_ptr = nullptr;
        }
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            auto current_lb = config.GetLowerBounds();
            auto current_ub = config.GetUpperBounds();
            int dim = std::stoi(dimension);
            int expected_size = 2 + dim;
            
            bool bounds_corrupted = false;
            if (current_lb.size() == expected_size && current_ub.size() == expected_size) {
                if (estimated_theta.size() >= 3) {
                    if (std::abs(current_lb[0] - estimated_theta[0]) < 1e-6 &&
                        std::abs(current_lb[1] - estimated_theta[1]) < 1e-6 &&
                        std::abs(current_ub[0] - estimated_theta[0]) < 1e-6 &&
                        std::abs(current_ub[1] - estimated_theta[1]) < 1e-6) {
                        bounds_corrupted = true;
                    }
                }
            }
            
            if (current_lb.size() != expected_size || current_ub.size() != expected_size || bounds_corrupted) {
                if (estimated_theta.size() >= 2) {
                    config.SetInitialTheta(std::vector<double>{estimated_theta[0], estimated_theta[1]});
                } else {
                    config.SetInitialTheta(std::vector<double>{1.0, 0.001});
                }
                config.GenerateThetaAndBoundsFromKernelType();
                
                if (estimated_theta.size() >= 2 + dim) {
                    config.SetInitialTheta(estimated_theta);
                    config.SetEstimatedTheta(estimated_theta);
                }
            }
        }
        
        VecchiaGP<double>::VecchiaPrediction(config, vecchia_data, measurements_ptr);
        
        if (data_already_loaded && vecchia_data_xptr_valid) {
            vecchia_data.release();
        }
        
        NumericVector predictions;
        
        if (config.GetVecchiaType() == vecchia::common::PARALLEL_SCALED_BLOCK_VECCHIA_GP) {
            predictions = NumericVector(test_size, 0.0);
        } else {
            std::string predictions_csv;
            if (estimated_theta.size() == 3) {
                predictions_csv = "log/conditional_simulation_k_" + 
                                std::to_string(block_size) + "_m_" + 
                                std::to_string(config.GetConditioningSize()) + 
                                "_theta_" + std::to_string(estimated_theta[0]) + "_" + 
                                std::to_string(estimated_theta[1]) + "_" + 
                                std::to_string(estimated_theta[2]) + 
                                "_seed_" + std::to_string(config.GetSeed()) + ".csv";
            } else {
                predictions_csv = "log/conditional_simulation_k_" + 
                                std::to_string(block_size) + "_m_" + 
                                std::to_string(config.GetConditioningSize()) + 
                                "_theta_" + std::to_string(estimated_theta[0]) + "_" + 
                                std::to_string(estimated_theta[1]) + "_" + 
                                std::to_string(estimated_theta[2]) + "_" + 
                                std::to_string(estimated_theta[3]) + 
                                "_seed_" + std::to_string(config.GetSeed()) + ".csv";
            }
            
            std::vector<double> predictions_vec;
            try {
                predictions_vec = readPredictionsFromCSV(predictions_csv, test_size);
            } catch (const std::exception& e) {
                Rcpp::warning("Could not read predictions from CSV file: %s. Returning zeros.", e.what());
                predictions_vec.assign(test_size, 0.0);
            }
            
            predictions = NumericVector(test_size);
            for (int i = 0; i < test_size && i < static_cast<int>(predictions_vec.size()); i++) {
                predictions[i] = predictions_vec[i];
            }
        }
        
        if (cleanup_temp_files) {
            std::remove(train_data_csv.c_str());
            std::remove(train_locs_csv.c_str());
            std::remove(test_locs_csv.c_str());
            std::remove(test_data_csv.c_str());
        }
        
        return predictions;
        
    } catch (const std::exception& e) {
        stop("Error in predict_data: %s", e.what());
    }
}

} // namespace vecchia::adapters

