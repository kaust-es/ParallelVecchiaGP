
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file SyntheticGenerator.cpp
 * @brief Implementation of the SyntheticGenerator class
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <data-generators/concrete/SyntheticGenerator.hpp>
#include <data-generators/LocationGenerator.hpp>
#include <data-loader/concrete/CSVLoader.hpp>

using namespace vecchia::generators::synthetic;
using namespace vecchia::common;
using namespace vecchia::configurations;

template<typename T>
SyntheticGenerator<T> *SyntheticGenerator<T>::GetInstance() {

    if (mpInstance == nullptr) {
        mpInstance = new SyntheticGenerator<T>();
    }
    return mpInstance;
}

template<typename T>
std::unique_ptr<VecchiaGBData<T>>
SyntheticGenerator<T>::CreateData(Configurations &aConfigurations,
                                  vecchia::kernels::Kernel<T> &aKernel) {

    int n = aConfigurations.GetProblemSize() * aConfigurations.GetTimeSlot();
    auto data = std::make_unique<VecchiaGBData<T>>(n, aConfigurations.GetDimension(), aConfigurations.GetBlockSize());

    // check for space-time kernel
    if (aConfigurations.GetTimeSlot() != 0 && n % aConfigurations.GetTimeSlot() != 0)
    {
        throw std::range_error(std::string("Your number of locations cannot be divided by t_slot!"));
    }

    // Allocated new Locations object.
    auto *locations = new dataunits::Locations<T>(n, aConfigurations.GetDimension());
    int parameters_number = aKernel.GetParametersNumbers();

    // Set initial theta values.
    Configurations::InitTheta(aConfigurations.GetInitialTheta(), parameters_number);
    aConfigurations.SetInitialTheta(aConfigurations.GetInitialTheta());

    // Generate Locations phase
    LocationGenerator<T>::GenerateLocations(n, aConfigurations.GetTimeSlot(), aConfigurations.GetDimension(),
                                            *locations);
    data->SetLocations(*locations);

    // TODO: Implement the Generate Descriptors phase.
    // TESTING_CHECK(magma_dmalloc_cpu(&h_obs, opts.num_loc));
    // omp_set_num_threads(opts.omp_numthreads);



    // Generate Descriptors phase
    // auto linear_algebra_solver = linearAlgebra::LinearAlgebraFactory<T>::CreateLinearAlgebraSolver(EXACT_DENSE);
    // linear_algebra_solver->GenerateSyntheticData(aConfigurations, data, aKernel);

    return data;
}

template<typename T>
void SyntheticGenerator<T>::ReleaseInstance() {
    if (mpInstance != nullptr) {
        mpInstance = nullptr;
    }
}

template<typename T> SyntheticGenerator<T> *SyntheticGenerator<T>::mpInstance = nullptr;
