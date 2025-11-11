
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

#include <data-loader/DataLoader.hpp>

using namespace std;

using namespace vecchia::dataLoader;
using namespace vecchia::common;

template<typename T>
std::unique_ptr<VecchiaGBData<T>>
DataLoader<T>::CreateData(configurations::Configurations &aConfigurations, vecchia::kernels::Kernel<T> &aKernel) {

    // create vectors that will be populated with read data.
    vector<T> measurements_vector;
    vector<T> x_locations;
    vector<T> y_locations;
    vector<T> z_locations;

    aKernel.SetPValue(aConfigurations.GetTimeSlot());
    int p = aKernel.GetVariablesNumber();

    //Read the data out of the CSV file.
    this->ReadData(aConfigurations, measurements_vector, x_locations, y_locations, z_locations, p);

    //create data object
    auto data = std::make_unique<VecchiaGBData<T>>(aConfigurations.GetProblemSize() / p,
                                                    aConfigurations.GetDimension(), aConfigurations.GetBlockSize());

    //Initialize the descriptors.
    // auto linear_algebra_solver = linearAlgebra::LinearAlgebraFactory<T>::CreateLinearAlgebraSolver(EXACT_DENSE);
    // linear_algebra_solver->InitiateDescriptors(aConfigurations, *data->GetDescriptorData(), p);
    // linear_algebra_solver->VecchiaGBLaSetTile(EXAGEOSTAT_UPPER_LOWER, 0, 0,
    //                                            data->GetDescriptorData()->GetDescriptor(CHAMELEON_DESCRIPTOR,
    //                                                                                     DESCRIPTOR_C).chameleon_desc);
    //populate data object with read data
    for (int i = 0; i < aConfigurations.GetProblemSize() / p; i++) {
        data->GetLocations()->GetLocationX()[i] = x_locations[i];
        data->GetLocations()->GetLocationY()[i] = y_locations[i];
        if (aConfigurations.GetDimension() != Dimension2D) {
            data->GetLocations()->GetLocationZ()[i] = z_locations[i];
        }
    }
    for (int i = 0; i < aConfigurations.GetProblemSize(); i++) {
        // ((T *) data->GetDescriptorData()->GetDescriptor(CHAMELEON_DESCRIPTOR,
        //                                                 DESCRIPTOR_Z).chameleon_desc->mat)[i] = measurements_vector[i];
    }

    return data;
}
