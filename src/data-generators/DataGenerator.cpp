
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file DataGenerator.cpp
 * @brief Implementation of DataGenerator class
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <data-generators/DataGenerator.hpp>
#include <data-generators/concrete/SyntheticGenerator.hpp>
#include <data-loader/concrete/CSVLoader.hpp>

using namespace vecchia::generators;
using namespace vecchia::dataLoader::csv;
using namespace vecchia::generators::synthetic;
using namespace vecchia::common;

template<typename T>
std::unique_ptr<DataGenerator<T>> DataGenerator<T>::CreateGenerator(configurations::Configurations &aConfigurations) {

    isSynthetic = aConfigurations.GetDataPath().empty();
    if (isSynthetic) {
        return std::unique_ptr<DataGenerator<T>>(SyntheticGenerator<T>::GetInstance());
    } else {
        return std::unique_ptr<DataGenerator<T>>(CSVLoader<T>::GetInstance());
    }
}

template<typename T>
DataGenerator<T>::~DataGenerator() {

    if (isSynthetic) {
        SyntheticGenerator<T>::GetInstance()->ReleaseInstance();
    } else {
        CSVLoader<T>::GetInstance()->ReleaseInstance();
    }
}

template<typename T> bool DataGenerator<T>::isSynthetic = false;
