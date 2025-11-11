// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file Point.cpp
 * @brief Implementation file for the Point class.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sohayla Abouzeid
 * @author Qilong Pan
 * @date 2025-09-29
**/

#include <data-units/Point.hpp>
#include <iostream>

using namespace vecchia::dataunits;

template<typename T>
Point<T>::Point() {
    ResetToZero(-1);
}

template<typename T>
Point<T>::Point(const T aCoordinates[3], const int &aCluster) {
    for (int i = 0; i < 3; i++) {
        mCoordinates[i] = aCoordinates[i];
    }
    mCluster = aCluster;
}

template<typename T>
bool Point<T>::operator==(const Point<T> &aPoint) const {
    for (int i = 0; i < 3; i++) {
        if (mCoordinates[i] != aPoint.mCoordinates[i]) {
            return false;
        }
    }
    return true;
}

template<typename T>
Point<T>& Point<T>::operator+=(const Point<T> &aPoint) {
    for (int i = 0; i < 3; i++) {
        mCoordinates[i] += aPoint.mCoordinates[i];
    }
    return *this;
}

template<typename T>
Point<T>& Point<T>::operator/=(const int &aCardinality) {
    for (int i = 0; i < 3; i++) {
        mCoordinates[i] /= static_cast<T>(aCardinality);
    }
    return *this;
}

template<typename T>
void Point<T>::SetCoordinates(const T aCoordinates[3]) {
    for (int i = 0; i < 3; i++) {
        mCoordinates[i] = aCoordinates[i];
    }
}

template<typename T>
const T* Point<T>::GetCoordinates() const {
    return mCoordinates;
}

template<typename T>
void Point<T>::SetCluster(const int &aCluster) {
    mCluster = aCluster;
}

template<typename T>
int Point<T>::GetCluster() const {
    return mCluster;
}

template<typename T>
void Point<T>::ResetToZero(const int &aCluster) {
    for (int i = 0; i < 3; i++) {
        mCoordinates[i] = 0;
    }
    mCluster = aCluster;
}

template<typename T>
void Point<T>::Print() const {
    for (int i = 0; i < 3; i++) {
        std::cout << mCoordinates[i] << " ";
    }
    std::cout << mCluster << std::endl;
}