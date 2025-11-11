
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file UnivariateSpacetimeMaternStationary.cpp
 * @brief Implementation of the UnivariateSpacetimeMaternStationary kernel.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2023-04-14
**/

#include<cmath>

#include <gsl/gsl_sf_bessel.h>

#include <kernels/concrete/UnivariateSpacetimeMaternStationary.hpp>


using namespace vecchia::kernels;
using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
UnivariateSpacetimeMaternStationary<T>::UnivariateSpacetimeMaternStationary() {
    this->mP = 1;
    this->mParametersNumber = 7;
}

template<typename T>
Kernel<T> *UnivariateSpacetimeMaternStationary<T>::Create() {
    KernelsConfigurations::GetParametersNumberKernelMap()["UnivariateSpacetimeMaternStationary"] = 7;
    return new UnivariateSpacetimeMaternStationary();
}

namespace vecchia::kernels {
    template<typename T> bool UnivariateSpacetimeMaternStationary<T>::plugin_name = plugins::PluginRegistry<vecchia::kernels::Kernel<T>>::Add(
            "UnivariateSpacetimeMaternStationary", UnivariateSpacetimeMaternStationary<T>::Create);
}

template<typename T>
void
UnivariateSpacetimeMaternStationary<T>::GenerateCovarianceMatrix(double *apMatrixA, const int &aRowsNumber,
                                                                 const int &aColumnsNumber, const int &aRowOffset,
                                                                 const int &aColumnOffset, Locations<T> &aLocation1,
                                                                 Locations<T> &aLocation2, Locations<T> &aLocation3,
                                                                 double *apLocalTheta, const int &aDistanceMetric) {

    int i, j;
    int i0 = aRowOffset;
    int j0;
    T z0, z1;
    T expr, expr2, expr3, expr4;
    T con;
    double sigma_square = apLocalTheta[0];

    con = pow(2, (apLocalTheta[2] - 1)) * tgamma(apLocalTheta[2]);
    con = 1.0 / con;
    con = sigma_square * con;
    int flag = 1;

    for (i = 0; i < aRowsNumber; i++) {
        j0 = aColumnOffset;
        z0 = aLocation1.GetLocationZ()[i0];
        for (j = 0; j < aColumnsNumber; j++) {
            z1 = aLocation2.GetLocationZ()[j0];

            expr = DistanceCalculationHelpers<T>::CalculateDistance(aLocation1, aLocation2, i0, j0, aDistanceMetric,
                                                                    flag) / apLocalTheta[1];
            expr2 = pow(pow(sqrt(pow(z0 - z1, 2)), 2 * apLocalTheta[4]) / apLocalTheta[3] + 1.0, apLocalTheta[5] / 2.0);
            expr3 = expr / expr2;
            expr4 = pow(pow(sqrt(pow(z0 - z1, 2)), 2 * apLocalTheta[4]) / apLocalTheta[3] + 1.0,
                        apLocalTheta[5] + apLocalTheta[6]);

            if (expr == 0) {
                apMatrixA[i + j * aRowsNumber] = sigma_square / expr4;
            } else {
                // Matern Function
                apMatrixA[i + j * aRowsNumber] =
                        con * pow(expr3, apLocalTheta[2]) * gsl_sf_bessel_Knu(apLocalTheta[2], expr3) / expr4;
            }
            j0++;
        }
        i0++;
    }
}