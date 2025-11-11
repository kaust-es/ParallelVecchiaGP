
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file UnivariateMaternDdsigmaSquareBeta.cpp
 * @brief Implementation of the UnivariateMaternDdsigmaSquareBeta kernel.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2023-04-14
**/

#include <kernels/concrete/UnivariateMaternDdsigmaSquareBeta.hpp>


using namespace vecchia::kernels;
using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
UnivariateMaternDdsigmaSquareBeta<T>::UnivariateMaternDdsigmaSquareBeta() {
    this->mP = 1;
    this->mParametersNumber = 3;
}

template<typename T>
Kernel<T> *UnivariateMaternDdsigmaSquareBeta<T>::Create() {
    KernelsConfigurations::GetParametersNumberKernelMap()["UnivariateMaternDdsigmaSquareBeta"] = 3;
    return new UnivariateMaternDdsigmaSquareBeta();
}

namespace vecchia::kernels {
    template<typename T> bool UnivariateMaternDdsigmaSquareBeta<T>::plugin_name = plugins::PluginRegistry<vecchia::kernels::Kernel<T>>::Add(
            "UnivariateMaternDdsigmaSquareBeta", UnivariateMaternDdsigmaSquareBeta<T>::Create);
}

template<typename T>
void UnivariateMaternDdsigmaSquareBeta<T>::GenerateCovarianceMatrix(double *apMatrixA, const int &aRowsNumber,
                                                                    const int &aColumnsNumber, const int &aRowOffset,
                                                                    const int &aColumnOffset, Locations<T> &aLocation1,
                                                                    Locations<T> &aLocation2, Locations<T> &aLocation3,
                                                                    double *apLocalTheta, const int &aDistanceMetric) {

    int i, j;
    int i0 = aRowOffset;
    int j0;
    T expr;
    T con;
    T beta_expr;
    con = pow(2, (apLocalTheta[2] - 1)) * tgamma(apLocalTheta[2]);
    con = 1.0 / con;
    int flag = aLocation1.GetLocationZ() == nullptr ? 0 : 1;

    for (i = 0; i < aRowsNumber; i++) {
        j0 = aColumnOffset;
        for (j = 0; j < aColumnsNumber; j++) {
            expr = DistanceCalculationHelpers<T>::CalculateDistance(aLocation1, aLocation2, i0, j0, aDistanceMetric,
                                                                    flag) / apLocalTheta[1];
            if (expr == 0) {
                apMatrixA[i + j * aRowsNumber] = 0.0;
            } else {
                // derivative with respect to sigma square and beta
                beta_expr = -apLocalTheta[2] / apLocalTheta[1] * pow(expr, apLocalTheta[2]) *
                            gsl_sf_bessel_Knu(apLocalTheta[2], expr) - pow(expr, apLocalTheta[2]) *
                                                                      (apLocalTheta[2] / expr *
                                                                       gsl_sf_bessel_Knu(apLocalTheta[2], expr) -
                                                                       gsl_sf_bessel_Knu(apLocalTheta[2] + 1, expr)) *
                                                                      expr / apLocalTheta[1];
                apMatrixA[i + j * aRowsNumber] = con * beta_expr;
            }
            j0++;
        }
        i0++;
    }
}