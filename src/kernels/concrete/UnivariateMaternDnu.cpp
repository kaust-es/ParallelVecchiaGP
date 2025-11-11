
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file UnivariateMaternDnu.cpp
 * @brief Implementation of the UnivariateMaternDnu kernel.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2023-04-14
**/

#include <kernels/concrete/UnivariateMaternDnu.hpp>


using namespace vecchia::kernels;
using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
UnivariateMaternDnu<T>::UnivariateMaternDnu() {
    this->mP = 1;
    this->mParametersNumber = 3;
}

template<typename T>
Kernel<T> *UnivariateMaternDnu<T>::Create() {
    KernelsConfigurations::GetParametersNumberKernelMap()["UnivariateMaternDnu"] = 3;
    return new UnivariateMaternDnu();
}

namespace vecchia::kernels {
    template<typename T> bool UnivariateMaternDnu<T>::plugin_name = plugins::PluginRegistry<vecchia::kernels::Kernel<T>>::Add(
            "UnivariateMaternDnu", UnivariateMaternDnu<T>::Create);
}

template<typename T>
void UnivariateMaternDnu<T>::GenerateCovarianceMatrix(double *apMatrixA, const int &aRowsNumber, const int &aColumnsNumber,
                                                      const int &aRowOffset, const int &aColumnOffset,
                                                      Locations<T> &aLocation1, Locations<T> &aLocation2,
                                                      Locations<T> &aLocation3, double *apLocalTheta,
                                                      const int &aDistanceMetric) {

    int i, j;
    int i0 = aRowOffset;
    int j0;
    T expr;
    T nu_expr;
    double sigma_square = apLocalTheta[0];
    int flag = aLocation1.GetLocationZ() == nullptr ? 0 : 1;

    for (i = 0; i < aRowsNumber; i++) {
        j0 = aColumnOffset;
        for (j = 0; j < aColumnsNumber; j++) {
            expr = DistanceCalculationHelpers<T>::CalculateDistance(aLocation1, aLocation2, i0, j0, aDistanceMetric,
                                                                    flag) / apLocalTheta[1];
            if (expr == 0) {
                apMatrixA[i + j * aRowsNumber] = 0.0;
            } else {
                //derivative with respect to nu
                nu_expr = -2 * log(2.0) * pow(2, -apLocalTheta[2]) * 1 / tgamma(apLocalTheta[2]) *
                          pow(expr, apLocalTheta[2]) * gsl_sf_bessel_Knu(apLocalTheta[2], expr) +
                          pow(2, 1 - apLocalTheta[2]) *
                          (-1 / tgamma(apLocalTheta[2]) * gsl_sf_psi(apLocalTheta[2]) * pow(expr, apLocalTheta[2]) *
                           gsl_sf_bessel_Knu(apLocalTheta[2], expr) + 1 / tgamma(apLocalTheta[2]) *
                                                                     (pow(expr, apLocalTheta[2]) * log(expr) *
                                                                      gsl_sf_bessel_Knu(apLocalTheta[2], expr) +
                                                                      pow(expr, apLocalTheta[2]) *
                                                                      BasselFunction<T>::CalculateDerivativeBesselNu(
                                                                              apLocalTheta[2], expr)));
                apMatrixA[i + j * aRowsNumber] = sigma_square * nu_expr;
            }
            j0++;

        }
        i0++;
    }
}