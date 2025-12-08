
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file UnivariateMaternDdnuNu.cpp
 * @brief Implementation of the UnivariateMaternDdnuNu kernel.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2023-04-14
**/

#include <kernels/concrete/UnivariateMaternDdnuNu.hpp>


using namespace vecchia::kernels;
using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
UnivariateMaternDdnuNu<T>::UnivariateMaternDdnuNu() {
    this->mP = 1;
    this->mParametersNumber = 3;
}

template<typename T>
Kernel<T> *UnivariateMaternDdnuNu<T>::Create() {
    KernelsConfigurations::GetParametersNumberKernelMap()["UnivariateMaternDdnuNu"] = 3;
    return new UnivariateMaternDdnuNu();
}

namespace vecchia::kernels {
    template<typename T> bool UnivariateMaternDdnuNu<T>::plugin_name = plugins::PluginRegistry<vecchia::kernels::Kernel<T>>::Add(
            "UnivariateMaternDdnuNu", UnivariateMaternDdnuNu<T>::Create);
}

template<typename T>
void
UnivariateMaternDdnuNu<T>::GenerateCovarianceMatrix(double *apMatrixA, const int &aRowsNumber, const int &aColumnsNumber,
                                                    const int &aRowOffset, const int &aColumnOffset,
                                                    Locations<T> &aLocation1, Locations<T> &aLocation2,
                                                    Locations<T> &aLocation3, double *apLocalTheta,
                                                    const int &aDistanceMetric) {

    int i, j;
    int i0 = aRowOffset;
    int j0;
    T expr;
    T con;
    T nu_expr;
    T nu_expr_dprime;
    double sigma_square = apLocalTheta[0];
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
                nu_expr = (1 - apLocalTheta[2]) * 1 / pow(2, apLocalTheta[2]) * 1 / tgamma(apLocalTheta[2]) *
                          pow(expr, apLocalTheta[2]) * gsl_sf_bessel_Knu(apLocalTheta[2], expr) +
                          pow(2, 1 - apLocalTheta[2]) *
                          (-1 / tgamma(apLocalTheta[2]) * gsl_sf_psi(apLocalTheta[2]) * pow(expr, apLocalTheta[2]) *
                           gsl_sf_bessel_Knu(apLocalTheta[2], expr) + 1 / tgamma(apLocalTheta[2]) *
                                                                     (pow(expr, apLocalTheta[2]) * log(expr) *
                                                                      gsl_sf_bessel_Knu(apLocalTheta[2], expr) +
                                                                      pow(expr, apLocalTheta[2]) *
                                                                      BasselFunction<T>::CalculateDerivativeBesselNu(
                                                                              apLocalTheta[2],
                                                                              expr)));
                nu_expr_dprime = (1 - apLocalTheta[2]) * 1 / pow(2, apLocalTheta[2]) * 1 / tgamma(apLocalTheta[2]) *
                                 pow(expr, apLocalTheta[2]) *
                                 BasselFunction<T>::CalculateDerivativeBesselNu(apLocalTheta[2], expr) +
                                 pow(2, 1 - apLocalTheta[2]) *
                                 (-1 / tgamma(apLocalTheta[2]) * gsl_sf_psi(apLocalTheta[2]) * pow(expr, apLocalTheta[2]) *
                                  BasselFunction<T>::CalculateDerivativeBesselNu(apLocalTheta[2], expr) +
                                  1 / tgamma(apLocalTheta[2]) *
                                  (pow(expr, apLocalTheta[2]) *
                                   log(expr) *
                                   BasselFunction<T>::CalculateDerivativeBesselNu(
                                           apLocalTheta[2],
                                           expr) +
                                   pow(expr, apLocalTheta[2]) *
                                   BasselFunction<T>::CalculateSecondDerivativeBesselNu(apLocalTheta[2], expr)));
                apMatrixA[i + j * aRowsNumber] =
                        (-0.5 * con * pow(expr, apLocalTheta[2]) * gsl_sf_bessel_Knu(apLocalTheta[2], expr) +
                         (1 - apLocalTheta[2]) / 2 * nu_expr -
                         (1 / apLocalTheta[2] + 0.5 * pow(apLocalTheta[2], 2)) * con * pow(expr, apLocalTheta[2]) *
                         gsl_sf_bessel_Knu(apLocalTheta[2], expr) - gsl_sf_psi(apLocalTheta[2]) * nu_expr +
                         log(expr) * nu_expr + nu_expr_dprime) * sigma_square;
            }
            j0++;
        }
        i0++;
    }
}