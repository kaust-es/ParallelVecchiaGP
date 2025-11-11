
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file UnivariateMaternDdbetaBeta.cpp
 * @brief Implementation of the UnivariateMaternDdbetaBeta kernel.
 * @version 1.1.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2023-04-14
**/

#include <kernels/concrete/UnivariateMaternDdbetaBeta.hpp>


using namespace vecchia::kernels;
using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
UnivariateMaternDdbetaBeta<T>::UnivariateMaternDdbetaBeta() {
    this->mP = 1;
    this->mParametersNumber = 3;
}

template<typename T>
Kernel<T> *UnivariateMaternDdbetaBeta<T>::Create() {
    KernelsConfigurations::GetParametersNumberKernelMap()["UnivariateMaternDdbetaBeta"] = 3;
    return new UnivariateMaternDdbetaBeta();
}

namespace vecchia::kernels {
    template<typename T> bool UnivariateMaternDdbetaBeta<T>::plugin_name = plugins::PluginRegistry<vecchia::kernels::Kernel<T>>::Add(
            "UnivariateMaternDdbetaBeta", UnivariateMaternDdbetaBeta<T>::Create);
}

template<typename T>
void
UnivariateMaternDdbetaBeta<T>::GenerateCovarianceMatrix(double *apMatrixA, const int &aRowsNumber, const int &aColumnsNumber,
                                                        const int &aRowOffset, const int &aColumnOffset,
                                                        Locations<T> &aLocation1, Locations<T> &aLocation2,
                                                        Locations<T> &aLocation3, double *apLocalTheta,
                                                        const int &aDistanceMetric) {

    int i, j;
    int i0 = aRowOffset;
    int j0;
    T expr;
    T con;
    T beta_expr;
    T beta_expr_prime;
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
                beta_expr = -apLocalTheta[2] / apLocalTheta[1] * pow(expr, apLocalTheta[2]) *
                            gsl_sf_bessel_Knu(apLocalTheta[2], expr) - pow(expr, apLocalTheta[2]) *
                                                                      (apLocalTheta[2] / expr *
                                                                       gsl_sf_bessel_Knu(apLocalTheta[2], expr) -
                                                                       gsl_sf_bessel_Knu(apLocalTheta[2] + 1, expr)) *
                                                                      expr / apLocalTheta[1];
                beta_expr_prime = -apLocalTheta[2] / apLocalTheta[1] * pow(expr, apLocalTheta[2]) *
                                  (apLocalTheta[2] / expr * gsl_sf_bessel_Knu(apLocalTheta[2], expr) -
                                   gsl_sf_bessel_Knu(apLocalTheta[2] + 1, expr)) - pow(expr, apLocalTheta[2]) * (-0.5 *
                                                                                                               ((apLocalTheta[2] /
                                                                                                                 expr *
                                                                                                                 gsl_sf_bessel_Knu(
                                                                                                                         apLocalTheta[2],
                                                                                                                         expr) -
                                                                                                                 gsl_sf_bessel_Knu(
                                                                                                                         apLocalTheta[2] +
                                                                                                                         1,
                                                                                                                         expr)) -
                                                                                                                pow(expr,
                                                                                                                    apLocalTheta[2]) +
                                                                                                                (apLocalTheta[2] /
                                                                                                                 expr *
                                                                                                                 gsl_sf_bessel_Knu(
                                                                                                                         apLocalTheta[2],
                                                                                                                         expr) -
                                                                                                                 gsl_sf_bessel_Knu(
                                                                                                                         apLocalTheta[2] +
                                                                                                                         1,
                                                                                                                         expr)) -
                                                                                                                pow(expr,
                                                                                                                    apLocalTheta[2]))) *
                                                                                  expr / apLocalTheta[1];
                apMatrixA[i + j * aRowsNumber] = (apLocalTheta[2] / pow(apLocalTheta[1], 2) * pow(expr, apLocalTheta[2]) *
                                                  gsl_sf_bessel_Knu(apLocalTheta[2], expr) -
                                                  apLocalTheta[2] / apLocalTheta[1] * beta_expr +
                                                  2 * expr / pow(apLocalTheta[1], 2) * pow(expr, apLocalTheta[2]) *
                                                  (apLocalTheta[2] / expr * gsl_sf_bessel_Knu(apLocalTheta[2], expr) -
                                                   gsl_sf_bessel_Knu(apLocalTheta[2] + 1, expr)) -
                                                  expr / apLocalTheta[1] * beta_expr_prime) * sigma_square *
                                                 con; // derivative with respect to beta
            }
            j0++;
        }
        i0++;
    }
}