
// Copyright (c) 2017-2024 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by King Abdullah University of Science and Technology (KAUST).

/**
 * @file BivariateMaternFlexible.cpp
 * @brief Implementation of the BivariateMaternFlexible kernel.
 * @version 1.0.0
 * @author Mahmoud ElKarargy
 * @author Sameh Abdulah
 * @date 2023-04-14
**/

#include <kernels/concrete/BivariateMaternFlexible.hpp>


using namespace vecchia::kernels;
using namespace vecchia::dataunits;
using namespace vecchia::helpers;

template<typename T>
BivariateMaternFlexible<T>::BivariateMaternFlexible() {
    this->mP = 2;
    this->mParametersNumber = 11;
}

template<typename T>
Kernel<T> *BivariateMaternFlexible<T>::Create() {
    KernelsConfigurations::GetParametersNumberKernelMap()["BivariateMaternFlexible"] = 11;
    return new BivariateMaternFlexible();
}

namespace vecchia::kernels {
    template<typename T> bool BivariateMaternFlexible<T>::plugin_name = plugins::PluginRegistry<vecchia::kernels::Kernel<T>>::Add(
            "BivariateMaternFlexible", BivariateMaternFlexible<T>::Create);
}

template<typename T>
void
BivariateMaternFlexible<T>::GenerateCovarianceMatrix(double *apMatrixA, const int &aRowsNumber, const int &aColumnsNumber,
                                                     const int &aRowOffset, const int &aColumnOffset,
                                                     Locations<T> &aLocation1, Locations<T> &aLocation2,
                                                     Locations<T> &aLocation3, double *apLocalTheta,
                                                     const int &aDistanceMetric) {

    int i, j;
    int i0 = aRowOffset;
    int j0;
    T expr1, expr2, expr12;
    T con1, con2, con12, scale12, rho, nu12, sigma_square11, sigma_square22;
    T scale1 = apLocalTheta[0], scale2 = apLocalTheta[1], nu1 = apLocalTheta[4], nu2 = apLocalTheta[5];

    //Remark 1 (c) of Apanasovich et al. (2012)
    scale12 = pow(0.5 * (pow(scale1, -2) + pow(scale2, -2)) + apLocalTheta[2] * (1 - apLocalTheta[3]), -0.5);
    //Theorem 1 (i) of Apanasovich et al. (2012).
    nu12 = 0.5 * (nu1 + nu2) + apLocalTheta[6] * (1 - apLocalTheta[7]);
    //Equation (8) of Apanasovich et al. (2012).
    rho = apLocalTheta[8] * apLocalTheta[9] * apLocalTheta[10] * pow(scale12, 2 * apLocalTheta[6] + (nu1 + nu2)) *
          tgamma(0.5 * (nu1 + nu2) + 1) * tgamma(nu12) / tgamma(nu12 + 1);
    //Equation (8) of Apanasovich et al. (2012).
    sigma_square11 = apLocalTheta[8] * apLocalTheta[8] * pow(scale1, 2 * apLocalTheta[6] + nu1 + nu1) * tgamma(nu1);
    //Equation (8) of Apanasovich et al. (2012).
    sigma_square22 = apLocalTheta[9] * apLocalTheta[9] * pow(scale2, 2 * apLocalTheta[6] + nu2 + nu2) * tgamma(nu2);

    con1 = pow(2, (nu1 - 1)) * tgamma(nu1);
    con1 = 1.0 / con1;
    con1 = sigma_square11 * con1;

    con2 = pow(2, (nu2 - 1)) * tgamma(nu2);
    con2 = 1.0 / con2;
    con2 = sigma_square22 * con2;

    con12 = pow(2, (nu12 - 1)) * tgamma(nu12);
    con12 = 1.0 / con12;
    con12 = rho * con12;

    i0 /= 2;
    int flag = aLocation1.GetLocationZ() == nullptr ? 0 : 1;

    for (i = 0; i < aRowsNumber; i += 2) {
        j0 = aColumnOffset / 2;
        for (j = 0; j < aColumnsNumber; j += 2) {
            expr1 = DistanceCalculationHelpers<T>::CalculateDistance(aLocation1, aLocation2, i0, j0, aDistanceMetric,
                                                                     flag) / scale1;
            expr2 = DistanceCalculationHelpers<T>::CalculateDistance(aLocation1, aLocation2, i0, j0, aDistanceMetric,
                                                                     flag) / scale2;
            expr12 = DistanceCalculationHelpers<T>::CalculateDistance(aLocation1, aLocation2, i0, j0, aDistanceMetric,
                                                                      flag) / scale12;
            if (expr1 == 0) {
                apMatrixA[i + j * aRowsNumber] = apLocalTheta[0];
                if (((i + 1) + j * aRowsNumber) < aRowsNumber * aColumnsNumber) {
                    apMatrixA[(i + 1) + j * aRowsNumber] = rho;
                }
                if ((i + (j + 1) * aRowsNumber) < aRowsNumber * aColumnsNumber) {
                    apMatrixA[i + (j + 1) * aRowsNumber] = rho;
                }
                if (((i + 1) + (j + 1) * aRowsNumber) < aRowsNumber * aColumnsNumber) {
                    apMatrixA[(i + 1) + (j + 1) * aRowsNumber] = apLocalTheta[1];
                }
            } else {
                apMatrixA[i + j * aRowsNumber] = con1 * pow(expr1, nu1) * gsl_sf_bessel_Knu(nu1, expr1);

                if (((i + 1) + j * aRowsNumber) < aRowsNumber * aColumnsNumber) {
                    apMatrixA[(i + 1) + j * aRowsNumber] = con12 * pow(expr12, nu12) * gsl_sf_bessel_Knu(nu12, expr12);
                }
                if ((i + (j + 1) * aRowsNumber) < aRowsNumber * aColumnsNumber) {
                    apMatrixA[i + (j + 1) * aRowsNumber] = con12 * pow(expr12, nu12) * gsl_sf_bessel_Knu(nu12, expr12);
                }
                if (((i + 1) + (j + 1) * aRowsNumber) < aRowsNumber * aColumnsNumber) {
                    apMatrixA[(i + 1) + (j + 1) * aRowsNumber] = con2 * pow(expr2, nu2) * gsl_sf_bessel_Knu(nu2, expr2);
                }
            }
            j0++;
        }
        i0++;
    }
}
