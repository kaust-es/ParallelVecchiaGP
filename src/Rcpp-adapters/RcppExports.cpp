// Copyright (c) 2017-2025 King Abdullah University of Science and Technology,
// All rights reserved.
// VecchiaGB is a software package, provided by STSDS group at King Abdullah University of Science and Technology (KAUST).

/**
 * @file RcppExports.cpp
 * @brief Rcpp export definitions for VecchiaGB module.
 * @version 1.0.0
 * @author Generated for R wrapper
 * @date 2025-01-01
**/

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

/**
 * @brief Rcpp module boot function for VecchiaGB.
 * @return A SEXP represents the Rcpp module.
 */
RcppExport SEXP _rcpp_module_boot_VecchiaGB();

/**
 * @brief Array of R function call entries.
 */
static const R_CallMethodDef CallEntries[] = {
        {"_rcpp_module_boot_VecchiaGB", (DL_FUNC) &_rcpp_module_boot_VecchiaGB, 0},
        {nullptr,                        nullptr,                                0}
};

/**
 * @brief R initialization function for VecchiaGB module.
 * @param dll The DllInfo structure.
 */
RcppExport void R_init_VecchiaGB(DllInfo *dll) {
    R_registerRoutines(dll, nullptr, CallEntries, nullptr, nullptr);
    R_useDynamicSymbols(dll, FALSE);
}

