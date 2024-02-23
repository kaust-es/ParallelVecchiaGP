/**
 *
 * Copyright (c) 2017-2023  King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * ExaGeoStat is a software package provided by KAUST
 **/
/**
 *
 * @file ckernel.h
 *
 * Core functions header file.
 *
 * @version 1.2.0
 *
 * @author Sameh Abdulah
 * @date 2022-11-09
 *
 **/

#ifndef _CKERNEL_H_
#define _CKERNEL_H_
#include <stdint.h>
#include <stdlib.h>
// #define assert(ignore)((void) 0)


#ifdef __cplusplus
extern "C"{
#endif


typedef struct {
    double* x;                ///< Values in X dimension.
    double* y;                ///< Values in Y dimension.
    double* z;                              ///< Values in Z dimension.
} location;



location *GenerateXYLoc(int n, int seed);

static uint32_t Part1By1(uint32_t x)
//! Spread lower bits of input
{
    x &= 0x0000ffff;
    // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff;
    // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333;
    // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555;
    // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}
static uint32_t Compact1By1(uint32_t x)
//! Collect every second bit into lower part of input
{
    x &= 0x55555555;
    // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333;
    // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff;
    // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}


static uint32_t EncodeMorton2(uint32_t x, uint32_t y)
//! Encode two inputs into one
{
    return (Part1By1(y) << 1) + Part1By1(x);
}

static uint32_t DecodeMorton2X(uint32_t code)
//! Decode first input
{
    return Compact1By1(code >> 0);
}

static uint32_t DecodeMorton2Y(uint32_t code)
//! Decode second input
{
    return Compact1By1(code >> 1);
}




double uniform_distribution(double rangeLow, double rangeHigh);


static int compare_uint32(const void *a, const void *b)
//! Compare two uint32_t
{
    uint32_t _a = *(uint32_t *) a;
    uint32_t _b = *(uint32_t *) b;
    if (_a < _b) return -1;
    if (_a == _b) return 0;
    return 1;
}

/********************************************************************/
/*********************Ordering*****************************/
/*******************************************************************/


static void zsort_locations(int n, location *locations)
//! Sort in Morton order (input points must be in [0;1]x[0;1] square])
{
    // Some sorting, required by spatial statistics code
    int i;
    uint16_t x, y;
    uint32_t z[n];
    // Encode data into vector z
    for (i = 0; i < n; i++) {
        x = (uint16_t) (locations->x[i] * (double) UINT16_MAX + .5);
        y = (uint16_t) (locations->y[i] * (double) UINT16_MAX + .5);
        z[i] = EncodeMorton2(x, y);
    }
    // Sort vector z
    qsort(z, n, sizeof(uint32_t), compare_uint32);
    // Decode data from vector z
    for (i = 0; i < n; i++) {
        x = DecodeMorton2X(z[i]);
        y = DecodeMorton2Y(z[i]);
        locations->x[i] = (double) x / (double) UINT16_MAX;
        locations->y[i] = (double) y / (double) UINT16_MAX;
    }
}

struct comb { // location and measure
    uint32_t z;
    double w;
};

static int cmpfunc_loc (const void * a, const void * b) {
    struct comb _a = *(const struct comb *)a;
    struct comb _b = *(const struct comb *)b;
    if(_a.z < _b.z) return -1;
    if(_a.z == _b.z) return 0;
    return 1;
}

static void zsort_reordering(int n, location * locations, double * w)
//! Sort in Morton order (input points must be in [0;1]x[0;1] square])
{
    int i;
    int n_measurement = 1;
    uint16_t x, y;
    struct comb *dat = (struct comb *) malloc (n * n_measurement * sizeof(struct comb));
    // Encode data into vector z
    for(i = 0; i < n; i++)
    {
        x = (uint16_t)(locations->x[i]*(double)UINT16_MAX +.5);
        y = (uint16_t)(locations->y[i]*(double)UINT16_MAX +.5);
        dat[i].z = EncodeMorton2(x, y);
        dat[i].w = w[i];
    }
    // Sort vector z
    qsort(dat, n, sizeof(struct comb), cmpfunc_loc);
    // Decode data from vector z
    for(i = 0; i < n; i++)
    {
        x = DecodeMorton2X(dat[i].z);
        y = DecodeMorton2Y(dat[i].z);
        locations->x[i] = (double)x/(double)UINT16_MAX;
        locations->y[i] = (double)y/(double)UINT16_MAX;
        w[i] = dat[i].w;
    }
}

// random ordering
static void random_reordering(int size, location* loc, double* h_C) {
    int your_seed_value = 42; // Set your desired seed value

    srand(your_seed_value);

    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // Swap x values
        double tempX = loc->x[i];
        loc->x[i] = loc->x[j];
        loc->x[j] = tempX;

        // Swap y values
        double tempY = loc->y[i];
        loc->y[i] = loc->y[j];
        loc->y[j] = tempY;

        // Swap obs values
        double tempObs = h_C[i];
        h_C[i] = h_C[j];
        h_C[j] = tempObs;

    }
}

/********************************************************************/
/*********************Covariance function*****************************/
/*******************************************************************/

//Generate the covariance matrix.
void core_scmg(float *A, int m, int n,
               int m0, int n0,
               location *l1, location *l2,
               double* localtheta, int distance_metric);

void core_dcmg(double* A, int m, int n,
            //    int m0, int n0,
               location *l1, location *l2,
               const double* localtheta, int distance_metric);

void core_dcmg_exp(double* A, int m, int n,
		// int m0, int n0, 
		location* l1,
		location* l2, const double* localtheta, int distance_metric);

void core_sdcmg(double* A, int m, int n,
                int m0, int n0,
                location *l1, location *l2,
                double* localtheta, int distance_metric);


void core_scmg_pow_exp(float *A, int m, int n,
                       int m0, int n0,
                       location *l1, location *l2,
                       double* localtheta, int distance_metric);

void core_dcmg_pow_exp(double* A, int m, int n,
                       int m0, int n0,
                       location *l1, location *l2,
                       double* localtheta, int distance_metric);


void core_sdcmg_pow_exp(double* A, int m, int n,
                        int m0, int n0,
                        location *l1, location *l2,
                        double* localtheta, int distance_metric);

// void core_dcmg_bivariate_parsimonious(double* A, int m, int n,
//                                       int m0, int n0, location *l1,
//                                       location *l2, double* localtheta, int distance_metric);
void core_dcmg_bivariate_parsimonious(double* A, int m, int n,
                                    //   int m0, int n0, 
                                      location *l1,
                                      location *l2, const double* localtheta, int distance_metric);

void core_dcmg_bivariate_parsimonious2(double* A, int m, int n,
                                       int m0, int n0, location *l1,
                                       location *l2, double* localtheta, int distance_metric, int size);

void core_dcmg_bivariate_flexible(double* A, int m, int n,
                                  int m0, int n0, location *l1,
                                  location *l2, double* localtheta, int distance_metric);

float core_smdet(float *A, int m, int n,
                 int m0, int n0);

double core_dmdet(double* A, int m, int n,
                  int m0, int n0);

void core_szcpy(float *Z, int m,
                int m0, float *r);

void core_dzcpy(double* Z, int m,
                int m0, double* r);

float core_sdotp(float *Z, float *dotproduct,
                 int n);

double core_ddotp(double* Z, double* dotproduct,
                  int n);

void core_dlag2s(int m, int n,
                 const double* A, int lda,
                 float *B, int ldb);

void core_slag2d(int m, int n,
                 const float *A, int lda,
                 double* B, int ldb);

void core_sprint(float *A,
                 int m, int n,
                 int m0, int n0);

void core_dprint(double* A,
                 int m, int n,
                 int m0, int n0);

void core_dcmg_nono_stat(double* A, int m, int n,
                         int m0, int n0, location *l1,
                         location *l2, location *lm, double* localtheta,
                         int distance_metric);

void core_dcmg_matern_dsigma_square(double* A, int m, int n,
                                    int m0, int n0, location *l1,
                                    location *l2, double* localtheta, int distance_metric);

void core_dcmg_matern_dnu(double* A, int m, int n,
                          int m0, int n0, location *l1,
                          location *l2, double* localtheta, int distance_metric);

void core_dcmg_matern_dbeta(double* A, int m, int n,
                            int m0, int n0, location *l1,
                            location *l2, double* localtheta, int distance_metric);

void core_dcmg_matern_ddsigma_square(double* A, int m, int n);

void core_dcmg_matern_ddsigma_square_beta(double* A, int m, int n,
                                          int m0, int n0, location *l1,
                                          location *l2, double* localtheta, int distance_metric);

void core_dcmg_matern_ddsigma_square_nu(double* A, int m, int n,
                                        int m0, int n0, location *l1,
                                        location *l2, double* localtheta, int distance_metric);

void core_dcmg_matern_ddbeta_beta(double* A, int m, int n,
                                  int m0, int n0, location *l1,
                                  location *l2, double* localtheta, int distance_metric);

void core_dcmg_matern_ddbeta_nu(double* A, int m, int n,
                                int m0, int n0, location *l1,
                                location *l2, double* localtheta, int distance_metric);

double core_dtrace(double* A, int m, int n,
                   int m0, int n0, double* trace);

double core_ng_loglike(double* Z, double* localtheta,
                       int m);

void core_ng_transform(double* Z, double* nan_flag, double* localtheta,
                       int m);

void core_g_to_ng(double* Z, double* localtheta,
                  int m);

double core_dtrace(double* A, int m, int n,
                   int m0, int n0, double* trace);

void core_dcmg_nuggets(double* A, int m, int n,
                       int m0, int n0, location *l1,
                       location *l2, double* localtheta, int distance_metric);

void core_dcmg_spacetime_bivariate_parsimonious(double* A, int m, int n,
                                                int m0, int n0, location *l1,
                                                location *l2, double* localtheta, int distance_metric);

void core_dcmg_non_stat(double* A, int m, int n, int m0,
                        int n0, location *l1, location *l2, double* localtheta, int distance_metric);

void core_dcmg_spacetime_matern(double* A, int m, int n,
                                int m0, int n0, location *l1,
                                location *l2, double* localtheta, int distance_metric);

void core_dcmg_matern_ddnu_nu(double* A, int m, int n, int m0, int n0, location *l1, location *l2, double* localtheta,
                              int distance_metric);

void core_ng_dcmg(double* A, int m, int n,
                  int m0, int n0, location *l1,
                  location *l2, double* localtheta, int distance_metric);

void core_ng_exp_dcmg(double* A, int m, int n,
                      int m0, int n0, location *l1,
                      location *l2, double* localtheta, int distance_metric);


void core_dcmg_trivariate_parsimonious(double* A, int m, int n,
                                       int m0, int n0, location *l1,
                                       location *l2, double* localtheta, int distance_metric);


#ifdef __cplusplus
}
#endif
#endif

