/**
 *
 * Copyright (c) 2014, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * ParallelVecchiaGP is a software package provided by KAUST
 **/
/**
 *
 * @file ckernel.cpp
 *
 * Generate 2D locations.
 *
 * @version 1.0.0
 *
 * @author Qilong Pan
 * @date 2024-03-08
 *
 **/
#include <cstdio>
#include <math.h>
// #include <gsl/gsl_sf_bessel.h>
// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_fft_complex.h>
#include <nlopt.h>
#include <stdint.h>
#include "ckernel.h"

#define pow_e (2.71828182845904)
#define PI (3.141592653589793)
#define earthRadiusKm (6371.0)


double uniform_distribution(double rangeLow, double rangeHigh) {
    double myRand = (double) rand() / (double) (1.0 + RAND_MAX);
    double range = rangeHigh - rangeLow;
    double myRand_scaled = (myRand * range) + rangeLow;
    return myRand_scaled;
}

location *GenerateXYLoc(int n, int seed)
//! Generate XY location for exact computation (MORSE)
{
    //initialization
    int i = 0, index = 0, j = 0;
    srand(seed);
    location *locations = (location *) malloc(sizeof(location *));
    //Allocate memory
    locations->x = (double* ) malloc(n * sizeof(double));
    locations->y = (double* ) malloc(n * sizeof(double));
    locations->z = NULL;

    int sqrtn = ceil(sqrt(n));

    int *grid = (int *) calloc((int) sqrtn, sizeof(int));

    for (i = 0; i < sqrtn; i++) {
        grid[i] = i + 1;
		// grid[i] = i + 0.0; // regular
    }

    for (i = 0; i < sqrtn && index < n; i++)
        for (j = 0; j < sqrtn && index < n; j++) {
            locations->x[index] = (grid[i] - 0.5 + uniform_distribution(-0.4, 0.4)) / sqrtn;
            locations->y[index] = (grid[j] - 0.5 + uniform_distribution(-0.4, 0.4)) / sqrtn;
			// grid (x, y)
			// locations->x[index] = (grid[i] + 1.0) / sqrtn;
            // locations->y[index] = (grid[j] + 1.0) / sqrtn;
            index++;
        }
    free(grid);
    zsort_locations(n, locations);
    return locations;
}
