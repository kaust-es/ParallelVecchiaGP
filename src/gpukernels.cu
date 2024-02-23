#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpukernels.h"

#define PI (3.141592653589793)
#define earthRadiusKm (6371.0)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gpuDotProducts
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernel to calculate multiple dot products in parallel
__global__ void DgpuDotProducts_kernel(
    double *a, double *b,
    double *results,
    int numDotProducts,
    int vectorSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Iterate over dot products
    for (int i = tid; i < numDotProducts; i += blockDim.x * gridDim.x)
    {
        double result = 0.0f;

        // Perform the dot product in parallel
        for (int j = 0; j < vectorSize; ++j)
        {
            result += a[i * vectorSize + j] * b[i * vectorSize + j];
        }

        // Store the result
        results[i] = result;
    }
}

void DgpuDotProducts(
    double *a, double *b,
    double *results,
    int numDotProducts,
    int vectorSize,
    cudaStream_t stream)
{

    int block_dim = 256;
    int grid_dim = (numDotProducts + block_dim - 1) / block_dim;

    DgpuDotProducts_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, results, numDotProducts, vectorSize);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gpuDotProducts - strided version
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernel to calculate multiple dot products in parallel

__global__ void DgpuDotProducts_Strided_kernel(
    double *a, double *b, double *results,
    int numDotProducts,
    int vectorSize,
    int lddvectorSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Iterate over dot products
    for (int i = tid; i < numDotProducts; i += blockDim.x * gridDim.x)
    {
        double result = 0.0f;

        // Perform the dot product in parallel
        for (int j = 0; j < vectorSize; ++j)
        {
            result += a[i * lddvectorSize + j] * b[i * lddvectorSize + j];
        }

        // Store the result
        results[i] = result;
    }
}

void DgpuDotProducts_Strided(double *a, double *b,
                             double *results,
                             int numDotProducts,
                             int vectorSize,
                             int lddvectorSize,
                             cudaStream_t stream)
{

    int block_dim = 256;
    int grid_dim = (numDotProducts + block_dim - 1) / block_dim;

    DgpuDotProducts_Strided_kernel<<<grid_dim, block_dim, 0, stream>>>(a, b, results, numDotProducts, vectorSize, lddvectorSize);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Covariance matrix generation 1/2 3/2 5/2 - strided version
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Dcmg_matern12_strided_1d_batched_kernel(
    double *A,
    int m, int n, int lddm,
    int Acon,
    int repeatNum, int batchCount,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    double localtheta0, double localtheta1,
    int distance_metric)
/*!
 * Returns covariance matrix tile using the aforementioned kernel.
 * @param[in] A: 1D array which saves the matrix entries by column.
 * @param[in] m: number of rows of tile.
 * @param[in] n: number of columns of tile.
 * @param[in] lddm: leading dimension of columns of tile.
 * @param[in] m0: Global row start point of tile.
 * @param[in] n0: Global column start point of tile.
 * @param[in] l1_x_cuda: value of x-axis of locaton vector l1.
 * @param[in] l1_y_cuda: value of y-axis of locaton vector l1.
 * @param[in] l2_x_cuda: value of x-axis of locaton vector l2.
 * @param[in] l2_y_cuda: value of y-axis of locaton vector l2.
 * @param[in] localtheta: there are three parameters to define this kernel.
 * @param[in] distance_metric: unused arguments
 * */
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long idy = blockIdx.y * blockDim.y + threadIdx.y;

    int mLocal = idx % m;
    int nLocal = idx / m;

    if (mLocal >= m || nLocal >= n)
        return;
    for (int i = 0; i < repeatNum; ++i)
    {
        if (idy >= batchCount)
        {
            return;
        }
        double *ALocal = A + idy * lddm * n;
        double *l1_x_cudaLocal = l1_x_cuda + idy * Acon;
        double *l1_y_cudaLocal = l1_y_cuda + idy * Acon;
        double *l2_x_cudaLocal = l2_x_cuda + idy * n;
        double *l2_y_cudaLocal = l2_y_cuda + idy * n;
        double scaled_distance = 0.0;
        // double expr1 = 0.0;
        double sigma_square = localtheta0;
        scaled_distance = sqrt(
                              (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) * (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) +
                              (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal]) * (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal])) /
                          localtheta1;

        ALocal[mLocal + nLocal * lddm] = sigma_square *
                                         exp(-(scaled_distance)); // power-exp kernel
        idy += blockDim.y * gridDim.y;
    }
}

__global__ void Dcmg_matern32_strided_1d_batched_kernel(
    double *A,
    int m, int n, int lddm,
    int Acon,
    int repeatNum, int batchCount,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    double localtheta0, double localtheta1,
    int distance_metric)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long idy = blockIdx.y * blockDim.y + threadIdx.y;

    int mLocal = idx % m;
    int nLocal = idx / m;

    if (mLocal >= m || nLocal >= n)
        return;
    for (int i = 0; i < repeatNum; ++i)
    {
        if (idy >= batchCount)
            return;
        double *ALocal = A + idy * lddm * n;
        double *l1_x_cudaLocal = l1_x_cuda + idy * Acon;
        double *l1_y_cudaLocal = l1_y_cuda + idy * Acon;
        double *l2_x_cudaLocal = l2_x_cuda + idy * n;
        double *l2_y_cudaLocal = l2_y_cuda + idy * n;
        double scaled_distance = 0.0;
        double sigma_square = localtheta0;
        scaled_distance = sqrt(
                              (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) * (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) +
                              (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal]) * (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal])) /
                          localtheta1;
        ALocal[mLocal + nLocal * lddm] = sigma_square *
                                         (1 + scaled_distance) *
                                         exp(-scaled_distance);
        idy += blockDim.y * gridDim.y;
    }
}

__global__ void Dcmg_matern52_strided_1d_batched_kernel(
    double *A,
    int m, int n, int lddm,
    int Acon,
    int repeatNum, int batchCount,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    double localtheta0, double localtheta1,
    int distance_metric)
{
    // iterate all the submatrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // iterate the number of independent block
    long long idy = blockIdx.y * blockDim.y + threadIdx.y;

    int mLocal = idx % m;
    int nLocal = idx / m;

    if (mLocal >= m || nLocal >= m)
        return;

    for (int i = 0; i < repeatNum; ++i)
    {
        if (idy >= batchCount)
            return;
        // the kernel computing
        double *ALocal = A + idy * lddm * n;
        double *l1_x_cudaLocal = l1_x_cuda + idy * Acon;
        double *l1_y_cudaLocal = l1_y_cuda + idy * Acon;
        double *l2_x_cudaLocal = l2_x_cuda + idy * n;
        double *l2_y_cudaLocal = l2_y_cuda + idy * n;
        double scaled_distance = 0.0;
        double sigma_square = localtheta0;
        scaled_distance = sqrt(
                              (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) * (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) +
                              (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal]) * (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal])) /
                          localtheta1;
        ALocal[mLocal + nLocal * lddm] = sigma_square *
                                         (1 + scaled_distance + pow(scaled_distance, 2) / 3) *
                                         exp(-scaled_distance);
        idy += blockDim.y * gridDim.y;
    }
}

__global__ void Dcmg_powexp_strided_1d_batched_kernel(
    double *A,
    int m, int n, int lddm,
    int Acon,
    int repeatNum, int batchCount,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    double localtheta0, double localtheta1,
    double localtheta2,
    int distance_metric)
{
    // iterate all the submatrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // iterate the number of independent block
    long long idy = blockIdx.y * blockDim.y + threadIdx.y;

    int mLocal = idx % m;
    int nLocal = idx / m;

    if (mLocal >= m || nLocal >= m)
        return;

    for (int i = 0; i < repeatNum; ++i)
    {
        if (idy >= batchCount)
            return;
        // the kernel computing
        double *ALocal = A + idy * lddm * n;
        double *l1_x_cudaLocal = l1_x_cuda + idy * Acon;
        double *l1_y_cudaLocal = l1_y_cuda + idy * Acon;
        double *l2_x_cudaLocal = l2_x_cuda + idy * n;
        double *l2_y_cudaLocal = l2_y_cuda + idy * n;
        double distance = 0.0;
        double expr1 = 0.0;
        double sigma_square = localtheta0;
        distance = sqrt(
            (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) * (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) +
            (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal]) * (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal]));

        expr1 = pow(distance, localtheta2);
        ALocal[mLocal + nLocal * lddm] = sigma_square *
                                         exp(-(expr1 / localtheta1));
        idy += blockDim.y * gridDim.y;
    }
}

__global__ void Dcmg_powexp_strided_1d_batched_kernel_earth_distance(
    double *A,
    int m, int n, int lddm,
    int Acon,
    int repeatNum, int batchCount,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    double localtheta0, double localtheta1,
    double localtheta2,
    int distance_metric)
{
    // iterate all the submatrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // iterate the number of independent block
    long long idy = blockIdx.y * blockDim.y + threadIdx.y;

    int mLocal = idx % m;
    int nLocal = idx / m;

    if (mLocal >= m || nLocal >= m)
        return;

    for (int i = 0; i < repeatNum; ++i)
    {
        if (idy >= batchCount)
            return;
        // the kernel computing
        double lat1r, lon1r, lat2r, lon2r, u, v;
        double expr = 0.0;
        double expr1 = 0.0;
        double sigma_square = localtheta0;
        double *ALocal = A + idy * lddm * n;
        double *l1_x_cudaLocal = l1_x_cuda + idy * Acon;
        double *l1_y_cudaLocal = l1_y_cuda + idy * Acon;
        double *l2_x_cudaLocal = l2_x_cuda + idy * n;
        double *l2_y_cudaLocal = l2_y_cuda + idy * n;

        lat1r = l1_x_cudaLocal[mLocal] * PI / 180;
        lon1r = l1_y_cudaLocal[mLocal] * PI / 180;
        lat2r = l2_x_cudaLocal[nLocal] * PI / 180;
        lon2r = l2_y_cudaLocal[nLocal] * PI / 180;
        u = sin((lat2r - lat1r) / 2.);
        v = sin((lon2r - lon1r) / 2.);
        expr = 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
        expr1 = pow(expr / 2523.64, localtheta2); // /9348.317 (soil data) 2523.64 (wind speed)
        ALocal[mLocal + nLocal * lddm] = sigma_square *
                                         exp(-(expr1 / localtheta1));
        idy += blockDim.y * gridDim.y;
    }
}

__global__ void Dcmg_powexp_nugget_strided_1d_batched_kernel(
    double *A,
    int m, int n, int lddm,
    int Acon,
    int repeatNum, int batchCount,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    double localtheta0, double localtheta1,
    double localtheta2, double localtheta3,
    int distance_metric)
{
    // iterate all the submatrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // iterate the number of independent block
    long long idy = blockIdx.y * blockDim.y + threadIdx.y;

    int mLocal = idx % m;
    int nLocal = idx / m;

    if (mLocal >= m || nLocal >= m)
        return;

    for (int i = 0; i < repeatNum; ++i)
    {
        if (idy >= batchCount)
            return;
        // the kernel computing
        double *ALocal = A + idy * lddm * n;
        double *l1_x_cudaLocal = l1_x_cuda + idy * Acon;
        double *l1_y_cudaLocal = l1_y_cuda + idy * Acon;
        double *l2_x_cudaLocal = l2_x_cuda + idy * n;
        double *l2_y_cudaLocal = l2_y_cuda + idy * n;
        double distance = 0.0;
        double expr1 = 0.0;
        double sigma_square = localtheta0;
        distance = sqrt(
            (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) * (l2_x_cudaLocal[nLocal] - l1_x_cudaLocal[mLocal]) +
            (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal]) * (l2_y_cudaLocal[nLocal] - l1_y_cudaLocal[mLocal]));

        expr1 = pow(distance, localtheta2);
        if (distance == 0)
        {
            ALocal[mLocal + nLocal * lddm] = sigma_square + localtheta3;
        }
        else
        {
            ALocal[mLocal + nLocal * lddm] = sigma_square *
                                             exp(-(expr1 / localtheta1));
        }
        idy += blockDim.y * gridDim.y;
    }
}

__global__ void Dcmg_powexp_nugget_strided_1d_batched_kernel_earth_distance(
    double *A,
    int m, int n, int lddm,
    int Acon,
    int repeatNum, int batchCount,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    double localtheta0, double localtheta1,
    double localtheta2, double localtheta3,
    int distance_metric)
{
    // iterate all the submatrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // iterate the number of independent block
    long long idy = blockIdx.y * blockDim.y + threadIdx.y;

    int mLocal = idx % m;
    int nLocal = idx / m;

    if (mLocal >= m || nLocal >= m)
        return;

    for (int i = 0; i < repeatNum; ++i)
    {
        if (idy >= batchCount)
            return;
        // the kernel computing
        double lat1r, lon1r, lat2r, lon2r, u, v;
        double expr = 0.0;
        double expr1 = 0.0;
        double sigma_square = localtheta0;
        double *ALocal = A + idy * lddm * n;
        double *l1_x_cudaLocal = l1_x_cuda + idy * Acon;
        double *l1_y_cudaLocal = l1_y_cuda + idy * Acon;
        double *l2_x_cudaLocal = l2_x_cuda + idy * n;
        double *l2_y_cudaLocal = l2_y_cuda + idy * n;

        lat1r = l1_x_cudaLocal[mLocal] * PI / 180;
        lon1r = l1_y_cudaLocal[mLocal] * PI / 180;
        lat2r = l2_x_cudaLocal[nLocal] * PI / 180;
        lon2r = l2_y_cudaLocal[nLocal] * PI / 180;
        u = sin((lat2r - lat1r) / 2.);
        v = sin((lon2r - lon1r) / 2.);
        expr = 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
        expr1 = pow(expr / 2523.64, localtheta2); // /9348.317 (soil data) 2523.64 (wind speed)
        if (expr == 0)
        {
            ALocal[mLocal + nLocal * lddm] = sigma_square + localtheta3;
        }
        else
        {
            ALocal[mLocal + nLocal * lddm] = sigma_square *
                                             exp(-(expr1 / localtheta1));
        }
        idy += blockDim.y * gridDim.y;
    }
}

/****************c/c++ wrapped function*************************/

void cudaDcmg_matern135_2_strided(
    double *A,
    int m, int n, int lddm, int Acon,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    const double *localtheta, int distance_metric,
    int batchCount_gpu,
    cudaStream_t stream)
{

    int maxBlockNum = 5120; // 40960;
    int repeatNum = (batchCount_gpu - 1) / maxBlockNum + 1;
    const int matrixSize = m * n;
    dim3 dimBlock(min(matrixSize, 128), 1);
    dim3 dimGrid((matrixSize - 1) / dimBlock.x + 1, maxBlockNum);
    if (localtheta[2] == 0.5)
    {
        Dcmg_matern12_strided_1d_batched_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, Acon, repeatNum, batchCount_gpu,
            l1_x_cuda, l1_y_cuda,
            l2_x_cuda, l2_y_cuda,
            localtheta[0], localtheta[1],
            distance_metric);
        return;
    }
    else if (localtheta[2] == 1.5)
    {
        Dcmg_matern32_strided_1d_batched_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, Acon, repeatNum, batchCount_gpu,
            l1_x_cuda, l1_y_cuda,
            l2_x_cuda, l2_y_cuda,
            localtheta[0], localtheta[1],
            distance_metric);
        return;
    }
    else if (localtheta[2] == 2.5)
    {
        Dcmg_matern52_strided_1d_batched_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, Acon, repeatNum, batchCount_gpu,
            l1_x_cuda, l1_y_cuda,
            l2_x_cuda, l2_y_cuda,
            localtheta[0], localtheta[1],
            distance_metric);
        return;
    }
    else
    {
        fprintf(stderr, "Other smoothness setting are still developing. \n");
        exit(0);
    }
}

void cudaDcmg_powexp_strided(
    double *A,
    int m, int n,
    int lddm, int Acon,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    const double *localtheta, int distance_metric,
    int batchCount_gpu,
    cudaStream_t stream)
{

    int maxBlockNum = 5120;
    const int matrixSize = m * n;
    int repeatNum = (batchCount_gpu - 1) / maxBlockNum + 1;
    dim3 dimBlock(min(matrixSize, 128), 1);
    dim3 dimGrid((matrixSize - 1) / dimBlock.x + 1, maxBlockNum);

    if (distance_metric == 0)
    {
        Dcmg_powexp_strided_1d_batched_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, Acon,
            repeatNum, batchCount_gpu,
            // int m0, int n0,
            l1_x_cuda, l1_y_cuda,
            l2_x_cuda, l2_y_cuda,
            localtheta[0], localtheta[1],
            localtheta[2],
            distance_metric);
    }
    else
    {
        Dcmg_powexp_strided_1d_batched_kernel_earth_distance<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, Acon,
            repeatNum, batchCount_gpu,
            // int m0, int n0,
            l1_x_cuda, l1_y_cuda,
            l2_x_cuda, l2_y_cuda,
            localtheta[0], localtheta[1],
            localtheta[2],
            distance_metric);
    }
}

void cudaDcmg_powexp_nugget_strided(
    double *A,
    int m, int n,
    int lddm, int Acon,
    // int m0, int n0,
    double *l1_x_cuda, double *l1_y_cuda,
    double *l2_x_cuda, double *l2_y_cuda,
    const double *localtheta, int distance_metric,
    int batchCount_gpu,
    cudaStream_t stream)
{

    int maxBlockNum = 5120;
    const int matrixSize = m * n;
    int repeatNum = (batchCount_gpu - 1) / maxBlockNum + 1;
    dim3 dimBlock(min(matrixSize, 128), 1);
    dim3 dimGrid((matrixSize - 1) / dimBlock.x + 1, maxBlockNum);

    if (distance_metric == 0)
    {
        Dcmg_powexp_nugget_strided_1d_batched_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, Acon,
            repeatNum, batchCount_gpu,
            // int m0, int n0,
            l1_x_cuda, l1_y_cuda,
            l2_x_cuda, l2_y_cuda,
            localtheta[0], localtheta[1],
            localtheta[2], localtheta[3],
            distance_metric);
    }
    else
    {
        Dcmg_powexp_nugget_strided_1d_batched_kernel_earth_distance<<<dimGrid, dimBlock, 0, stream>>>(
            A, m, n, lddm, Acon,
            repeatNum, batchCount_gpu,
            // int m0, int n0,
            l1_x_cuda, l1_y_cuda,
            l2_x_cuda, l2_y_cuda,
            localtheta[0], localtheta[1],
            localtheta[2], localtheta[3],
            distance_metric);
    }
}
