#ifndef LLG_H
#define LLG_H

#include <cmath>

/*
dot product strided version
*/
cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const double *x, int incx,
                          const double *y, int incy,
                          double *result)
{
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const float *x, int incx,
                          const float *y, int incy,
                          float *result)
{
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const cuComplex *x, int incx,
                          const cuComplex *y, int incy,
                          cuComplex *result)
{
  return cublasCdotc(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
                          const cuDoubleComplex *x, int incx,
                          const cuDoubleComplex *y, int incy,
                          cuDoubleComplex *result)
{
  return cublasZdotc(handle, n, x, incx, y, incy, result);
}

/*
determinant for log(det(A)) = log(det(L)det(L^T))
strided version
*/

template <class T>
void core_Xlogdet(T *L, int An, int ldda, T *logdet_result_h)
{
  T *L_h = (T *)malloc(sizeof(T) * An * ldda);
  cudaMemcpy(L_h, L, sizeof(T) * An * ldda, cudaMemcpyDeviceToHost);
  *logdet_result_h = 0;
  for (int i = 0; i < An; i++)
  {
    // printf("%d L diagnal value %lf\n", i, L_h[i + i * ldda]);
    // printf("%d the value is %lf \n", i, * logdet_result_h);
    // printf("%d the value is %p \n", i, logdet_result_h);
    if (L_h[i + i * ldda] > 0)
      *logdet_result_h += log(L_h[i + i * ldda] * L_h[i + i * ldda]);
  }
  // printf("the value is %lf \n", * logdet_result_h);
  // printf("-----------------------------------");
  free(L_h);
}

/*
non-strided version

template <class T>
void core_Xlogdet(T **L, int An, int lda, T *logdet_result_h)
{
  T *L_h = (T *)malloc(sizeof(T) * An * lda);
  cudaMemcpy(L_h, L, sizeof(T) * An * lda, cudaMemcpyDeviceToHost);
  for (int i = 0; i < An; i++)
  {
    // printf("L diagnal value %lf\n", L_h[i + i * m]);
    if (L_h[i + i * lda] > 0)
      * logdet_result_h += log(L_h[i + i * lda] * L_h[i + i * lda]);
  }
  // printf("the value is %lf \n", logdet_result_h[0]);
  free(L_h);
}
*/

// /*
// GEAD; general addition for matrix/vector
// */

// cublasStatus_t cublasXgeam(cublasHandle_t handle,
//                            cublasOperation_t transa, cublasOperation_t transb,
//                            int m, int n,
//                            const float *alpha,
//                            const float *A, int lda,
//                            const float *beta,
//                            const float *B, int ldb,
//                            float *C, int ldc)
// {
//   return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
// }

// cublasStatus_t cublasXgeam(cublasHandle_t handle,
//                            cublasOperation_t transa, cublasOperation_t transb,
//                            int m, int n,
//                            const double *alpha,
//                            const double *A, int lda,
//                            const double *beta,
//                            const double *B, int ldb,
//                            double *C, int ldc)
// {
//   return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
// }

// cublasStatus_t cublasXgeam(cublasHandle_t handle,
//                            cublasOperation_t transa, cublasOperation_t transb,
//                            int m, int n,
//                            const cuComplex *alpha,
//                            const cuComplex *A, int lda,
//                            const cuComplex *beta,
//                            const cuComplex *B, int ldb,
//                            cuComplex *C, int ldc)
// {
//   return cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
// }

// cublasStatus_t cublasXgeam(cublasHandle_t handle,
//                            cublasOperation_t transa, cublasOperation_t transb,
//                            int m, int n,
//                            const cuDoubleComplex *alpha,
//                            const cuDoubleComplex *A, int lda,
//                            const cuDoubleComplex *beta,
//                            const cuDoubleComplex *B, int ldb,
//                            cuDoubleComplex *C, int ldc)
// {
//   return cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
// }

/*
Some Auxiliary print function
*/
void printLocations(int N, location *locations)
{
  fprintf(stderr, "\n---------------------------------\n");
  for (int i = 0; i < N; i++)
  {
    fprintf(stderr, "%d th location: (%lf, %lf)\n", i, locations->x[i], locations->y[i]);
  }
  fprintf(stderr, "-----------------------------------\n");
}

template <class T>
void printMatrixCPU(int m, int n, T *h_A, int lda, int i)
{
  printf("-------------------------------\n");
  printf("%d batch of all. (CPU)\n", i);
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      // colunm-wised matrix
      // printf("(%d)", i + j * lda);
      printf("%lg ,", (double)h_A[i + j * lda]);
      // printf(", ");
    }
    printf("\n");
  }
  printf("-------------------------------\n");
}

template <class T>
void printVectorCPU(int m, T *h_C, int ldc, int i)
{
  printf("-------------------------------\n");
  printf("%d batch of all. (CPU)\n", i);
  for (int i = 0; i < m; i++)
  {
    // printf("(%d)", i + j * lda);
    printf("%lg ,", (double)h_C[i]);
    // printf(", ");
  }
  printf("\n");
  printf("-------------------------------\n");
}

template <class T>
void printMatrixGPU(int Am, int An, T *d_A, int lda)
{
  printf("-------------------------------\n");
  printf("Convariance matrix in batch. (GPU)\n");
  T *h_A = (T *)malloc(sizeof(T) * An * lda);
  cudaMemcpy(h_A, d_A, sizeof(T) * An * lda, cudaMemcpyDeviceToHost);
  // double sum = 0;
  for (int i = 0; i < Am; i++)
  {
    for (int j = 0; j < An; j++)
    {
      // colunm-wised matrix
      // printf("(%d)", i + j * lda);
      printf("%.10lf ", (double)h_A[i + j * lda]);
      // sum += (double)h_A[i + j * lda];
    }
    printf("\n");
  }
  // printf("The sum is %lf \n", sum);
  printf("-------------------------------\n");
  free(h_A);
}

template <class T>
void printVecGPU(int Cm, int Cn, T *d_C, int lda, int i)
{
  printf("-------------------------------\n");
  printf("%dst batch of all. (GPU) vector\n", i);
  T *h_C = (T *)malloc(sizeof(T) * Cn * lda);
  cudaMemcpy(h_C, d_C, sizeof(T) * Cn * lda, cudaMemcpyDeviceToHost);
  for (int i = 0; i < Cm; i++)
  {
    printf("(%d)", i);
    printf("%lf ", (double)h_C[i]);
  }
  printf("\n-------------------------------\n");
  free(h_C);
}

template <class T>
void printVecGPUv1(int Cm, T *d_C)
{
  printf("-------------------------------\n");
  T *h_C = (T *)malloc(sizeof(T) * Cm);
  cudaMemcpy(h_C, d_C, sizeof(T) * Cm, cudaMemcpyDeviceToHost);
  for (int i = 0; i < Cm; i++)
  {
    // printf("(%d)", i);
    printf("%lf, ", (double)h_C[i]);
  }
  printf("\n-------------------------------\n");
  free(h_C);
}


template<class T>
void Xrand_matrix(long rows, long cols, T* A, long LDA)
{
	long i;
  long size_a = cols * LDA;
  for(i = 0; i < size_a; i++) A[i] = ( (T)rand() ) / (T)RAND_MAX;
}

#endif