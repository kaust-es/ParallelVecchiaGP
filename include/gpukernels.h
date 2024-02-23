// aux_operations.h
#ifndef AUX_OPERATIONS_H
#define AUX_OPERATIONS_H

void DgpuDotProducts(double *a, double *b,
                     double *results, int numDotProducts,
                     int vectorSize,
                     cudaStream_t stream);
void DgpuDotProducts_Strided(double *a, double *b,
                             double *results, int numDotProducts, 
                             int vectorSize, int lddvectorSize,
                             cudaStream_t stream);

void cudaDcmg_matern135_2_strided( 
        double *A, 
        int m, int n, int lddm, int Acon,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        const double *localtheta, int distance_metric, 
        int batchCount_gpu,
        cudaStream_t stream);

void cudaDcmg_powexp_strided( 
        double *A, 
        int m, int n, int lddm, int Acon,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        const double *localtheta, int distance_metric, 
        int batchCount_gpu,
        cudaStream_t stream);

void cudaDcmg_powexp_nugget_strided( 
        double *A, 
        int m, int n, int lddm, int Acon,
        // int m0, int n0, 
        double* l1_x_cuda, double* l1_y_cuda, 
        double* l2_x_cuda, double* l2_y_cuda,
        const double *localtheta, int distance_metric, 
        int batchCount_gpu,
        cudaStream_t stream);

#endif // AUX_OPERATIONS_H
