#ifndef CUDA_CODE_CUH
#define CUDA_CODE_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <nvToolsExt.h>

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C"  void end_end_residual(double *data, int M, int N, int m, int *U, int uN);
extern "C" double* causal_order(double *data, int m, int n);


__global__ void compute_mean(double *data, double *mean, int size);
__global__ void compute_covariance_variance(double *xi, double *xj, double *mean_xi, double *mean_xj, double *covariance, double *variance, int size);
__global__ void element_wise_division(double *r, double *constant_std, double *result, int n);
__global__ void compute_residual(double *xi, double *xj, double *scaling_factor, double *residual, int size);
__global__ void compute_std(double *A, double *mean, double *std, int size);
__global__ void compute_log_cosh(double *u, double *log_cosh_sum, int size);
__global__ void compute_u_exp(double *u, double *u_exp_sum, int size);
__global__ void standardize_column(double *A, int m, int n, double *means, double *stds, int startrow);
__global__ void calculate_statistics(double *A, int m, int n, double *means, double *stds);
__global__ void process_column(double *d_X, double *column, int m, int n, int col_idx);
void divonhost(double *sum, double *d_sum, int m);
void divsqtonhost(double *sum, double *d_sum, int m);
void update(double *d_X, double *d_residual_ij, int m, int n, int col);


#endif //CUDA_CODE_CUH
