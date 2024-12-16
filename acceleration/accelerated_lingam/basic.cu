#include "basic.cuh"

__global__ void compute_mean(double *data, double *mean, int size) {

    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    if (i == 0) {
        *mean = 0.0f;
    }

    __syncthreads();

    if (i < size) {
        sum = data[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write the result for this block to global memory
    if (tid == 0) {
        atomicAdd(mean, sdata[0]);
    }

}


__global__ void compute_covariance_variance(double *xi, double *xj, double *mean_xi, double *mean_xj, double *covariance, double *variance, int size)
{
    // we're not dividing by size because we eventually
    //divide the cov and var to get the scaling factor
    // we're also dynamically allocating shared memory for
    // covariance and variance separately.
    extern __shared__ double sharedMemory[];
    double *sharedCov = sharedMemory;
    double *sharedVar = sharedMemory + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx == 0) {
        *covariance = 0.0f;
        *variance = 0.0f;
    }
    __syncthreads();

    if (tid < blockDim.x)
    {
        sharedCov[tid] = 0.0f;
        sharedVar[tid] = 0.0f;
    }

    __syncthreads();

    // this set the coresponding thread position to the
    //component of the summation - basically maps indices in
    //N to blockindices.
    if (idx < size)
    {
        double cov_component = (xi[idx] - *mean_xi) * (xj[idx] - *mean_xj);
        double var_component = fabsf(xj[idx] - *mean_xj) * fabsf(xj[idx] - *mean_xj);
        sharedCov[tid] = cov_component;
        sharedVar[tid] = var_component;
    }

    __syncthreads();

    // reduction in shared memory for covariance and variance
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedCov[tid] += sharedCov[tid + s];
            sharedVar[tid] += sharedVar[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(covariance, sharedCov[0]);
        atomicAdd(variance, sharedVar[0]);
    }

}


__global__ void element_wise_division(double *r, double *constant_std, double *result, int n)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n) {
        if (*constant_std != 0.0f) {
            result[i] = r[i] / *constant_std;
        } else {
            result[i] = 0.0;
        }
    }
}


__global__ void compute_residual(double *xi, double *xj, double *scaling_factor, double *residual, int size)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
    {
        if (*scaling_factor != 0.0f) {
            residual[i] = xi[i] - *scaling_factor * xj[i];
        }
        else {
            residual[i] = xi[i];
        }
    }
}


__global__ void compute_std(double *A, double *mean, double *std, int size)
{
    extern __shared__ double sharedVar[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx == 0) {
        *std = 0.0f;
    }

    if (tid < blockDim.x)
    {
        sharedVar[tid] = 0.0f;
    }

    __syncthreads();

    if (idx < size)
    {
        double var_component = (A[idx] - *mean) * (A[idx] - *mean);
        sharedVar[tid] = var_component;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedVar[tid] += sharedVar[tid + s];

        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(std, sharedVar[0]);
    }

}


__global__ void compute_log_cosh(double *u, double *log_cosh_sum, int size)
{
    // log(cosh(u))
    extern __shared__ double shared_data[];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (col == 0) {
        *log_cosh_sum = 0.0f;
    }
    __syncthreads();

    if (col < size)
        shared_data[tid] = logf(coshf(u[col]));
    else
        shared_data[tid] = 0.0f;

    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(log_cosh_sum, shared_data[0]);
    }
}


__global__ void compute_u_exp(double *u, double *u_exp_sum, int size)
{
    // u*exp(-u^2 / 2)
    extern __shared__ double shared_data[];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (col == 0) {
        *u_exp_sum = 0.0f;
    }

    if (col < size)
        shared_data[tid] = u[col] * expf(-0.5f * u[col] * u[col]);

    else
        shared_data[tid] = 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(u_exp_sum, shared_data[0]);
    }
}


__global__ void standardize_column(double *A, int m, int n, double *means, double *stds, int startrow)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x + startrow;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n)
    {

        double value = A[row * n + col];
        double mean = means[col];
        double std = stds[col];
        if (std > 0)
            A[row * n + col] = (value - mean) / std;
        else
            A[row * n + col] = 0;
    }
}


__global__ void calculate_statistics(double *A, int m, int n, double *means, double *stds)
{
    // we're not using shared memory here and we're
    //only using global memory layout. and we're doing the division
    //in the kernel
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n)
    {
        double sum = 0.0f;

        for (int i = 0; i < m; ++i)
        {
            double value = A[i * n + col];
            sum += value;
        }

        double mean = sum / m;
        means[col] = mean;
        double sq_sum = 0.0f;

        for (int i = 0; i < m; ++i)
        {
            double value = A[i * n + col];
            sq_sum += (value - mean) * (value - mean);
        }
        stds[col] = sqrtf(sq_sum / m);
    }
}


__global__ void process_column(double *d_X, double *column, int m, int n, int col_idx) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        column[row] = d_X[row * n + col_idx];
    }
}


void divonhost(double *sum, double *d_sum, int m){
    cudaMemcpy(sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    *sum /= m;
    cudaMemcpy(d_sum, sum, sizeof(double), cudaMemcpyHostToDevice);
}


void divsqtonhost(double *sum, double *d_sum, int m){
    cudaMemcpy(sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    *sum /= m;
    *sum = sqrtf(*sum);
    cudaMemcpy(d_sum, sum, sizeof(double), cudaMemcpyHostToDevice);
}


void compute_M(double *d_X, double* xi_std, double* xj_std, double *M_list, int m, int n, double *d_mean_xi, double *d_mean_xj, double *d_residual_ij, double *d_residual_ji, double *d_covariance_i, double *d_variance_i, double *d_covariance_j, double *d_variance_j, double *scaling_factor_ij, double *scaling_factor_ji, double *means_ri, double *means_rj, double *ri_j_std, double *rj_i_std, double *ri_j_div, double *rj_i_div, double *d_log_cosh_sum_1, double *d_u_exp_sum_1, double *d_log_cosh_sum_2, double *d_u_exp_sum_2, double *d_log_cosh_sum_3, double *d_u_exp_sum_3, double *d_log_cosh_sum_4, double *d_u_exp_sum_4) {

    const double k1 = 79.047f;
    const double k2 = 7.4129f;
    const double gamma = 0.37457f;
    double M = 0.0f;


    dim3 threads(256);
    dim3 blocks((m + threads.x - 1) / threads.x);
    size_t sharedMemSize = 4 * threads.x * sizeof(double);

    double log_cosh_mean_1 = 0, u_exp_mean_1 = 0.;
    double log_cosh_mean_2 = 0, u_exp_mean_2 = 0.;
    double log_cosh_mean_3 = 0, u_exp_mean_3 = 0.;
    double log_cosh_mean_4 = 0, u_exp_mean_4 = 0.;

    double *mean_xi,  *mean_xj;

    cudaMallocHost(&mean_xi, sizeof(double));
    cudaMallocHost(&mean_xj, sizeof(double));

    double *h_means_ri,  *h_means_rj;

    cudaMallocHost(&h_means_ri, sizeof(double));
    cudaMallocHost(&h_means_rj, sizeof(double));

    double *h_ri_j_std,  *h_rj_i_std;

    cudaMallocHost(&h_ri_j_std, sizeof(double));
    cudaMallocHost(&h_rj_i_std, sizeof(double));

    const int numStreams = 512;
    cudaStream_t streams[numStreams];

    for (int s = 0; s < numStreams; ++s) {
        cudaStreamCreate(&streams[s]);
    }

    //we're iterating through columns
    for (int i = 0; i < n; ++i) {
        int streamIndex = i % numStreams;
        M = 0.0f;

        for (int j = 0; j < n; ++j) {
            if (i != j) {
                process_column<<<blocks, threads, 0, streams[streamIndex]>>>(d_X, xi_std, m, n, i);
                process_column<<<blocks, threads, 0 ,streams[streamIndex]>>>(d_X, xj_std, m, n, j);
                // gpuErrchk(cudaDeviceSynchronize());

                compute_mean<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(xi_std, d_mean_xi, m);
                compute_mean<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(xj_std, d_mean_xj, m);

                divonhost(mean_xi, d_mean_xi, m);
                divonhost(mean_xj, d_mean_xj, m);

                // gpuErrchk(cudaDeviceSynchronize());

                // covariance and variance are reduced in the kernel
                compute_covariance_variance<<<blocks, threads, 2 * sharedMemSize, streams[streamIndex]>>>(xi_std, xj_std, d_mean_xi, d_mean_xj, d_covariance_i, d_variance_j, m);
                compute_covariance_variance<<<blocks, threads, 2 * sharedMemSize, streams[streamIndex]>>>(xj_std, xi_std, d_mean_xj, d_mean_xi, d_covariance_j, d_variance_i, m);

                // gpuErrchk(cudaDeviceSynchronize());

                element_wise_division<<<blocks, threads, 0, streams[streamIndex]>>>(d_covariance_i, d_variance_j, scaling_factor_ij, 1);
                element_wise_division<<<blocks, threads, 0, streams[streamIndex]>>>(d_covariance_j, d_variance_i, scaling_factor_ji, 1);


                compute_residual<<<blocks, threads, 0, streams[streamIndex]>>>(xi_std, xj_std, scaling_factor_ij, d_residual_ij, m);
                compute_residual<<<blocks, threads, 0, streams[streamIndex]>>>(xj_std, xi_std, scaling_factor_ji, d_residual_ji, m);
                // gpuErrchk(cudaDeviceSynchronize());

                compute_mean<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(d_residual_ij, means_ri, m);
                compute_mean<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(d_residual_ji, means_rj, m);

                divonhost(h_means_ri, means_ri, m);
                divonhost(h_means_rj, means_rj, m);

                // gpuErrchk(cudaDeviceSynchronize());

                compute_std<<<blocks, threads, 2 * sharedMemSize, streams[streamIndex]>>>(d_residual_ij, means_ri, ri_j_std, m);
                compute_std<<<blocks, threads, 2 * sharedMemSize, streams[streamIndex]>>>(d_residual_ji, means_rj, rj_i_std, m);

                divsqtonhost(h_ri_j_std, ri_j_std, m);
                divsqtonhost(h_rj_i_std, rj_i_std, m);

                // gpuErrchk(cudaDeviceSynchronize());

                element_wise_division<<<blocks, threads, 0, streams[streamIndex]>>>(d_residual_ij, ri_j_std, ri_j_div, m);
                element_wise_division<<<blocks, threads, 0, streams[streamIndex]>>>(d_residual_ji, rj_i_std, rj_i_div, m);

                // gpuErrchk(cudaDeviceSynchronize());

                compute_log_cosh<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(xi_std, d_log_cosh_sum_1, m);
                compute_u_exp<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(xi_std, d_u_exp_sum_1, m);

                divonhost(&log_cosh_mean_1, d_log_cosh_sum_1, m);
                divonhost(&u_exp_mean_1, d_u_exp_sum_1, m);

                double entropy = (1.0f + logf(2.0f * M_PI)) / 2.0f - (k1 * (log_cosh_mean_1 - gamma) * (log_cosh_mean_1 - gamma)) - k2 * u_exp_mean_1 * u_exp_mean_1;


                compute_log_cosh<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(xj_std, d_log_cosh_sum_2, m);
                compute_u_exp<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(xj_std, d_u_exp_sum_2, m);

                divonhost(&log_cosh_mean_2, d_log_cosh_sum_2, m);
                divonhost(&u_exp_mean_2, d_u_exp_sum_2, m);

                double entropy2 = (1.0f + logf(2.0f * M_PI)) / 2.0f - (k1 * (log_cosh_mean_2 - gamma) * (log_cosh_mean_2 - gamma)) - k2 * u_exp_mean_2 * u_exp_mean_2;


                compute_log_cosh<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(ri_j_div, d_log_cosh_sum_3, m);
                compute_u_exp<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(ri_j_div, d_u_exp_sum_3, m);

                divonhost(&log_cosh_mean_3, d_log_cosh_sum_3, m);
                divonhost(&u_exp_mean_3, d_u_exp_sum_3, m);

                double entropy3 = (1.0f + logf(2.0f * M_PI)) / 2.0f - (k1 * (log_cosh_mean_3 - gamma) * (log_cosh_mean_3 - gamma)) - k2 * u_exp_mean_3 * u_exp_mean_3;


                compute_log_cosh<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(rj_i_div, d_log_cosh_sum_4, m);
                compute_u_exp<<<blocks, threads, sharedMemSize, streams[streamIndex]>>>(rj_i_div, d_u_exp_sum_4, m);

                divonhost(&log_cosh_mean_4, d_log_cosh_sum_4, m);
                divonhost(&u_exp_mean_4, d_u_exp_sum_4, m);


                double entropy4 = (1.0f + logf(2.0f * M_PI)) / 2.0f - (k1 * (log_cosh_mean_4 - gamma) * (log_cosh_mean_4 - gamma)) - k2 * u_exp_mean_4 * u_exp_mean_4;

                // gpuErrchk(cudaDeviceSynchronize());
                double mi_diff = (entropy2 + entropy3) - (entropy + entropy4);

                if (mi_diff < 0) {
                    M += mi_diff * mi_diff;
                }
            }
        }

        M_list[i] = -1.0f * M;

    }

    for (int s = 0; s < numStreams; ++s) {

        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);

    }

    cudaFreeHost(mean_xi);
    cudaFreeHost(mean_xj);
    cudaFreeHost(h_means_ri);
    cudaFreeHost(h_means_rj);
    cudaFreeHost(h_ri_j_std);
    cudaFreeHost(h_rj_i_std);

}


void update(double *d_X, double *d_residual_ij, int m, int n, int col)
{
    double *h_X;
    double *h_residual_ij;

    cudaMallocHost(&h_X, m * n * sizeof(double));
    cudaMallocHost(&h_residual_ij, m * sizeof(double));

    cudaMemcpy(h_residual_ij, d_residual_ij, m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_X, d_X, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int row = 0; row < m; ++row)
    {
        h_X[row * n + col] = h_residual_ij[row];
    }

    cudaMemcpy(d_X, h_X, m * n * sizeof(double), cudaMemcpyHostToDevice);

    cudaFreeHost(h_X);
    cudaFreeHost(h_residual_ij);
}


extern "C" double* causal_order(double *data, int m, int n)
{
    double *A;
    double *mlist;


    gpuErrchk(cudaMalloc(&A, m * n * sizeof(double)));
    gpuErrchk(cudaMemcpy(A, data, m * n * sizeof(double), cudaMemcpyHostToDevice));

    double *means;
    gpuErrchk(cudaMalloc(&means, n * sizeof(double)));
    double *stds;
    gpuErrchk(cudaMalloc(&stds, n * sizeof(double)));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;

    calculate_statistics<<<blocksPerGrid, threadsPerBlock>>>(A, m, n, means, stds);
    gpuErrchk(cudaDeviceSynchronize());

    int block_i = 16;
    dim3 block(block_i, block_i);
    int numChunks = (m + block_i - 1) / block_i;

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int startRow = chunk *  block_i;
        dim3 grid((block_i + block.x - 1) / block.x,
                         (n + block.y - 1) / block.y);
        standardize_column<<<grid, block>>>(A, m, n, means, stds, startRow);
    }

    gpuErrchk(cudaDeviceSynchronize());

    double *d_mean_xi,  *d_mean_xj;
    double *d_residual_ij,  *d_residual_ji,  *d_covariance_i, *d_variance_i,  *d_covariance_j, *d_variance_j, *scaling_factor_ij, *scaling_factor_ji;

    mlist = (double*)malloc(n * sizeof(double));
    gpuErrchk(cudaMalloc(&d_mean_xi,  sizeof(double)));
    gpuErrchk(cudaMalloc(&d_mean_xj, sizeof(double)));;

    gpuErrchk(cudaMalloc(&d_residual_ij, m * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_residual_ji, m * sizeof(double)));

    gpuErrchk(cudaMalloc(&d_covariance_i, sizeof(double)));
    gpuErrchk(cudaMalloc(&d_variance_i, sizeof(double)));
    gpuErrchk(cudaMalloc(&d_covariance_j, sizeof(double)));
    gpuErrchk(cudaMalloc(&d_variance_j, sizeof(double)));

    gpuErrchk(cudaMalloc(&scaling_factor_ij, sizeof(double)));
    gpuErrchk(cudaMalloc(&scaling_factor_ji, sizeof(double)));

    cudaMemset(d_mean_xi, 0, sizeof(double));
    cudaMemset(d_mean_xj, 0, sizeof(double));
    cudaMemset(d_covariance_i, 0, sizeof(double));
    cudaMemset(d_variance_i, 0, sizeof(double));
    cudaMemset(d_covariance_j, 0, sizeof(double));
    cudaMemset(d_variance_j, 0, sizeof(double));


    double *means_ri, *means_rj,  *ri_j_std,  *rj_i_std,  *ri_j_div, *rj_i_div;
    double *d_log_cosh_sum_1, *d_u_exp_sum_1, *d_log_cosh_sum_2, *d_u_exp_sum_2, *d_log_cosh_sum_3, *d_u_exp_sum_3, *d_log_cosh_sum_4, *d_u_exp_sum_4;

    gpuErrchk(cudaMalloc(&means_ri, sizeof(double)));
    gpuErrchk(cudaMalloc(&means_rj, sizeof(double)));
    gpuErrchk(cudaMalloc(&ri_j_std, sizeof(double)));
    gpuErrchk(cudaMalloc(&rj_i_std, sizeof(double)));
    gpuErrchk(cudaMalloc(&ri_j_div, m * sizeof(double)));
    gpuErrchk(cudaMalloc(&rj_i_div, m * sizeof(double)));
    cudaMalloc(&d_log_cosh_sum_1, sizeof(double));
    cudaMalloc(&d_u_exp_sum_1, sizeof(double));
    cudaMalloc(&d_log_cosh_sum_2, sizeof(double));
    cudaMalloc(&d_u_exp_sum_2, sizeof(double));
    cudaMalloc(&d_log_cosh_sum_3, sizeof(double));
    cudaMalloc(&d_u_exp_sum_3, sizeof(double));
    cudaMalloc(&d_log_cosh_sum_4, sizeof(double));
    cudaMalloc(&d_u_exp_sum_4, sizeof(double));

    double *xi_std, *xj_std;
    cudaMalloc(&xi_std, m * sizeof(double));
    cudaMalloc(&xj_std, m * sizeof(double));


    compute_M(A, xi_std, xj_std, mlist, m, n, d_mean_xi, d_mean_xj, d_residual_ij,  d_residual_ji,  d_covariance_i, d_variance_j, d_covariance_j, d_variance_j, scaling_factor_ij, scaling_factor_ji, means_ri, means_rj,  ri_j_std,  rj_i_std,  ri_j_div, rj_i_div,  d_log_cosh_sum_1, d_u_exp_sum_1, d_log_cosh_sum_2, d_u_exp_sum_2, d_log_cosh_sum_3, d_u_exp_sum_3, d_log_cosh_sum_4, d_u_exp_sum_4);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(means));
    gpuErrchk(cudaFree(stds));

    gpuErrchk(cudaFree(d_mean_xi));
    gpuErrchk(cudaFree(d_mean_xj));
    gpuErrchk(cudaFree(d_residual_ij));

    gpuErrchk(cudaFree(d_residual_ji));
    gpuErrchk(cudaFree(d_covariance_i));
    gpuErrchk(cudaFree(d_variance_i));

    gpuErrchk(cudaFree(d_covariance_j));
    gpuErrchk(cudaFree(d_variance_j));
    gpuErrchk(cudaFree(scaling_factor_ij));

    gpuErrchk(cudaFree(scaling_factor_ji));
    gpuErrchk(cudaFree(means_ri));
    gpuErrchk(cudaFree(means_rj));

    gpuErrchk(cudaFree(ri_j_std));
    gpuErrchk(cudaFree(rj_i_std));
    gpuErrchk(cudaFree(ri_j_div));

    gpuErrchk(cudaFree(rj_i_div));
    gpuErrchk(cudaFree(d_log_cosh_sum_1));
    gpuErrchk(cudaFree(d_u_exp_sum_1));
    gpuErrchk(cudaFree(d_log_cosh_sum_2));
    gpuErrchk(cudaFree(d_u_exp_sum_2));
    gpuErrchk(cudaFree(d_log_cosh_sum_3));
    gpuErrchk(cudaFree(d_u_exp_sum_3));
    gpuErrchk(cudaFree(d_log_cosh_sum_4));
    gpuErrchk(cudaFree(d_u_exp_sum_4));

    gpuErrchk(cudaFree(xi_std));
    gpuErrchk(cudaFree(xj_std));

    cudaFree(A);

    // std::cout << "mlist:" << mlist[0];

    return mlist;
}
