#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <algorithm>
#include <bits/stdc++.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include <iostream>

// Used for Global Memory Approach
#define BLOCK_DIM 8
#define THREADS_PER_BLOCK 64
// Default to support at least 1000 observations with factor 0.2
#ifndef MAX_K_RUNGE
    #define MAX_K_RUNGE 20001
#endif
#define THREAD_BLOCK_SIZE 32
#define THREAD_BLOCK_SIZE_SHUFFLE 32
#define BIG_FLOAT 3.402823466e38 // Taken from: https://stackoverflow.com/questions/8812422/how-to-find-epsilon-min-and-max-constants-for-cuda

#define SHUFFLE_NEIGHBORS 15

// TODO: check, if there are better way to calculate digamma
// Taken from: http://web.science.mq.edu.au/~mjohnson/code/digamma.c
/**
 * Evaluate digamma function for a given x
 *
 * @param x        value to evauluate digamma on
 */
__host__ __device__ float digamma(float x) {
    float result = 0, xx, xx2, xx4;
    for ( ; x < 7; ++x)
      result -= 1/x;
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += logf(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
    return result;
  }

__global__ void
pipeline_mi_calculation_l0_runge(float* obs_dev, int* block_starts, int* tail_indices, float* mi_values, size_t obs_count, size_t dim, int k, int n_blks, size_t sig_blocklength, size_t has_tail, size_t tail_length){
    extern __shared__ float shared_mem[];
    __shared__ int tail_index;
    __shared__ int X_a_start_index;
    __shared__ int Y_a_start_index;
    __shared__ int X_b_start_index;
    __shared__ int Y_b_start_index;
    float k_distances[MAX_K_RUNGE];

    // Define the required ID's
    size_t t_x = threadIdx.x;
    size_t b_y = blockIdx.y;
    int perm_id = blockIdx.x;
    int obs_id_a = min((THREAD_BLOCK_SIZE * b_y) + t_x, obs_count -1);

    if(t_x == 0){
        tail_index = tail_indices[perm_id];
        X_a_start_index = 0;
        Y_a_start_index = THREAD_BLOCK_SIZE;
        X_b_start_index = 2 * THREAD_BLOCK_SIZE;
        Y_b_start_index = 3 * THREAD_BLOCK_SIZE;
    }
    __syncthreads();

    // Calculate Permuted X ID
    if(obs_id_a >= obs_count){
        obs_id_a = obs_count - 1;
    }
    int from_index_X_a = -1;
    if(obs_id_a < tail_index){
        from_index_X_a = block_starts[(perm_id * n_blks) + (obs_id_a / sig_blocklength)] + obs_id_a - ((obs_id_a / sig_blocklength) * sig_blocklength);
    }else{
        if(obs_id_a - tail_index < tail_length){
            from_index_X_a = (n_blks * sig_blocklength) + obs_id_a - ((obs_id_a / sig_blocklength) * sig_blocklength);
        }else{
            from_index_X_a = block_starts[(perm_id * n_blks) + ((obs_id_a - tail_length) / sig_blocklength)] + (obs_id_a - tail_length) - (((obs_id_a - tail_length) / sig_blocklength) * sig_blocklength);
        }
    }

    // Load and Init Fixed Data into Shared Memory
    shared_mem[X_a_start_index + t_x] = obs_dev[from_index_X_a];
    shared_mem[Y_a_start_index + t_x] = obs_dev[obs_count + obs_id_a];

    for(int i = 0; i < k; i++){
        k_distances[i] = BIG_FLOAT;
    }
    __syncthreads();

    // Calculate kth distance
    for(int b = 0; b < obs_count; b+=THREAD_BLOCK_SIZE){
        // Load Current Observation Portion
        int obs_id_b = min(b + t_x, obs_count -1);
        int from_index_X_b = -1;
        if(obs_id_b < tail_index){
            from_index_X_b = block_starts[(perm_id * n_blks) + (obs_id_b / sig_blocklength)] + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
        }else{
            if(obs_id_b - tail_index < tail_length){
                from_index_X_b = (n_blks * sig_blocklength) + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
            }else{
                from_index_X_b = block_starts[(perm_id * n_blks) + ((obs_id_b - tail_length) / sig_blocklength)] + (obs_id_b - tail_length) - (((obs_id_b - tail_length) / sig_blocklength) * sig_blocklength);
            }
        }
        shared_mem[X_b_start_index + t_x] = obs_dev[from_index_X_b];
        shared_mem[Y_b_start_index + t_x] = obs_dev[obs_count + obs_id_b];
        __syncthreads();
        // Calculate Distance for Observation Portion and Insert them at the correct Position
        for(int t = 0; t < THREAD_BLOCK_SIZE; t++){
            if(b + t > obs_count - 1){
                continue;
            }

            float tmp_chebyshev = max(abs(shared_mem[X_a_start_index + t_x] - shared_mem[X_b_start_index + t]), abs(shared_mem[Y_a_start_index + t_x] - shared_mem[Y_b_start_index + t]));

            // Find out where to insert current distance
            int k_index = k;
            while(k_index > 0 && tmp_chebyshev < k_distances[k_index - 1]){
                k_index--;
            }

            // Insert
            for(int i = k - 2; i >= k_index; i--){
                k_distances[i + 1] = k_distances[i];
            }
            if(k_index < k){
                k_distances[k_index] = tmp_chebyshev;
            }
        }
    }

    float kth_dist = k_distances[k-1];

    // Count nx, ny
    int nx_counter = 0;
    int ny_counter = 0;
    for(int b = 0; b < obs_count; b+=THREAD_BLOCK_SIZE){
        // Load Current Observation Portion
        int obs_id_b = min(b + t_x, obs_count -1);
        int from_index_X_b = -1;
        if(obs_id_b < tail_index){
            from_index_X_b = block_starts[(perm_id * n_blks) + (obs_id_b / sig_blocklength)] + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
        }else{
            if(obs_id_b - tail_index < tail_length){
                from_index_X_b = (n_blks * sig_blocklength) + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
            }else{
                from_index_X_b = block_starts[(perm_id * n_blks) + ((obs_id_b - tail_length) / sig_blocklength)] + (obs_id_b - tail_length) - (((obs_id_b - tail_length) / sig_blocklength) * sig_blocklength);
            }
        }
        shared_mem[X_b_start_index + t_x] = obs_dev[from_index_X_b];
        shared_mem[Y_b_start_index + t_x] = obs_dev[obs_count + obs_id_b];
        __syncthreads();

        // Calculate Distance to Observation Portion and count corresponding nx, ny
        for(int t = 0; t < THREAD_BLOCK_SIZE; t++){
            if(b + t > obs_count - 1){
                continue;
            }
            float tmp_chebyshev_x = abs(shared_mem[X_a_start_index + t_x] - shared_mem[X_b_start_index + t]);
            nx_counter += (tmp_chebyshev_x <= kth_dist);

            float tmp_chebyshev_y = abs(shared_mem[Y_a_start_index + t_x] - shared_mem[Y_b_start_index + t]);
            ny_counter += (tmp_chebyshev_y <= kth_dist);
        }
    }
    float partial_mi_value = digamma(nx_counter) + digamma(ny_counter);

    if((THREAD_BLOCK_SIZE * b_y) + t_x < obs_count){
        atomicAdd(mi_values+perm_id, partial_mi_value);
    }
}

__global__ void
pipeline_mi_calculation_l0_row_runge(float* obs_dev, int* block_starts, int* tail_indices, float* mi_values, size_t obs_count, int k, size_t n_blks, size_t sig_blocklength, int has_tail, size_t tail_length, int x_id, int* y_candidates){
    extern __shared__ float shared_mem[];
    __shared__ int tail_index;
    __shared__ int X_a_start_index;
    __shared__ int Y_a_start_index;
    __shared__ int X_b_start_index;
    __shared__ int Y_b_start_index;
    float k_distances[MAX_K_RUNGE];

    // Define the required ID's
    size_t t_x = threadIdx.x;
    // y_pos for correct position in mi_values, block_starts, tail_indices
    int y_pos = blockIdx.x;
    // y_id for correct access to observational data
    int y_id = y_candidates[y_pos];
    size_t b_y = blockIdx.z;
    int perm_id = blockIdx.y;
    int perm_dim = gridDim.y;
    int obs_id_a = min((THREAD_BLOCK_SIZE * b_y) + t_x, obs_count -1);

    if(t_x == 0){
        tail_index = tail_indices[(y_pos * perm_dim) + perm_id];
        X_a_start_index = 0;
        Y_a_start_index = THREAD_BLOCK_SIZE;
        X_b_start_index = 2 * THREAD_BLOCK_SIZE;
        Y_b_start_index = 3 * THREAD_BLOCK_SIZE;
    }
    __syncthreads();

    // Calculate Permuted X ID
    if(obs_id_a >= obs_count){
        obs_id_a = obs_count - 1;
    }
    int from_index_X_a = -1;
    if(obs_id_a < tail_index){
        from_index_X_a = block_starts[(y_pos * perm_dim * n_blks) + (perm_id * n_blks) + (obs_id_a / sig_blocklength)] + obs_id_a - ((obs_id_a / sig_blocklength) * sig_blocklength);
    }else{
        if(obs_id_a - tail_index < tail_length){
            from_index_X_a = (n_blks * sig_blocklength) + obs_id_a - ((obs_id_a / sig_blocklength) * sig_blocklength);
        }else{
            from_index_X_a = block_starts[(y_pos * perm_dim * n_blks) + (perm_id * n_blks) + ((obs_id_a - tail_length) / sig_blocklength)] + (obs_id_a - tail_length) - (((obs_id_a - tail_length) / sig_blocklength) * sig_blocklength);
        }
    }

    // Load and Init Fixed Data into Shared Memory
    shared_mem[X_a_start_index + t_x] = obs_dev[(x_id * obs_count) + from_index_X_a];
    shared_mem[Y_a_start_index + t_x] = obs_dev[(y_id * obs_count) + obs_id_a];

    for(int i = 0; i < k; i++){
        k_distances[i] = BIG_FLOAT;
    }
    __syncthreads();

    // Calculate kth distance
    for(size_t b = 0; b < obs_count; b+=THREAD_BLOCK_SIZE){
        // Load Current Observation Portion
        int obs_id_b = min(b + t_x, obs_count -1);
        int from_index_X_b = -1;
        if(obs_id_b < tail_index){
            from_index_X_b = block_starts[(y_pos * perm_dim * n_blks) + (perm_id * n_blks) + (obs_id_b / sig_blocklength)] + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
        }else{
            if(obs_id_b - tail_index < tail_length){
                from_index_X_b = (n_blks * sig_blocklength) + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
            }else{
                from_index_X_b = block_starts[(y_pos * perm_dim * n_blks) + (perm_id * n_blks) + ((obs_id_b - tail_length) / sig_blocklength)] + (obs_id_b - tail_length) - (((obs_id_b - tail_length) / sig_blocklength) * sig_blocklength);
            }
        }
        shared_mem[X_b_start_index + t_x] = obs_dev[(x_id * obs_count) + from_index_X_b];
        shared_mem[Y_b_start_index + t_x] = obs_dev[(y_id * obs_count) + obs_id_b];
        __syncthreads();
        // Calculate Distance for Observation Portion and Insert them at the correct Position
        for(int t = 0; t < THREAD_BLOCK_SIZE; t++){
            if(b + t > obs_count - 1){
                continue;
            }

            float tmp_chebyshev = max(abs(shared_mem[X_a_start_index + t_x] - shared_mem[X_b_start_index + t]), abs(shared_mem[Y_a_start_index + t_x] - shared_mem[Y_b_start_index + t]));

            // Find out where to insert current distance
            int k_index = k;
            while(k_index > 0 && tmp_chebyshev < k_distances[k_index - 1]){
                k_index--;
            }

            // Insert
            for(int i = k - 2; i >= k_index; i--){
                k_distances[i + 1] = k_distances[i];
            }
            if(k_index < k){
                k_distances[k_index] = tmp_chebyshev;
            }
        }
    }

    float kth_dist = k_distances[k-1];

    // Count nx, ny
    int nx_counter = 0;
    int ny_counter = 0;
    for(size_t b = 0; b < obs_count; b+=THREAD_BLOCK_SIZE){
        // Load Current Observation Portion
        int obs_id_b = min(b + t_x, obs_count -1);
        int from_index_X_b = -1;
        if(obs_id_b < tail_index){
            from_index_X_b = block_starts[(y_pos * perm_dim * n_blks) + (perm_id * n_blks) + (obs_id_b / sig_blocklength)] + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
        }else{
            if(obs_id_b - tail_index < tail_length){
                from_index_X_b = (n_blks * sig_blocklength) + obs_id_b - ((obs_id_b / sig_blocklength) * sig_blocklength);
            }else{
                from_index_X_b = block_starts[(y_pos * perm_dim * n_blks) + (perm_id * n_blks) + ((obs_id_b - tail_length) / sig_blocklength)] + (obs_id_b - tail_length) - (((obs_id_b - tail_length) / sig_blocklength) * sig_blocklength);
            }
        }
        shared_mem[X_b_start_index + t_x] = obs_dev[(x_id * obs_count) + from_index_X_b];
        shared_mem[Y_b_start_index + t_x] = obs_dev[(y_id * obs_count) + obs_id_b];
        __syncthreads();

        // Calculate Distance to Observation Portion and count corresponding nx, ny
        for(int t = 0; t < THREAD_BLOCK_SIZE; t++){
            if(b + t > obs_count - 1){
                continue;
            }
            float tmp_chebyshev_x = abs(shared_mem[X_a_start_index + t_x] - shared_mem[X_b_start_index + t]);
            nx_counter += (tmp_chebyshev_x <= kth_dist);

            float tmp_chebyshev_y = abs(shared_mem[Y_a_start_index + t_x] - shared_mem[Y_b_start_index + t]);
            ny_counter += (tmp_chebyshev_y <= kth_dist);
        }
    }
    float partial_mi_value =  digamma(nx_counter) + digamma(ny_counter);

    if((THREAD_BLOCK_SIZE * b_y) + t_x < obs_count){
        atomicAdd(mi_values + ((y_pos * perm_dim) +perm_id), partial_mi_value);
    }
}

float pval_l0_cuda_shared(const float*  obs,
                          size_t        obs_count,
                          size_t        dim,
                          int           k,
                          size_t        permutations)
{
    // Constants
    const int BLOCKS_PER_PERMUTATION = (obs_count + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int = sizeof(int);
    cudaSetDevice(0);

    // Load Data to GPU
    float* obs_dev = NULL;
    cudaMalloc(&obs_dev, obs_count * dim * size_of_float);
    cudaMemcpy(obs_dev, obs, obs_count * dim * size_of_float, cudaMemcpyHostToDevice);
    
    float* mi_values =  NULL;
    cudaMallocManaged(&mi_values, (permutations+1) * size_of_float);
    for(int i = 0; i < (permutations + 1); i++){
        mi_values[i] = log(obs_count);
    }

    // Prepare Block Permutations
    int* block_starts = NULL;
    size_t sig_blocklength = max((size_t)1, obs_count / 20);
    size_t n_blks = obs_count / sig_blocklength;
    cudaMallocManaged(&block_starts, (permutations + 1) * n_blks * size_of_int);
    for(int i = 0; i < permutations + 1; i++){
        for(int j = 0; j < n_blks; j++){
            block_starts[(i*n_blks)+j] = (sig_blocklength * j);
        }
    }

    int* tail_indices = NULL;
    cudaMallocManaged(&tail_indices, (permutations + 1) * size_of_int);
    size_t tail_length = obs_count % n_blks;
    int has_tail = tail_length != 0;
    srand(time(NULL));

    for(int i = 0; i < permutations + 1; i++){
        if(i > 0){
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(block_starts+(i*n_blks), block_starts+((i+1) * n_blks), g);
            if(has_tail){
                tail_indices[i] = block_starts[rand() % n_blks];
            }else{
                tail_indices[i] = obs_count;
            }
        }
        else{
            if(has_tail){
                tail_indices[0] = sig_blocklength * n_blks;
            }
            else{
                tail_indices[i] = obs_count;
            }
        }
    }

    // Pipelined MI Calculation
    dim3 grid0(permutations + 1, BLOCKS_PER_PERMUTATION, 1);
    dim3 block0(THREAD_BLOCK_SIZE, 1, 1);
    pipeline_mi_calculation_l0_runge<<<grid0, block0, (4 * THREAD_BLOCK_SIZE) * size_of_float>>>(obs_dev, block_starts, tail_indices, mi_values, obs_count, dim, k, n_blks, sig_blocklength, has_tail, tail_length);   
    cudaDeviceSynchronize();

    // Calculate PValue
    float div = obs_count * 1.0;
    float base_mi = digamma(k) + digamma(obs_count) - mi_values[0]/div;
    float mi = 0.0;
    float pvalue = 0.0;
    for(int i = 1; i < permutations+1; i++){
        mi = digamma(k) + digamma(obs_count) - (mi_values[i]/div);
        if(mi >= base_mi){
            pvalue += 1.0;
        }
    }
    pvalue = pvalue / permutations;

    cudaFree(obs_dev);
    cudaFree(mi_values);
    cudaFree(block_starts);
    cudaFree(tail_indices);

    return pvalue;
}

__global__ void
pipeline_mi_calculation_ln_runge(float* obs_yz_dev, float* x_permutations_dev, float* mi_values, size_t obs_count, size_t dim, int k, int k_perm){
    extern __shared__ float shared_mem[];
    float xyz_k_distances[MAX_K_RUNGE];

    // Define the required ID's
    size_t t_x = threadIdx.x;
    size_t b_y = blockIdx.y;
    int perm_id = blockIdx.x;
    int obs_id_a = min((THREAD_BLOCK_SIZE * b_y) + t_x, obs_count -1);

    // Calculate Permuted X ID
    if(obs_id_a >= obs_count){
        obs_id_a = obs_count - 1;
    }

    // Load and Init Fixed Data into Shared Memory
    shared_mem[t_x] = x_permutations_dev[(obs_count * perm_id) + obs_id_a];
    for(int d = 1; d < dim; d++){
        shared_mem[(THREAD_BLOCK_SIZE * d) + t_x] = obs_yz_dev[(obs_count * (d-1)) + obs_id_a];
    }
    for(int i = 0; i < k; i++){
        xyz_k_distances[i] = BIG_FLOAT;
    }

    __syncthreads();

    // Calculate kth distances
    for(int obs_id_b = 0; obs_id_b < obs_count; obs_id_b++){
        // z distance
        float chebyshev = 0.0;
        for(int d = 2; d < dim; d++){
            chebyshev = max(chebyshev, abs(shared_mem[(THREAD_BLOCK_SIZE * d) + t_x] - obs_yz_dev[((d - 1) * obs_count) + obs_id_b]));
        }
        // x distance
        chebyshev = max(abs(shared_mem[t_x] - x_permutations_dev[(perm_id * obs_count) + obs_id_b]), chebyshev);
        // y distance
        chebyshev = max(abs(shared_mem[THREAD_BLOCK_SIZE + t_x] - obs_yz_dev[obs_id_b]), chebyshev);

        // Insert xyz distances
        int k_index = k;
        while(k_index > 0 && chebyshev < xyz_k_distances[k_index - 1]){
            k_index--;
        }

        for(int i = k - 2; i >= k_index; i--){
            xyz_k_distances[i + 1] = xyz_k_distances[i];
        }
        if(k_index < k){
            xyz_k_distances[k_index] = chebyshev;
        }
    }
    float xyz_kth_dist = xyz_k_distances[k - 1];

    // Count nxz, nyz, nz
    int nxz_counter = 0;
    int nyz_counter = 0;
    int nz_counter = 0;
    for(int obs_id_b = 0; obs_id_b < obs_count; obs_id_b++){
        // Calculate z distance
        float chebyshev = 0.0;
        for(int d = 2; d < dim; d++){
            chebyshev = max(chebyshev, abs(shared_mem[(THREAD_BLOCK_SIZE * d) + t_x] - obs_yz_dev[((d - 1) * obs_count) + obs_id_b]));
        }
        if (chebyshev <= xyz_kth_dist) {
            nz_counter += 1;
            nxz_counter += (abs(shared_mem[t_x] - x_permutations_dev[(perm_id * obs_count) + obs_id_b]) <= xyz_kth_dist);
            nyz_counter += (abs(shared_mem[THREAD_BLOCK_SIZE + t_x] - obs_yz_dev[obs_id_b]) <= xyz_kth_dist);
        }
    }

    float partial_mi_value = (digamma(nxz_counter) + digamma(nyz_counter) - digamma(nz_counter));

    if((THREAD_BLOCK_SIZE * b_y) + t_x < obs_count){
        atomicAdd(mi_values+perm_id, partial_mi_value);
    }
}

__global__ void
pipeline_mi_calculation_ln_row_runge(float* obs_dev, float* x_permutations_dev, float* mi_values, size_t obs_count, int dim, int k, int k_perm, int x_id, size_t permutations, int* sList_dev, size_t sEntries, size_t varCount, int* yCandidates, size_t yDim, int* yPosition, int* oldSEntryPointer, int* newSEntryPointer){
    extern __shared__ float shared_mem[];
    float xyz_k_distances[MAX_K_RUNGE];

    // Define the required ID's
    int t_x = threadIdx.x;
    // y limited to one adjacent to x
    int yPos = yPosition[blockIdx.x];
    int y = yCandidates[yPos];
    /// y == x no mi needed
    if (y == x_id) {
        return;
    }
    // select from new data structure
    int entryInS = oldSEntryPointer[yPos];
    // in case of splitted separation sets, we could exceed the entryInS due to skipping when y is part of S
    if (entryInS >= sEntries) {
        return;
    }
    // y in separation set --> no mi needed --> use next entry
    bool valid = false;
    while (!valid) {
        for (int s = 0; s < (dim - 2); ++s) {
            if (y == sList_dev[entryInS * (dim -2) + s]) {
                entryInS += 1;
                valid = false;
                break;
            } else {
                valid = true;
            }
        }
    }
    if (t_x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        newSEntryPointer[yPos] = entryInS + 1;
    }
    // in case of splitted separation sets, we could exceed the entryInS due to skipping when y is part of S
    if (entryInS >= sEntries) {
        return;
    }
    int perm_id = blockIdx.y;
    int bpp = blockIdx.z;
    int obs_id_a = (THREAD_BLOCK_SIZE * bpp) + t_x;
    if (obs_id_a < obs_count) {
        // load X or Perm of X
        if (perm_id == 0) {
            shared_mem[t_x] = obs_dev[(obs_count * x_id) + obs_id_a];
        } else {
            shared_mem[t_x] = x_permutations_dev[entryInS * permutations * obs_count + (obs_count * (perm_id - 1)) + obs_id_a];
        }
        // load Y
        shared_mem[THREAD_BLOCK_SIZE + t_x] = obs_dev[(obs_count * y) + obs_id_a];
        // load all Z
        for(int d = 2; d < dim; d++){
            shared_mem[(THREAD_BLOCK_SIZE * d) + t_x] = obs_dev[(obs_count * sList_dev[entryInS * (dim - 2) + d - 2]) + obs_id_a];
        }

        for(int i = 0; i < k; i++){
            xyz_k_distances[i] = BIG_FLOAT;
        }
    }
    __syncthreads();

    int obs_id_block_start = 0;
    // Calculate kth distances
    for (int obs_id_b = t_x; obs_id_b < obs_count + (blockDim.x - (obs_count % blockDim.x)); obs_id_b +=blockDim.x) {
        // Block process observations & load obs_b to shared_mem
        if (obs_id_b < obs_count) {
            if (perm_id == 0) {
                shared_mem[(blockDim.x * dim) + t_x] = obs_dev[(obs_count * x_id) + obs_id_b];
            } else {
                shared_mem[(blockDim.x * dim) + t_x] = x_permutations_dev[entryInS * permutations * obs_count + (obs_count * (perm_id - 1)) + obs_id_b];
            }
            // load Y
            shared_mem[(blockDim.x * dim) + blockDim.x + t_x] = obs_dev[(obs_count * y) + obs_id_b];
            // load all Z
            for(int d = 2; d < dim; d++){
                shared_mem[(blockDim.x * dim) + (blockDim.x * d) + t_x] = obs_dev[(obs_count * sList_dev[entryInS * (dim - 2) + d - 2]) + obs_id_b];
            }
        }
        __syncthreads();
        if (obs_id_a < obs_count) {
            // Process individual observations
            for (int cur_obs_b = 0; cur_obs_b < blockDim.x; ++cur_obs_b) {
                if (obs_id_block_start + cur_obs_b < obs_count) {
                    // Calc z distance
                    float chebyshev = 0.0;
                    for (int d = 2; d < dim; d++){
                        chebyshev = max(chebyshev, abs(shared_mem[(blockDim.x * d) + t_x] - shared_mem[(blockDim.x * dim) + (blockDim.x * d) + cur_obs_b]));
                    }
                    // Calc x distance
                    chebyshev = max(abs(shared_mem[t_x] - shared_mem[(blockDim.x * dim) + cur_obs_b]), chebyshev);
                    // Calc y distance
                    chebyshev = max(abs(shared_mem[blockDim.x + t_x] - shared_mem[(blockDim.x * dim) + blockDim.x + cur_obs_b]), chebyshev);

                    // Insert xyz distances
                    int k_index = k;
                    while(k_index > 0 && chebyshev < xyz_k_distances[k_index - 1]){
                        k_index--;
                    }

                    for(int i = k - 2; i >= k_index; i--){
                        xyz_k_distances[i + 1] = xyz_k_distances[i];
                    }
                    if(k_index < k){
                        xyz_k_distances[k_index] = chebyshev;
                    }
                }
            }
        }
        obs_id_block_start += blockDim.x;
    }
    float xyz_kth_dist = xyz_k_distances[k - 1];
    // Count nxz, nyz, nz
    int nxz_counter = 0;
    int nyz_counter = 0;
    int nz_counter = 0;
    obs_id_block_start = 0;
    for (int obs_id_b = t_x; obs_id_b < obs_count + (blockDim.x - (obs_count % blockDim.x)); obs_id_b +=blockDim.x) {
        // Block process observations & load obs_b to shared_mem
        if (obs_id_b < obs_count) {
            if (perm_id == 0) {
                shared_mem[(blockDim.x * dim) + t_x] = obs_dev[(obs_count * x_id) + obs_id_b];
            } else {
                shared_mem[(blockDim.x * dim) + t_x] = x_permutations_dev[entryInS * permutations * obs_count + (obs_count * (perm_id - 1)) + obs_id_b];
            }
            // load Y
            shared_mem[(blockDim.x * dim) + blockDim.x + t_x] = obs_dev[(obs_count * y) + obs_id_b];
            // load all Z
            for(int d = 2; d < dim; d++){
                shared_mem[(blockDim.x * dim) + (blockDim.x * d) + t_x] = obs_dev[(obs_count * sList_dev[entryInS * (dim - 2) + d - 2]) + obs_id_b];
            }
        }
        __syncthreads();
        if (obs_id_a < obs_count) {
            // Process individual observations
            for (int cur_obs_b = 0; cur_obs_b < blockDim.x; ++cur_obs_b) {
                if (obs_id_block_start + cur_obs_b < obs_count) {
                    // Calculate z distance
                    float chebyshev = 0.0;
                    for(int d = 2; d < dim; d++){
                        chebyshev = max(chebyshev, abs(shared_mem[(blockDim.x * d) + t_x] - shared_mem[(blockDim.x * dim) + (blockDim.x * d) + cur_obs_b]));
                    }
                    if (chebyshev <= xyz_kth_dist) {
                        nz_counter += 1;
                        nxz_counter += (abs(shared_mem[t_x] - shared_mem[(blockDim.x * dim) + cur_obs_b]) <= xyz_kth_dist);
                        nyz_counter += (abs(shared_mem[blockDim.x + t_x] - shared_mem[(blockDim.x * dim) + blockDim.x + cur_obs_b]) <= xyz_kth_dist);
                    }
                }
            }
        }
        obs_id_block_start += blockDim.x;
    }
    if(obs_id_a < obs_count) {
        float partial_mi_value = (digamma(nxz_counter) + digamma(nyz_counter) - digamma(nz_counter));
        atomicAdd(&mi_values[yPos * (permutations + 1) + perm_id], partial_mi_value);
    }
}

__global__ void
pipeline_rperm_ln_multi(float* obs_z, int* perm_x, int* used, size_t obs_count, size_t dim, size_t permutations){
    // random order going through X (could this be achieved through scheduling of thread blocks ?)
    // neighbors for each X_i of size count == 5 / 10 --> depends on variable shuffle_neighbors (reuse neighbor calc)
    // calculate shuffled neighbors
    // access use index through atomiccas auf position in neighbors (yes we go through it ordered within the local list, but do not guarantee that each thread is executed in order --> different used)
    // NOTE WE ALLOW FOR DIVERGENCE OF UP TO SHUFFLE NEIGHBORS
    extern __shared__ float shared_mem[];
    float z_k_distances[SHUFFLE_NEIGHBORS];
    int z_k_pos[SHUFFLE_NEIGHBORS];

    // Define the required ID's
    size_t t_x = threadIdx.x;
    size_t b_y = blockIdx.y;
    size_t pos = min((THREAD_BLOCK_SIZE_SHUFFLE * b_y) + t_x, obs_count -1);

    // Init shared and local data structures
    for(int d = 0; d < dim; ++d){
        shared_mem[(THREAD_BLOCK_SIZE_SHUFFLE * d) + t_x] = obs_z[(obs_count * d) + pos];
    }
    for(int i = 0; i < SHUFFLE_NEIGHBORS; i++){
        z_k_distances[i] = BIG_FLOAT;
    }
    __syncthreads();
    // Calculate the SHUFFLE_NEIGHBORS nearest neighbors of z
    for(int obs_id = 0; obs_id < obs_count; obs_id++){
        // Skip position of current z'
        if (obs_id == pos) {
            continue;
        }
        // Calculate distances of current z' to all z
        float z_chebyshev = 0.0;
        for(int d = 0; d < dim; d++){
            z_chebyshev = max(z_chebyshev, abs(shared_mem[(THREAD_BLOCK_SIZE_SHUFFLE * d) + t_x] - obs_z[(obs_count * d) + obs_id]));
        }

        // Insert z distances
        int k_index = SHUFFLE_NEIGHBORS;
        while(k_index > 0 && z_chebyshev < z_k_distances[k_index - 1]){
            k_index--;
        }

        for(int i = SHUFFLE_NEIGHBORS - 2; i >= k_index; i--){
            z_k_distances[i + 1] = z_k_distances[i];
            z_k_pos[i + 1] = z_k_pos[i];
        }
        if(k_index < SHUFFLE_NEIGHBORS){
            z_k_distances[k_index] = z_chebyshev;
            z_k_pos[k_index] = obs_id;
        }
    }
    /// ITERATE THROUGH ALL PERMUTATIONS, no need to recalculate neighbors

    curandStatePhilox4_32_10_t state;
    curand_init(0, pos, 0, &state);
    for(int b_x = 0; b_x < permutations; ++b_x) {
        // SHUFFLE NEIGHBORS
        for(int i = SHUFFLE_NEIGHBORS-1; i > 0; i--) {
            int j = curand(&state) % (i+1) ;
            float t = z_k_pos[i];
            z_k_pos[i] = z_k_pos[j];
            z_k_pos[j] = t;
        }
        int count = 0;
        while ( atomicCAS(&used[b_x * obs_count + z_k_pos[count]],0, 1) != 0  && count < SHUFFLE_NEIGHBORS - 1) {
            ++count;
        }
        perm_x[b_x * obs_count + pos] = z_k_pos[count];
    }
}

__global__ void
pipeline_rperm_ln_multi_all(float* obs, float* perm_x, int* used, size_t obs_count, size_t dim, size_t permutations, int* sList_dev, size_t sEntries, int x_id){
    // random order going through X (could this be achieved through scheduling of thread blocks ?)
    // neighbors for each X_i of size count == 5 / 10 --> depends on variable shuffle_neighbors (reuse neighbor calc)
    // calculate shuffled neighbors
    // access use index through atomiccas auf position in neighbors (yes we go through it ordered within the local list, but do not guarantee that each thread is executed in order --> different used)
    // NOTE WE ALLOW FOR DIVERGENCE OF UP TO SHUFFLE NEIGHBORS
    extern __shared__ float shared_mem[];
    float z_k_distances[SHUFFLE_NEIGHBORS];
    int z_k_pos[SHUFFLE_NEIGHBORS];
    // TODO currently fixed value of size dimension!
    float my_point[10];

    // Define the required ID's
    size_t t_x = threadIdx.x;
    size_t b_y = blockIdx.y;
    int b_x = blockIdx.x; // sList_dev Pos
    int pos = min((THREAD_BLOCK_SIZE_SHUFFLE * b_y) + t_x, obs_count -1);

    // Init shared and local data structures
    for(int d = 0; d < dim; ++d){
        my_point[d] = obs[(obs_count * sList_dev[b_x * dim + d]) + pos];
    }
    for(int i = 0; i < SHUFFLE_NEIGHBORS; i++){
        z_k_distances[i] = BIG_FLOAT;
    }
    __syncthreads();
    // Calculate the SHUFFLE_NEIGHBORS nearest neighbors of z
    for(int obs_block = 0; obs_block < (obs_count + THREAD_BLOCK_SIZE_SHUFFLE - 1) / THREAD_BLOCK_SIZE_SHUFFLE; ++obs_block) {
        /// USE SHARED MEM TO LOAD 32 VALUES of obs and than each thread proceesses
        if (obs_block * THREAD_BLOCK_SIZE_SHUFFLE + t_x < obs_count) {
            for (int d = 0; d < dim; ++d){
                shared_mem[(THREAD_BLOCK_SIZE_SHUFFLE * d) + t_x] = obs[(obs_count * sList_dev[b_x * dim + d]) + obs_block * THREAD_BLOCK_SIZE_SHUFFLE + t_x];
            }
        }
        __syncthreads();
        for (int block_pos = 0; block_pos < THREAD_BLOCK_SIZE_SHUFFLE; ++block_pos) {
            int obs_id = obs_block * THREAD_BLOCK_SIZE_SHUFFLE + block_pos;
            // Skip position of current z'
            if (obs_id == pos) {
                continue;
            }
            if (obs_id == obs_count) {
                break;
            }
            // Calculate distances of current z' to all z
            float z_chebyshev = 0.0;
            for(int d = 0; d < dim; d++){
                z_chebyshev = max(z_chebyshev, abs(my_point[d] - shared_mem[(THREAD_BLOCK_SIZE_SHUFFLE * d) + block_pos]));
            }

            // Insert z distances
            int k_index = SHUFFLE_NEIGHBORS;
            while(k_index > 0 && z_chebyshev < z_k_distances[k_index - 1]){
                k_index--;
            }

            for(int i = SHUFFLE_NEIGHBORS - 2; i >= k_index; i--){
                z_k_distances[i + 1] = z_k_distances[i];
                z_k_pos[i + 1] = z_k_pos[i];
            }
            if(k_index < SHUFFLE_NEIGHBORS){
                z_k_distances[k_index] = z_chebyshev;
                z_k_pos[k_index] = obs_id;
            }
        }            

    }
    /// ITERATE THROUGH ALL PERMUTATIONS, no need to recalculate neighbors
    curandStatePhilox4_32_10_t state;
    curand_init(0, pos, 0, &state);
    for(int p = 0; p < permutations; ++p) {
        // SHUFFLE NEIGHBORS
        for(int i = SHUFFLE_NEIGHBORS-1; i > 0; i--) {
            int j = curand(&state) % (i+1) ;
            float t = z_k_pos[i];
            z_k_pos[i] = z_k_pos[j];
            z_k_pos[j] = t;
        }
        int count = 0;
        while ( atomicCAS(&used[p * obs_count * sEntries + b_x * obs_count + z_k_pos[count]],0, 1) != 0  && count < SHUFFLE_NEIGHBORS - 1) {
            ++count;
        }
        perm_x[b_x * permutations * obs_count + p * obs_count + pos] = obs[x_id * obs_count + z_k_pos[count]];
        
    }
}

int binom(int n, int k) {
    if (n < k || k < 0)
    return 0;
    if (k == 0 || k == n)
    return 1;
    return binom(n - 1, k - 1) + binom(n - 1, k);
}

void perm_cuda_multi(const float* obs, size_t obs_count, size_t dim, size_t permutations, int* x_permutations){
    cudaProfilerStart();
    const int BLOCKS_PER_PERMUTATION = (obs_count + THREAD_BLOCK_SIZE_SHUFFLE - 1) / THREAD_BLOCK_SIZE_SHUFFLE;
    const unsigned int size_of_float = sizeof(float);
    cudaSetDevice(0);
    // Load Data to GPU
    float* obs_z_dev = NULL;
    int* perm_x_dev = NULL;
    int* used = NULL;
    cudaMalloc(&obs_z_dev, obs_count * dim * size_of_float);
    cudaMemcpy(obs_z_dev, obs, obs_count * dim * size_of_float, cudaMemcpyHostToDevice);
    cudaMalloc(&perm_x_dev, obs_count * permutations * sizeof(int));
    cudaMalloc(&used, obs_count * permutations * sizeof(int));
    cudaMemset(used, 0, obs_count * permutations * sizeof(int));
    cudaMemset(perm_x_dev, 0, obs_count * permutations * sizeof(int));
    cudaDeviceSynchronize();

    // Pipelined Perm Calculation
    dim3 grid0(1, BLOCKS_PER_PERMUTATION, 1);
    dim3 block0(THREAD_BLOCK_SIZE_SHUFFLE, 1, 1);
    pipeline_rperm_ln_multi<<<grid0, block0, THREAD_BLOCK_SIZE_SHUFFLE * dim * size_of_float>>>(obs_z_dev, perm_x_dev, used, obs_count, dim, permutations);
    cudaDeviceSynchronize();
    cudaMemcpy(x_permutations, perm_x_dev, obs_count * permutations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Free Resources
    cudaFree(obs_z_dev);
    cudaFree(perm_x_dev);
    cudaFree(used);
    cudaProfilerStop();
}

void perm_cuda_multi_all(const float* obs, size_t obs_count, size_t vars, size_t permutations, float* x_permutations,
                        int* sList_c, size_t dim, size_t sEntries, int x_id)
{

    cudaProfilerStart();
    const int BLOCKS_PER_PERMUTATION = (obs_count + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    const unsigned int size_of_float = sizeof(float);

    float* obs_dev = NULL;
    float* perm_dev = NULL;
    int* sList_dev = NULL;
    int* used = NULL;
    cudaMalloc(&obs_dev, obs_count * vars * size_of_float);
    cudaMemcpy(obs_dev, obs, obs_count * vars * size_of_float, cudaMemcpyHostToDevice);
    cudaMalloc(&perm_dev, obs_count * permutations * sEntries * size_of_float);
    cudaMalloc(&used, obs_count * permutations * sEntries * sizeof(int));
    cudaMalloc(&sList_dev, dim * sEntries * sizeof(int));
    cudaMemset(used, 0, obs_count * permutations * sEntries * sizeof(int));
    cudaMemset(perm_dev, 0, obs_count * permutations * sEntries * size_of_float);
    cudaMemcpy(sList_dev, sList_c, dim * sEntries * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Pipelined Perm Calculation
    dim3 grid0(sEntries, BLOCKS_PER_PERMUTATION, 1);
    dim3 block0(THREAD_BLOCK_SIZE_SHUFFLE, 1, 1);
    pipeline_rperm_ln_multi_all<<<grid0, block0, THREAD_BLOCK_SIZE_SHUFFLE * dim * size_of_float>>>(obs_dev,
        perm_dev, used, obs_count, dim, permutations, sList_dev, sEntries, x_id);
    cudaDeviceSynchronize();
    cudaMemcpy(x_permutations, perm_dev, obs_count * permutations * sEntries * size_of_float, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // Free Resources
    cudaFree(obs_dev);
    cudaFree(perm_dev);
    cudaFree(used);
    cudaFree(sList_dev);
    cudaProfilerStop();
}

void pval_l0_row_cuda(const float* data_c,
                      size_t obs_count,
                      int k,
                      size_t permutations,
                      int x_id,
                      size_t varCount,
                      float* pvalOfX,
                      int* candidates_c,
                      size_t yDim)
{
    cudaProfilerStart();
    const int BLOCKS_PER_PERMUTATION = (obs_count + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int = sizeof(int);
    cudaSetDevice(0);

    // Load observations to GPU
    float* obs_dev = NULL;
    cudaMalloc(&obs_dev, obs_count * varCount * size_of_float);
    cudaMemcpy(obs_dev, data_c, obs_count * varCount * size_of_float, cudaMemcpyHostToDevice);

    // Load tmpval structure for MI values to GPU
    float* mi_values =  NULL;
    cudaMallocManaged(&mi_values, (permutations+1) * yDim * size_of_float);
    memset(mi_values, 0, (permutations+1) * yDim * size_of_float);

    // Load neighboring y values to GPU
    int* y_candidates_dev = NULL;
    cudaMalloc(&y_candidates_dev, yDim * sizeof(int));
    cudaMemcpy(y_candidates_dev, candidates_c, yDim * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Prepare Block Permutations
    int* block_starts = NULL;
    size_t sig_blocklength = max((size_t)1, obs_count / 20);
    size_t n_blks = obs_count / sig_blocklength;
    cudaMallocManaged(&block_starts, yDim * (permutations + 1) * n_blks * size_of_int);
    for (size_t y = 0; y < yDim; y++) {
        for(size_t i = 0; i < permutations + 1; i++){
            for(size_t j = 0; j < n_blks; j++){
                block_starts[(y * n_blks * (permutations + 1)) + (i * n_blks) + j] = (sig_blocklength * j);
            }
        }
    }

    int* tail_indices = NULL;
    cudaMallocManaged(&tail_indices, yDim * (permutations + 1) * size_of_int);
    size_t tail_length = obs_count % n_blks;
    int has_tail = tail_length != 0;
    srand(time(NULL));
    for (size_t y = 0; y < yDim; ++y) {
        for(size_t i = 0; i < permutations + 1; i++){
            if(i > 0){
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(block_starts+((y * n_blks * (permutations + 1)) + (i*n_blks)),
                             block_starts+((y * n_blks * (permutations + 1)) + ((i+1) * n_blks)), g);
                if(has_tail){
                    tail_indices[(y * (permutations + 1)) + i] = block_starts[(y * n_blks * (permutations + 1)) +rand() % n_blks];
                }else{
                    tail_indices[(y * (permutations + 1)) + i] = obs_count;
                }
            }
            else{
                if(has_tail){
                    tail_indices[(y * (permutations + 1))] = sig_blocklength * n_blks;
                }
                else{
                    tail_indices[(y * (permutations + 1)) + i] = obs_count;
                }
            }
        }
    }
    float pvalue, base_mi, mi, div;
    // Pipelined MI Calculation
    dim3 grid0(yDim, permutations + 1, BLOCKS_PER_PERMUTATION);
    dim3 block0(THREAD_BLOCK_SIZE, 1, 1);
    pipeline_mi_calculation_l0_row_runge<<<grid0, block0, (4 * THREAD_BLOCK_SIZE) * size_of_float>>>(obs_dev, block_starts, tail_indices, mi_values, obs_count, k, n_blks, sig_blocklength, has_tail, tail_length, x_id, y_candidates_dev);
    div = obs_count * 1.0;
    cudaDeviceSynchronize();
    for (size_t j = 0; j < yDim; ++j) {
        pvalue = 0.0;
        base_mi = digamma(k) + digamma(obs_count) - mi_values[j * (permutations+1)]/div;
        for(size_t i = 1; i < permutations+1; i++){
            mi = digamma(k) + digamma(obs_count) - (mi_values[j * (permutations+1) + i]/div);
            if(mi >= base_mi){
                pvalue += 1.0;
            }
        }
        pvalOfX[j] = pvalue / (permutations * 1.0);
    }

    // Free Resources
    cudaFree(obs_dev);
    cudaFree(mi_values);
    cudaFree(y_candidates_dev);
    cudaFree(block_starts);
    cudaFree(tail_indices);
    cudaProfilerStop();
}

// candidates == neighboring y, s_list == combination of y
// optimized for early termination
void pval_ln_row_cuda(const float* obs, const float* x_permutations_c, size_t obs_count, int k, int k_perm, size_t permutations, int x_id, size_t varCount, int lvl, int* sList_c, size_t sEntries, int* sOfX, float* pvalOfX, float alpha, int* candidates_c, size_t* yDim, size_t originalYDim, bool splitted) {

    cudaProfilerStart();
    const int BLOCKS_PER_PERMUTATION = (obs_count + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    const unsigned int size_of_float = sizeof(float);
    cudaSetDevice(0);

    // Load observations to GPU
    float* obs_dev = NULL;
    cudaMalloc(&obs_dev, obs_count * varCount * size_of_float);
    cudaMemcpy(obs_dev, obs, obs_count * varCount * size_of_float, cudaMemcpyHostToDevice);

    // Load tmval structure for MI values to GPU
    float* mi_values =  NULL;
    cudaMallocManaged(&mi_values, (permutations+1) * *yDim * size_of_float);

    // Load permutations p for X with all Z to GPU 
    float* x_permutations_dev = NULL;
    cudaMalloc(&x_permutations_dev, obs_count * permutations * sEntries * size_of_float);
    cudaMemcpy(x_permutations_dev, x_permutations_c, obs_count * permutations * sEntries * size_of_float, cudaMemcpyHostToDevice);

    // Load combinations Z to GPU
    int* sList_dev = NULL;
    cudaMalloc(&sList_dev, lvl * sEntries * sizeof(int));
    cudaMemcpy(sList_dev, sList_c, lvl * sEntries * sizeof(int), cudaMemcpyHostToDevice);

    // Load neighboring y values to GPU
    int* y_candidates_dev = NULL;
    cudaMalloc(&y_candidates_dev, *yDim * sizeof(int));
    cudaMemcpy(y_candidates_dev, candidates_c, *yDim * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int* yPosition = NULL;
    cudaMallocManaged(&yPosition, *yDim * sizeof(int));
    for(int i = 0; i < *yDim; ++i) {
        yPosition[i] = i;
    }
    int* oldSEntryPointer = NULL;
    cudaMallocManaged(&oldSEntryPointer, *yDim * sizeof(int));
    memset(oldSEntryPointer, 0, *yDim * sizeof(int));
    int* newSEntryPointer = NULL;
    cudaMallocManaged(&newSEntryPointer, *yDim * sizeof(int));
    memset(newSEntryPointer, 0, *yDim * sizeof(int));
    size_t remainingY = *yDim;
    size_t counter = (splitted) ? sEntries : binom(originalYDim-1, lvl);
    float div = obs_count * 1.0;
    while (counter > 0 && remainingY > 0) {
        memset(mi_values, 0, (permutations+1) * *yDim * size_of_float);
        dim3 grid0(remainingY, permutations + 1, BLOCKS_PER_PERMUTATION);
        dim3 block0(THREAD_BLOCK_SIZE, 1, 1);
        pipeline_mi_calculation_ln_row_runge<<<grid0, block0, 2 * THREAD_BLOCK_SIZE * (lvl + 2) * size_of_float>>>(obs_dev, x_permutations_dev, mi_values, obs_count, (lvl + 2), k, k_perm, x_id, permutations, sList_dev, sEntries, varCount, y_candidates_dev, *yDim, yPosition, oldSEntryPointer, newSEntryPointer);
        cudaDeviceSynchronize();
        float pvalue, base_mi, cmi;
        int yPos = 0;
        for (int j = 0; j < remainingY; ++j) {
            // skip entries that have accessed all sep sets, either because element was skipped in previous, but not last or exactly in the last run
            if(oldSEntryPointer[yPosition[j]] == sEntries || newSEntryPointer[yPosition[j]] > sEntries) {
                yPosition[yPos] = yPosition[j];
                ++yPos;
                continue;
            }
            bool remains = true;
            if(candidates_c[yPosition[j]] == x_id) {
                yPosition[yPos] = yPosition[j];
                ++yPos;
                continue;
            }
            pvalue = 0.0;
            base_mi = digamma(k) - (mi_values[yPosition[j] * (permutations+1)]/div);
            for(int i = 1; i < permutations+1; i++){
                cmi = digamma(k) - (mi_values[yPosition[j] * (permutations+1) + i]/div);
                if(cmi >= base_mi){
                    pvalue += 1.0;
                } 
            }
            pvalue = pvalue / (permutations * 1.0);
            if (pvalue > pvalOfX[candidates_c[yPosition[j]]]) {
                pvalOfX[candidates_c[yPosition[j]]] = pvalue;
            }
            if (pvalue > alpha) {
                for (int i = 0; i < lvl; ++i) {
                    sOfX[candidates_c[yPosition[j]] * lvl + i] = sList_c[(newSEntryPointer[yPosition[j]] - 1) * lvl + i];
                }
                remains = false;
            }
            oldSEntryPointer[yPosition[j]] = newSEntryPointer[yPosition[j]];
            if (remains) {
                yPosition[yPos] = yPosition[j];
                ++yPos;
            }
        }
        remainingY = yPos;
        --counter;
    }
    // prepare return for remaining y for next sepset chunk
    *yDim = remainingY;
    for (int i = 0; i < remainingY; ++i) {
        candidates_c[i] = candidates_c[yPosition[i]];
    }
    // Free Resources
    cudaFree(obs_dev);
    cudaFree(mi_values);
    cudaFree(x_permutations_dev);
    cudaFree(sList_dev);
    cudaFree(yPosition);
    cudaFree(y_candidates_dev);
    cudaFree(oldSEntryPointer);
    cudaFree(newSEntryPointer);
    cudaProfilerStop();
}

float pval_ln_cuda(const float* obs, const float* x_permutations, size_t obs_count, size_t dim, int k, int k_perm, size_t permutations){
    // Constants
    const int BLOCKS_PER_PERMUTATION = (obs_count + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    const unsigned int size_of_float = sizeof(float);
    cudaSetDevice(0);

    // Load Data to GPU
    float* obs_yz_dev = NULL;
    cudaMalloc(&obs_yz_dev, obs_count * (dim - 1) * size_of_float);
    cudaMemcpy(obs_yz_dev, obs + obs_count, obs_count * (dim - 1) * size_of_float, cudaMemcpyHostToDevice);

    float* mi_values =  NULL;
    cudaMallocManaged(&mi_values, (permutations+1) * size_of_float);
    for(int i = 0; i < (permutations + 1); i++){
        mi_values[i] = 0;
    }

    float* x_permutations_dev = NULL;
    cudaMalloc(&x_permutations_dev, obs_count * (permutations + 1) * size_of_float);
    cudaMemcpy(x_permutations_dev, obs, obs_count * size_of_float, cudaMemcpyHostToDevice);
    cudaMemcpy(x_permutations_dev + obs_count, x_permutations, obs_count * permutations * size_of_float, cudaMemcpyHostToDevice);

    float pvalue = 0.0;
    // Pipelined MI Calculation
    dim3 grid0(permutations + 1, BLOCKS_PER_PERMUTATION, 1);
    dim3 block0(THREAD_BLOCK_SIZE, 1, 1);
    pipeline_mi_calculation_ln_runge<<<grid0, block0, THREAD_BLOCK_SIZE * dim * size_of_float>>>(obs_yz_dev, x_permutations_dev, mi_values, obs_count, dim, k, k_perm);
    cudaDeviceSynchronize();
    float base_mi = digamma(k) - mi_values[0]/obs_count;
    for(int i = 1; i < permutations+1; i++){
        if((digamma(k) - mi_values[i]/obs_count) >= base_mi){
            pvalue += 1.0;
        }
    }
    cudaDeviceSynchronize();
    // Calculate PValue
    pvalue = pvalue / permutations;

    // Free Resources
    cudaFree(obs_yz_dev);
    cudaFree(mi_values);
    cudaFree(x_permutations_dev);
    return pvalue;
}

size_t call_init_gpu_cuda() {
    cudaSetDevice(0);
    size_t free = 0;
    size_t total = 0;
    cudaFree(0);
    cudaMemGetInfo(&free, &total);
    return free;
}
