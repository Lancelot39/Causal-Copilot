#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="algorithm/tests"

# Set number of parallel jobs
N_JOBS=2

# Define GPU devices to use in rotation
GPU_DEVICES=(2 7) # Modify this array with your available GPU IDs

# Check if GPU_DEVICES array is empty and provide a default if needed
if [ ${#GPU_DEVICES[@]} -eq 0 ]; then
    echo "Warning: No GPU devices specified. Using GPU 0 as default."
    GPU_DEVICES=(0)
fi

# Define CPU allocation per job
CPUS_PER_JOB=8 # Number of CPU cores to allocate per job

# Get list of algorithms from config
ALGORITHMS=(PC DirectLiNGAM) # GOLEM FCI CDNOD FGES XGES GRaSP GES 
#             NOTEARSLinear BAMB IAMBnPC MBOR InterIAMB # HITONMB
#             #Hybrid CALM NOTEARSNonlinear CORL
# )

# Define algorithm hyperparameter combinations for linear and nonlinear settings
# Format: "algorithm_name:hyperparameter_json"
ALGORITHM_HYPERPARAMS=(
    # # PC algorithm - nonlinear variants
    # "PC:{\"indep_test\":\"rcit_cpu\"}"
    # "PC:{\"indep_test\":\"fastkci_cpu\"}"
    # "PC:{\"indep_test\":\"kci_cpu\"}"
    
    # # # # FCI algorithm - nonlinear variants
    # "FCI:{\"indep_test\":\"rcit\"}"
    # "FCI:{\"indep_test\":\"fastkci\"}"
    # "FCI:{\"indep_test\":\"kci\"}"
    
    # # # # CDNOD algorithm - nonlinear variants
    # "CDNOD:{\"indep_test\":\"rcit\"}"
    # # "CDNOD:{\"indep_test\":\"fastkci\"}"
    # # "CDNOD:{\"indep_test\":\"kci_cpu\"}"
    
    # # # # DirectLiNGAM - kernel-based nonlinear variant
    # "DirectLiNGAM:{\"measure\":\"kernel\",\"gpu\":false}"
    
    # # # Markov blanket algorithms - nonlinear independence tests
    # "BAMB:{\"indep_test\":\"rcit\"}"
    # "BAMB:{\"indep_test\":\"fastkci\"}"
    # "BAMB:{\"indep_test\":\"kci\"}"
    
    # "IAMBnPC:{\"indep_test\":\"rcit\"}"
    # "IAMBnPC:{\"indep_test\":\"fastkci\"}"
    # "IAMBnPC:{\"indep_test\":\"kci\"}"
    
    # "MBOR:{\"indep_test\":\"rcit\"}"
    # "MBOR:{\"indep_test\":\"fastkci\"}"
    # "MBOR:{\"indep_test\":\"kci\"}"
    
    # "InterIAMB:{\"indep_test\":\"rcit\"}"
    # "InterIAMB:{\"indep_test\":\"fastkci\"}"
    # "InterIAMB:{\"indep_test\":\"kci\"}"

    # # Linear methods first
    
    # # # GRaSP algorithm - linear score-based
    # "GRaSP:{\"score_func\":\"local_score_BIC\"}"

    # # # GES algorithm - linear score-based
    # "GES:{\"score_func\":\"local_score_BIC\"}"
    
    # # # PC algorithm - linear variant
    # "PC:{\"indep_test\":\"fisherz_cpu\"}"
    "PC:{\"indep_test\":\"fisherz_gpu\"}"
    "PC:{\"indep_test\":\"cmiknn_gpu\"}"
    
    # # # FCI algorithm - linear variant
    # "FCI:{\"indep_test\":\"fisherz\"}"
    
    # # # # CDNOD algorithm - linear variant
    "CDNOD:{\"indep_test\":\"fisherz_gpu\"}"
    "CDNOD:{\"indep_test\":\"cmiknn_gpu\"}"
    # "CDNOD:{\"indep_test\":\"fisherz_cpu\"}"
    
    # # # # # FGES algorithm
    # "FGES:{\"sparsity\":10}"
    # "FGES:{\"sparsity\":5}"
    # "FGES:{\"sparsity\":2}"
    # "FGES:{\"sparsity\":1}"
    
    # # # # # XGES algorithm
    # "XGES:{\"alpha\":0.5}"
    # "XGES:{\"alpha\":1}"
    # "XGES:{\"alpha\":2}"
    # "XGES:{\"alpha\":4}"
    
    # # # DirectLiNGAM algorithm - linear non-Gaussian method
    # "DirectLiNGAM:{\"measure\":\"pwling\",\"gpu\":false}"
    # "DirectLiNGAM:{\"measure\":\"pwling\",\"gpu\":true}"
    # "DirectLiNGAM:{\"measure\":\"kernel\",\"gpu\":true}"
    
    # # NOTEARSLinear algorithm - gradient-based optimization
    # "NOTEARSLinear:{\"lambda1\":0.01,\"loss_type\":\"l2\"}"
    
    # # # Markov blanket algorithms - linear independence tests
    # "BAMB:{\"indep_test\":\"fisherz\"}"
    # "IAMBnPC:{\"indep_test\":\"fisherz\"}"
    # "MBOR:{\"indep_test\":\"fisherz\"}"
    # "InterIAMB:{\"indep_test\":\"fisherz\"}"

    # # # # # GOLEM algorithm
    # "GOLEM:{}"
    
    # Nonlinear methods

    # GES algorithm - nonlinear score
    # "GES:{\"score_func\":\"local_score_marginal_general\"}"
    
    # GRaSP algorithm - nonlinear score
    # "GRaSP:{\"score_func\":\"local_score_marginal_general\"}"
)

# Create a temporary directory for parallel job logs
LOG_DIR="$SCRIPT_DIR/benchmark_logs_v3/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"

# Create a file to track GPU assignment across processes
GPU_COUNTER_FILE="${LOG_DIR}/gpu_counter"
echo "0" > "$GPU_COUNTER_FILE"

# Create a file to track CPU assignment across processes
CPU_COUNTER_FILE="${LOG_DIR}/cpu_counter"
echo "0" > "$CPU_COUNTER_FILE"

# Function to get the next GPU device in rotation using a file-based counter
get_next_gpu() {
    local gpu_devices=("$@")
    # Use flock to ensure atomic read-modify-write operation
    (
        flock -x 200
        gpu_counter=$(cat "$GPU_COUNTER_FILE")
        # Safety check to ensure gpu_devices array is not empty
        if [ ${#gpu_devices[@]} -eq 0 ]; then
            echo "0"  # Default to GPU 0 if array is empty
            return
        fi
        gpu=${gpu_devices[$gpu_counter]}
        # Increment counter and wrap around
        gpu_counter=$(( (gpu_counter + 1) % ${#gpu_devices[@]} ))
        echo "$gpu_counter" > "$GPU_COUNTER_FILE"
        echo "$gpu"
    ) 200>"${LOG_DIR}/gpu_lock"
}

# Function to get the next set of CPUs to use
get_next_cpus() {
    # Use flock to ensure atomic read-modify-write operation
    (
        flock -x 201
        cpu_counter=$(cat "$CPU_COUNTER_FILE")
        # Calculate CPU range for this job
        start_cpu=$((cpu_counter * CPUS_PER_JOB))
        end_cpu=$(((cpu_counter + 1) * CPUS_PER_JOB - 1))
        cpu_range="${start_cpu}-${end_cpu}"
        
        # Increment counter for next job
        total_cpu_groups=$(nproc --all)
        total_cpu_groups=$((total_cpu_groups / CPUS_PER_JOB))
        # Ensure we don't divide by zero
        if [ $total_cpu_groups -eq 0 ]; then
            total_cpu_groups=1
        fi
        cpu_counter=$(( (cpu_counter + 1) % total_cpu_groups ))
        echo "$cpu_counter" > "$CPU_COUNTER_FILE"
        
        echo "$cpu_range"
    ) 201>"${LOG_DIR}/cpu_lock"
}

# Function to run benchmark for a single algorithm with logging
run_benchmark() {
    local algo="$1"
    local params_json="$2"
    local gpu_devices_str="$3"
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/${algo}_${timestamp}.log"
    
    # Convert the GPU devices string back to an array
    IFS=' ' read -r -a gpu_devices <<< "$gpu_devices_str"
    
    # Get the next GPU device in rotation
    local gpu_device=$(get_next_gpu "${gpu_devices[@]}")
    
    # Get the next CPU range to use
    local cpu_range=$(get_next_cpus)
    
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting benchmark for algorithm: $algo with params: $params_json on GPU $gpu_device with CPUs $cpu_range" | tee -a "$log_file"
    
    # Run the benchmark with params if provided
    if [ -n "$params_json" ]; then
        # Create a temporary file to store the JSON parameters
        local param_file="${LOG_DIR}/${algo}_params_${params_json}_${timestamp}.json"
        echo "$params_json" > "$param_file"
        
        # Run with the param file
        CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_range python "$SCRIPT_DIR/benchmarking.py" --algorithm "$algo" --param_file "$param_file" >> "$log_file" 2>&1
        
        # Clean up
        rm -f "$param_file"
    else
        CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_range python "$SCRIPT_DIR/benchmarking.py" --algorithm "$algo" >> "$log_file" 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Benchmark for algorithm $algo completed successfully." | tee -a "$log_file"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Benchmark for algorithm $algo encountered an error." | tee -a "$log_file"
    fi
}

# Convert GPU_DEVICES array to space-separated string for passing to parallel
GPU_DEVICES_STR=$(printf "%s " "${GPU_DEVICES[@]}")

# Export functions for parallel use
export -f run_benchmark
export -f get_next_gpu
export -f get_next_cpus
export GPU_COUNTER_FILE
export CPU_COUNTER_FILE
export LOG_DIR
export SCRIPT_DIR
export N_JOBS
export CPUS_PER_JOB

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting all benchmarks..."
echo "CPU allocation: $CPUS_PER_JOB cores per job"

# Check if we should run with hyperparameters
if [ "$1" == "--with-params" ]; then
    echo "Running benchmarks with specified hyperparameters..."
    
    # Check if sequential flag is provided
    if [ "$2" == "--sequential" ]; then
        echo "Running hyperparameter benchmarks sequentially..."
        for combo in "${ALGORITHM_HYPERPARAMS[@]}"; do
            # Extract algorithm and parameters safely
            algo="${combo%%:*}"
            params="${combo#*:}"
            run_benchmark "$algo" "$params" "$GPU_DEVICES_STR"
        done
    else
        echo "Running hyperparameter benchmarks in parallel..."
        # Create job list file for parallel
        JOB_LIST_FILE="${LOG_DIR}/job_list.txt"
        
        # Write job configurations to the job list file
        for combo in "${ALGORITHM_HYPERPARAMS[@]}"; do
            # Extract algorithm and parameters safely
            algo="${combo%%:*}"
            params="${combo#*:}"
            echo "$algo::$params::$GPU_DEVICES_STR" >> "$JOB_LIST_FILE"
        done
        
        # Use a custom delimiter that won't appear in the params
        cat "$JOB_LIST_FILE" | parallel --jobs $N_JOBS --colsep '::' 'run_benchmark {1} {2} {3}'
        
        # Clean up
        rm -f "$JOB_LIST_FILE"
    fi
    
# Otherwise run standard benchmarks
else
    # Check if sequential flag is provided
    if [ "$1" == "--sequential" ]; then
        echo "Running benchmarks sequentially..."
        for algo in "${ALGORITHMS[@]}"; do
            run_benchmark "$algo" "" "$GPU_DEVICES_STR"
        done
    else
        echo "Running benchmarks in parallel..."
        # Use GNU parallel to run jobs in parallel
        printf '%s\n' "${ALGORITHMS[@]}" | \
        parallel --jobs $N_JOBS \
            'run_benchmark {} "" "'"$GPU_DEVICES_STR"'"'
    fi
fi

echo "[$(date +"%Y-%m-%d %H:%M:%S")] All benchmarks completed. Logs available in $LOG_DIR/"