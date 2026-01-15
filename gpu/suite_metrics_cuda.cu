// CUDA kernels for suite-level metrics computation
// Stage 3: GPU acceleration for confusion matrix and promotion gating

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Compute confusion matrix elements in parallel
// Inputs:
//   test_events: (n_test_events,) events with idx and match
//   promoted_indices: (n_promoted,) sorted array of promoted indices
// Output:
//   confusion_counts: (4,) array [tp, fp, tn, fn]
__global__ void compute_confusion_matrix_kernel(
    const int32_t* test_indices,
    const bool* test_matches,
    const int32_t* promoted_indices,
    int32_t n_test_events,
    int32_t n_promoted,
    int32_t* confusion_counts  // [tp, fp, tn, fn]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test_events) return;
    
    int32_t test_idx = test_indices[idx];
    bool truth = test_matches[idx];
    
    // Binary search to check if test_idx is in promoted_indices
    bool pred = false;
    int left = 0;
    int right = n_promoted - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (promoted_indices[mid] == test_idx) {
            pred = true;
            break;
        } else if (promoted_indices[mid] < test_idx) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    // Atomic updates to confusion matrix
    if (pred && truth) {
        atomicAdd(&confusion_counts[0], 1);  // TP
    } else if (pred && !truth) {
        atomicAdd(&confusion_counts[1], 1);  // FP
    } else if (!pred && !truth) {
        atomicAdd(&confusion_counts[2], 1);  // TN
    } else {  // !pred && truth
        atomicAdd(&confusion_counts[3], 1);  // FN
    }
}

// Aggregate training events by index (parallel reduction)
// Inputs:
//   train_indices: (n_train_events,) index IDs
//   train_matches: (n_train_events,) match flags
// Output:
//   index_totals: (max_index + 1,) total counts per index
//   index_matches: (max_index + 1,) match counts per index
__global__ void aggregate_train_events_kernel(
    const int32_t* train_indices,
    const bool* train_matches,
    int32_t n_train_events,
    int32_t* index_totals,
    int32_t* index_matches,
    int32_t max_index
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_train_events) return;
    
    int32_t index_id = train_indices[idx];
    if (index_id >= 0 && index_id <= max_index) {
        atomicAdd(&index_totals[index_id], 1);
        if (train_matches[idx]) {
            atomicAdd(&index_matches[index_id], 1);
        }
    }
}

// Launch functions
extern "C" {
    void compute_confusion_matrix_cuda_launch(
        const int32_t* test_indices,
        const bool* test_matches,
        const int32_t* promoted_indices,
        int32_t n_test_events,
        int32_t n_promoted,
        int32_t* confusion_counts  // Host memory, will be filled
    ) {
        // Allocate device memory
        int32_t* d_confusion_counts;
        int32_t* d_test_indices;
        bool* d_test_matches;
        int32_t* d_promoted_indices;
        
        cudaMalloc(&d_confusion_counts, 4 * sizeof(int32_t));
        cudaMalloc(&d_test_indices, n_test_events * sizeof(int32_t));
        cudaMalloc(&d_test_matches, n_test_events * sizeof(bool));
        cudaMalloc(&d_promoted_indices, n_promoted * sizeof(int32_t));
        
        // Initialize confusion counts to zero
        cudaMemset(d_confusion_counts, 0, 4 * sizeof(int32_t));
        
        // Copy data to device
        cudaMemcpy(d_test_indices, test_indices, n_test_events * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_test_matches, test_matches, n_test_events * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_promoted_indices, promoted_indices, n_promoted * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((n_test_events + block.x - 1) / block.x);
        compute_confusion_matrix_kernel<<<grid, block>>>(
            d_test_indices, d_test_matches, d_promoted_indices,
            n_test_events, n_promoted, d_confusion_counts
        );
        
        // Copy results back
        cudaMemcpy(confusion_counts, d_confusion_counts, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_confusion_counts);
        cudaFree(d_test_indices);
        cudaFree(d_test_matches);
        cudaFree(d_promoted_indices);
    }
    
    void aggregate_train_events_cuda_launch(
        const int32_t* train_indices,
        const bool* train_matches,
        int32_t n_train_events,
        int32_t* index_totals,
        int32_t* index_matches,
        int32_t max_index
    ) {
        // Allocate device memory
        int32_t* d_index_totals;
        int32_t* d_index_matches;
        int32_t* d_train_indices;
        bool* d_train_matches;
        
        cudaMalloc(&d_index_totals, (max_index + 1) * sizeof(int32_t));
        cudaMalloc(&d_index_matches, (max_index + 1) * sizeof(int32_t));
        cudaMalloc(&d_train_indices, n_train_events * sizeof(int32_t));
        cudaMalloc(&d_train_matches, n_train_events * sizeof(bool));
        
        // Initialize to zero
        cudaMemset(d_index_totals, 0, (max_index + 1) * sizeof(int32_t));
        cudaMemset(d_index_matches, 0, (max_index + 1) * sizeof(int32_t));
        
        // Copy data to device
        cudaMemcpy(d_train_indices, train_indices, n_train_events * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_train_matches, train_matches, n_train_events * sizeof(bool), cudaMemcpyHostToDevice);
        
        // Launch kernel
        dim3 block(256);
        dim3 grid((n_train_events + block.x - 1) / block.x);
        aggregate_train_events_kernel<<<grid, block>>>(
            d_train_indices, d_train_matches, n_train_events,
            d_index_totals, d_index_matches, max_index
        );
        
        // Copy results back
        cudaMemcpy(index_totals, d_index_totals, (max_index + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(index_matches, d_index_matches, (max_index + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_index_totals);
        cudaFree(d_index_matches);
        cudaFree(d_train_indices);
        cudaFree(d_train_matches);
    }
}
