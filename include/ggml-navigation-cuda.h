#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernels for Navigation paradigm operations

// Dual quaternion inner product: ⟨q, k⟩dq = Q ⊗ K*
void ggml_navigation_dual_quat_inner_product_cuda_launch(
    const float* q,           // [n, 8] query dual quaternions
    const float* k,           // [n, 8] key dual quaternions
    float* scores,           // [n, 8] output: dual quaternion inner product
    int n,
    cudaStream_t stream
);

// Extract primal similarity from dual quaternion inner product
void ggml_navigation_extract_primal_similarity_cuda_launch(
    const float* score_dq,    // [n, 8] dual quaternion inner product results
    float* similarities,      // [n] output: primal similarity scores
    int n,
    cudaStream_t stream
);

// Extract dual velocity magnitude from dual quaternion inner product
void ggml_navigation_extract_dual_velocity_cuda_launch(
    const float* score_dq,    // [n, 8] dual quaternion inner product results
    float* velocities,        // [n] output: dual velocity magnitudes
    int n,
    cudaStream_t stream
);

// Batch dual quaternion inner product with velocity-based gating
void ggml_navigation_batch_inner_product_gated_cuda_launch(
    const float* queries,     // [n_queries, 8] query dual quaternions
    const float* keys,        // [n_keys, 8] key dual quaternions
    const int* query_indices, // [n_pairs] query indices
    const int* key_indices,   // [n_pairs] key indices
    float* scores_dq,         // [n_pairs, 8] output: dual quaternion inner products
    float* similarities,      // [n_pairs] output: primal similarities
    float* velocities,        // [n_pairs] output: dual velocities
    bool* should_drill_down, // [n_pairs] output: drill-down decision
    int n_pairs,
    float similarity_threshold,  // Relevance threshold (default: 0.5)
    float velocity_threshold,    // Uncertainty threshold tau (default: 0.1)
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
