#include <cuda_runtime.h>
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// Kernel: Dual quaternion inner product for Navigation paradigm
// Computes ⟨q, k⟩dq = Q ⊗ K* (Hamilton product of Q with conjugate of K)
// Used for relevance check (primal similarity) and uncertainty signal (dual velocity)
__global__ void ggml_navigation_dual_quat_inner_product_cuda(
    const float* q,           // [n, 8] query dual quaternions
    const float* k,           // [n, 8] key dual quaternions
    float* scores,           // [n, 8] output: dual quaternion inner product
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Load query dual quaternion Q
    Quaternion q_r(q[idx * 8 + 0], q[idx * 8 + 1], 
                   q[idx * 8 + 2], q[idx * 8 + 3]);
    Quaternion q_d(q[idx * 8 + 4], q[idx * 8 + 5], 
                   q[idx * 8 + 6], q[idx * 8 + 7]);
    dualQuat q_dq(q_r, q_d);
    
    // Load key dual quaternion K
    Quaternion k_r(k[idx * 8 + 0], k[idx * 8 + 1], 
                   k[idx * 8 + 2], k[idx * 8 + 3]);
    Quaternion k_d(k[idx * 8 + 4], k[idx * 8 + 5], 
                   k[idx * 8 + 6], k[idx * 8 + 7]);
    dualQuat k_dq(k_r, k_d);
    
    // Step 1: Compute K* (conjugate of K)
    dualQuat k_conj = dualQuat::conjugate(k_dq);
    
    // Step 2: Compute Q ⊗ K* (Hamilton product)
    dualQuat qk_star = dualQuat::mul(q_dq, k_conj);
    
    // Store result: [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]
    scores[idx * 8 + 0] = qk_star.real.w;
    scores[idx * 8 + 1] = qk_star.real.x;
    scores[idx * 8 + 2] = qk_star.real.y;
    scores[idx * 8 + 3] = qk_star.real.z;
    scores[idx * 8 + 4] = qk_star.dual.w;
    scores[idx * 8 + 5] = qk_star.dual.x;
    scores[idx * 8 + 6] = qk_star.dual.y;
    scores[idx * 8 + 7] = qk_star.dual.z;
}

// Kernel: Extract primal similarity from dual quaternion inner product
// Uses rotation quaternion's w component (cosine similarity)
__global__ void ggml_navigation_extract_primal_similarity_cuda(
    const float* score_dq,    // [n, 8] dual quaternion inner product results
    float* similarities,      // [n] output: primal similarity scores
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Extract w component of primal part (rotation quaternion)
    float w = score_dq[idx * 8 + 0];
    
    // Normalize to [0, 1] range (for normalized quaternions, w ∈ [-1, 1])
    float similarity = (w + 1.0f) / 2.0f;
    
    similarities[idx] = similarity;
}

// Kernel: Extract dual velocity magnitude from dual quaternion inner product
// Uses translation quaternion's magnitude as "semantic variance" signal
__global__ void ggml_navigation_extract_dual_velocity_cuda(
    const float* score_dq,    // [n, 8] dual quaternion inner product results
    float* velocities,        // [n] output: dual velocity magnitudes
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Extract dual part (translation quaternion): [4:8]
    float dx = score_dq[idx * 8 + 4];
    float dy = score_dq[idx * 8 + 5];
    float dz = score_dq[idx * 8 + 6];
    float dw = score_dq[idx * 8 + 7];
    
    // Compute magnitude: ||zd|| = sqrt(dx² + dy² + dz² + dw²)
    float velocity = sqrtf(dx*dx + dy*dy + dz*dz + dw*dw);
    
    velocities[idx] = velocity;
}

// Kernel: Batch dual quaternion inner product with velocity-based gating
// Processes multiple query-key pairs in parallel and applies velocity threshold
__global__ void ggml_navigation_batch_inner_product_gated_cuda(
    const float* queries,     // [n_queries, 8] query dual quaternions
    const float* keys,         // [n_keys, 8] key dual quaternions
    const int* query_indices,  // [n_pairs] query indices
    const int* key_indices,    // [n_pairs] key indices
    float* scores_dq,       // [n_pairs, 8] output: dual quaternion inner products
    float* similarities,      // [n_pairs] output: primal similarities
    float* velocities,        // [n_pairs] output: dual velocities
    bool* should_drill_down, // [n_pairs] output: drill-down decision
    int n_pairs,
    float similarity_threshold,  // Relevance threshold (default: 0.5)
    float velocity_threshold     // Uncertainty threshold tau (default: 0.1)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;
    
    int q_idx = query_indices[idx];
    int k_idx = key_indices[idx];
    
    // Load query and key
    Quaternion q_r(queries[q_idx * 8 + 0], queries[q_idx * 8 + 1], 
                   queries[q_idx * 8 + 2], queries[q_idx * 8 + 3]);
    Quaternion q_d(queries[q_idx * 8 + 4], queries[q_idx * 8 + 5], 
                   queries[q_idx * 8 + 6], queries[q_idx * 8 + 7]);
    dualQuat q_dq(q_r, q_d);
    
    Quaternion k_r(keys[k_idx * 8 + 0], keys[k_idx * 8 + 1], 
                   keys[k_idx * 8 + 2], keys[k_idx * 8 + 3]);
    Quaternion k_d(keys[k_idx * 8 + 4], keys[k_idx * 8 + 5], 
                   keys[k_idx * 8 + 6], keys[k_idx * 8 + 7]);
    dualQuat k_dq(k_r, k_d);
    
    // Compute inner product: Q ⊗ K*
    dualQuat k_conj = dualQuat::conjugate(k_dq);
    dualQuat qk_star = dualQuat::mul(q_dq, k_conj);
    
    // Store dual quaternion result
    scores_dq[idx * 8 + 0] = qk_star.real.w;
    scores_dq[idx * 8 + 1] = qk_star.real.x;
    scores_dq[idx * 8 + 2] = qk_star.real.y;
    scores_dq[idx * 8 + 3] = qk_star.real.z;
    scores_dq[idx * 8 + 4] = qk_star.dual.w;
    scores_dq[idx * 8 + 5] = qk_star.dual.x;
    scores_dq[idx * 8 + 6] = qk_star.dual.y;
    scores_dq[idx * 8 + 7] = qk_star.dual.z;
    
    // Extract primal similarity
    float w = qk_star.real.w;
    float similarity = (w + 1.0f) / 2.0f;
    similarities[idx] = similarity;
    
    // Extract dual velocity
    float dx = qk_star.dual.w;
    float dy = qk_star.dual.x;
    float dz = qk_star.dual.y;
    float dw = qk_star.dual.z;
    float velocity = sqrtf(dx*dx + dy*dy + dz*dz + dw*dw);
    velocities[idx] = velocity;
    
    // Policy Gate: Drill down if similarity > threshold AND velocity > tau
    should_drill_down[idx] = (similarity > similarity_threshold) && (velocity > velocity_threshold);
}

// Host wrappers
extern "C" {
    void ggml_navigation_dual_quat_inner_product_cuda_launch(
        const float* q,
        const float* k,
        float* scores,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_navigation_dual_quat_inner_product_cuda<<<blocks, threads, 0, stream>>>(
            q, k, scores, n
        );
    }
    
    void ggml_navigation_extract_primal_similarity_cuda_launch(
        const float* score_dq,
        float* similarities,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_navigation_extract_primal_similarity_cuda<<<blocks, threads, 0, stream>>>(
            score_dq, similarities, n
        );
    }
    
    void ggml_navigation_extract_dual_velocity_cuda_launch(
        const float* score_dq,
        float* velocities,
        int n,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ggml_navigation_extract_dual_velocity_cuda<<<blocks, threads, 0, stream>>>(
            score_dq, velocities, n
        );
    }
    
    void ggml_navigation_batch_inner_product_gated_cuda_launch(
        const float* queries,
        const float* keys,
        const int* query_indices,
        const int* key_indices,
        float* scores_dq,
        float* similarities,
        float* velocities,
        bool* should_drill_down,
        int n_pairs,
        float similarity_threshold,
        float velocity_threshold,
        cudaStream_t stream
    ) {
        int threads = 256;
        int blocks = (n_pairs + threads - 1) / threads;
        ggml_navigation_batch_inner_product_gated_cuda<<<blocks, threads, 0, stream>>>(
            queries, keys, query_indices, key_indices,
            scores_dq, similarities, velocities, should_drill_down,
            n_pairs, similarity_threshold, velocity_threshold
        );
    }
}
