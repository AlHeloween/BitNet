#include <cuda_runtime.h>
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"
#include "../../lib/cuda_math.h"

using namespace DynaMap::math;

// Dual-quaternion attention using proper Q⊗K* formula
// Score(Q,K) = Re(Q⊗K*) + εDu(Q⊗K*)
// Where Q⊗K* is Hamilton product of Q with conjugate of K
// Real part (Re): Semantic similarity from rotation component
// Dual part (Du): Kinematic penalty from translation component (filters invalid paths)

__global__ void ggml_dq_attention_scores_cuda(
    const float* q,           // [batch, seq_q, 8]
    const float* k,           // [batch, seq_k, 8]
    float* scores,            // [batch, seq_q, seq_k]
    int batch_size,
    int seq_q,
    int seq_k,
    float semantic_weight,
    float contextual_weight
) {
    int batch = blockIdx.x;
    int i = blockIdx.y;  // query position
    int j = threadIdx.x;  // key position
    
    if (batch >= batch_size || i >= seq_q || j >= seq_k) {
        return;
    }
    
    // Load query dual quaternion Q
    const float* q_i = &q[(batch * seq_q + i) * 8];
    Quaternion q_r(q_i[0], q_i[1], q_i[2], q_i[3]);
    Quaternion q_d(q_i[4], q_i[5], q_i[6], q_i[7]);
    dualQuat q_dq(q_r, q_d);
    
    // Load key dual quaternion K
    const float* k_j = &k[(batch * seq_k + j) * 8];
    Quaternion k_r(k_j[0], k_j[1], k_j[2], k_j[3]);
    Quaternion k_d(k_j[4], k_j[5], k_j[6], k_j[7]);
    dualQuat k_dq(k_r, k_d);
    
    // Step 1: Compute K* (conjugate of K)
    dualQuat k_conj = dualQuat::conjugate(k_dq);
    
    // Step 2: Compute Q⊗K* (Hamilton product)
    dualQuat qk_star = dualQuat::mul(q_dq, k_conj);
    
    // Step 3: Extract real part (semantic similarity from rotation)
    // The w component of rotation quaternion represents cosine similarity
    float semantic_sim = qk_star.real.w;
    
    // Step 4: Extract dual part (kinematic penalty from translation)
    // Compute magnitude of translation quaternion as penalty
    float kinematic_penalty = sqrtf(
        qk_star.dual.w*qk_star.dual.w +
        qk_star.dual.x*qk_star.dual.x +
        qk_star.dual.y*qk_star.dual.y +
        qk_star.dual.z*qk_star.dual.z
    );
    
    // Step 5: Combine: semantic similarity minus kinematic penalty
    // Higher semantic + lower penalty = higher score
    // The kinematic penalty acts as a "hallucination filter"
    float score = semantic_weight * semantic_sim - contextual_weight * kinematic_penalty;
    
    scores[(batch * seq_q + i) * seq_k + j] = score;
}

// Host wrapper
extern "C" {
    void ggml_dq_attention_scores_cuda_launch(
        const float* q,
        const float* k,
        float* scores,
        int batch_size,
        int seq_q,
        int seq_k,
        float semantic_weight,
        float contextual_weight,
        cudaStream_t stream
    ) {
        dim3 blocks(batch_size, seq_q);
        int threads = seq_k;
        ggml_dq_attention_scores_cuda<<<blocks, threads, 0, stream>>>(
            q, k, scores, batch_size, seq_q, seq_k, semantic_weight, contextual_weight
        );
    }
}
