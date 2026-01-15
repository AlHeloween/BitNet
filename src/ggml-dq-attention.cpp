#include "ggml-dq-attention.h"
#include "ggml-dual-quaternion.h"
#include "ggml.h"
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"

#include <cmath>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
// CUDA kernel launch function
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
    );
}
#endif

using namespace DynaMap::math;

struct ggml_tensor* ggml_dq_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    struct ggml_tensor* mask,
    float semantic_weight,
    float contextual_weight
) {
    // q: [batch, seq_q, 8]
    // k: [batch, seq_k, 8]
    // v: [batch, seq_k, 8]
    // output: [batch, seq_q, 8]
    
    GGML_ASSERT(q->n_dims == 3);
    GGML_ASSERT(k->n_dims == 3);
    GGML_ASSERT(v->n_dims == 3);
    GGML_ASSERT(q->ne[2] == 8);
    GGML_ASSERT(k->ne[2] == 8);
    GGML_ASSERT(v->ne[2] == 8);
    GGML_ASSERT(q->ne[0] == k->ne[0] && q->ne[0] == v->ne[0]);  // batch must match
    GGML_ASSERT(k->ne[1] == v->ne[1]);  // seq_k must match
    
    int batch_size = q->ne[0];
    int seq_q = q->ne[1];
    int seq_k = k->ne[1];
    
    // Create attention scores: [batch, seq_q, seq_k]
    int64_t scores_ne[3] = {batch_size, seq_q, seq_k};
    struct ggml_tensor* scores = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, scores_ne);
    
    const float* q_data = (const float*)q->data;
    const float* k_data = (const float*)k->data;
    float* scores_data = (float*)scores->data;
    
    // Check if we should use CUDA backend
    bool use_cuda = false;
#ifdef GGML_USE_CUDA
    // Check if tensors are on CUDA backend
    // Note: GGML backend type checking - if available, use CUDA
    if (q->backend != GGML_BACKEND_TYPE_CPU || 
        k->backend != GGML_BACKEND_TYPE_CPU) {
        use_cuda = true;
    }
    // Also check if CUDA is available at runtime
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        use_cuda = true;  // CUDA available, prefer GPU
    }
#endif
    
    if (use_cuda) {
#ifdef GGML_USE_CUDA
        // Use CUDA kernel for attention score computation
        cudaStream_t stream = 0;  // Default stream
        ggml_dq_attention_scores_cuda_launch(
            q_data, k_data, scores_data,
            batch_size, seq_q, seq_k,
            semantic_weight, contextual_weight,
            stream
        );
        // Synchronize to ensure computation is complete
        cudaStreamSynchronize(stream);
        
        // Apply mask on CPU (if provided)
        if (mask) {
            const float* mask_data = (const float*)mask->data;
            for (int b = 0; b < batch_size; b++) {
                for (int i = 0; i < seq_q; i++) {
                    for (int j = 0; j < seq_k; j++) {
                        float mask_val = mask_data[(b * seq_q + i) * seq_k + j];
                        if (mask_val == 0.0f) {
                            scores_data[(b * seq_q + i) * seq_k + j] = -1e9f;
                        }
                    }
                }
            }
        }
#else
        use_cuda = false;  // Fallback to CPU if CUDA not compiled
#endif
    }
    
    if (!use_cuda) {
        // CPU implementation: Compute attention scores using dual quaternion similarity
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_q; i++) {
                for (int j = 0; j < seq_k; j++) {
                    // Load query dual quaternion Q
                const float* q_i = &q_data[(b * seq_q + i) * 8];
                Quaternion q_r(q_i[0], q_i[1], q_i[2], q_i[3]);
                Quaternion q_d(q_i[4], q_i[5], q_i[6], q_i[7]);
                
                // Load key dual quaternion K
                const float* k_j = &k_data[(b * seq_k + j) * 8];
                Quaternion k_r(k_j[0], k_j[1], k_j[2], k_j[3]);
                Quaternion k_d(k_j[4], k_j[5], k_j[6], k_j[7]);
                
                // Compute proper dual quaternion inner product: Score(Q,K) = Re(Q⊗K*) + εDu(Q⊗K*)
                // Step 1: Compute K* (conjugate of K)
                dualQuat k_dq(k_r, k_d);
                dualQuat k_conj = dualQuat::conjugate(k_dq);
                
                // Step 2: Compute Q⊗K* (Hamilton product)
                dualQuat q_dq(q_r, q_d);
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
                
                // Apply mask if provided
                if (mask) {
                    const float* mask_data = (const float*)mask->data;
                    float mask_val = mask_data[(b * seq_q + i) * seq_k + j];
                    if (mask_val == 0.0f) {
                        score = -1e9f;  // Large negative for softmax
                    }
                }
                
                    scores_data[(b * seq_q + i) * seq_k + j] = score;
                }
            }
        }
    }
    
    // Apply softmax
    struct ggml_tensor* attention_weights = ggml_soft_max(ctx, scores);
    
    // Weighted sum of values
    // output[i] = sum_j (attention_weights[i, j] * v[j])
    int64_t out_ne[3] = {batch_size, seq_q, 8};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, out_ne);
    
    const float* v_data = (const float*)v->data;
    const float* attn_data = (const float*)attention_weights->data;
    float* out_data = (float*)result->data;
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_q; i++) {
            float out_dq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            
            for (int j = 0; j < seq_k; j++) {
                float attn_weight = attn_data[(b * seq_q + i) * seq_k + j];
                
                // Load value: v[b, j, :]
                const float* v_j = &v_data[(b * seq_k + j) * 8];
                
                // Scalar multiplication: attn_weight * v_j
                for (int k = 0; k < 8; k++) {
                    out_dq[k] += attn_weight * v_j[k];
                }
            }
            
            // Store output
            float* out = &out_data[(b * seq_q + i) * 8];
            for (int k = 0; k < 8; k++) {
                out[k] = out_dq[k];
            }
        }
    }
    
    return result;
}
