#include "llama-aurora-navigation.h"
#include "ggml-aurora-memory.h"
#include "aurora_memory_bank.h"
#include "ggml.h"
#include "ggml-gpu-metrics.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include "ggml-navigation-cuda.h"
#endif

// Initialize Navigation paradigm parameters with defaults
void llama_aurora_navigation_params_init(struct llama_aurora_navigation_params* params) {
    if (!params) return;
    params->enable_navigation_paradigm = false;
    params->tau = 0.1f;
    params->max_depth = 7;
    params->max_nodes_per_level = 4;
    params->apply_temporal_decay = false;
    params->current_time = 0.0f;
}

// Helper: Compute dual quaternion conjugate
// For dual quaternion q = q_p + ε*q_d, conjugate is q* = q_p* + ε*q_d*
// where q_p* is quaternion conjugate (w, -x, -y, -z)
static void dual_quat_conjugate(const float* q, float* q_conj) {
    // Primal part (quaternion conjugate): [w, x, y, z] -> [w, -x, -y, -z]
    q_conj[0] = q[0];   // w
    q_conj[1] = -q[1];  // -x
    q_conj[2] = -q[2];  // -y
    q_conj[3] = -q[3];  // -z
    
    // Dual part (quaternion conjugate): [w, x, y, z] -> [w, -x, -y, -z]
    q_conj[4] = q[4];   // w
    q_conj[5] = -q[5];  // -x
    q_conj[6] = -q[6];  // -y
    q_conj[7] = -q[7];  // -z
}

// Helper: Hamilton product of two quaternions
// q1 ⊗ q2 = (w1*w2 - x1*x2 - y1*y2 - z1*z2,
//            w1*x2 + x1*w2 + y1*z2 - z1*y2,
//            w1*y2 - x1*z2 + y1*w2 + z1*x2,
//            w1*z2 + x1*y2 - y1*x2 + z1*w2)
static void hamilton_product(const float* q1, const float* q2, float* result) {
    float w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    float w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
    
    result[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    result[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    result[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    result[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

// Helper: Dual quaternion Hamilton product
// For dual quaternions q = q_p + ε*q_d, k = k_p + ε*k_d:
// q ⊗ k = q_p ⊗ k_p + ε*(q_p ⊗ k_d + q_d ⊗ k_p)
static void dual_quat_hamilton_product(const float* q, const float* k, float* result) {
    // Primal part: q_p ⊗ k_p
    hamilton_product(q, k, result);
    
    // Dual part: q_p ⊗ k_d + q_d ⊗ k_p
    float dual_temp1[4], dual_temp2[4];
    hamilton_product(q, k + 4, dual_temp1);      // q_p ⊗ k_d
    hamilton_product(q + 4, k, dual_temp2);    // q_d ⊗ k_p
    
    // Add dual parts
    for (int i = 0; i < 4; i++) {
        result[4 + i] = dual_temp1[i] + dual_temp2[i];
    }
}

// Extract primal similarity from dual quaternion inner product
// Uses rotation quaternion's w component (cosine similarity)
float llama_aurora_extract_primal_similarity(struct ggml_tensor* score_dq) {
    if (!score_dq || score_dq->ne[0] != 8) {
        return 0.0f;
    }
    
    // Extract w component of primal part (rotation quaternion)
    float w = ((float*)score_dq->data)[0];
    
    // Normalize to [0, 1] range (for normalized quaternions, w ∈ [-1, 1])
    float similarity = (w + 1.0f) / 2.0f;
    
    return similarity;
}

// Extract dual velocity magnitude from dual quaternion inner product
// Uses translation quaternion's magnitude as "semantic variance" signal
float llama_aurora_extract_dual_velocity(struct ggml_tensor* score_dq) {
    if (!score_dq || score_dq->ne[0] != 8) {
        return 0.0f;
    }
    
    float* data = (float*)score_dq->data;
    
    // Extract dual part (translation quaternion): [4:8]
    float dx = data[4];
    float dy = data[5];
    float dz = data[6];
    float dw = data[7];
    
    // Compute magnitude: ||zd|| = sqrt(dx² + dy² + dz² + dw²)
    float velocity = sqrtf(dx*dx + dy*dy + dz*dz + dw*dw);
    
    return velocity;
}

// Recursive drill-down search using velocity-based gating
// This implements the Navigation paradigm algorithm
struct ggml_tensor* llama_aurora_fractal_drill_down(
    struct ggml_context* ctx,
    struct ggml_tensor* query_dq,
    aurora_memory_banks_t* memory_banks,
    const struct llama_aurora_navigation_params* nav_params
) {
    if (!ctx || !query_dq || !memory_banks || !nav_params) {
        return NULL;
    }
    
    if (!nav_params->enable_navigation_paradigm) {
        return NULL;
    }
    
    // Validate query_dq shape: must be [8] (dual quaternion)
    if (query_dq->ne[0] != 8) {
        return NULL;
    }
    
    // Extract query dual quaternion
    float query[8];
    memcpy(query, query_dq->data, 8 * sizeof(float));
    
    // Collect selected embeddings from recursive search
    std::vector<float> selected_embeddings;
    
    // Start recursive search at root level (Level 0)
    // For now, we'll use a simplified approach:
    // 1. Query memory banks for top-K entries
    // 2. For each entry, compute dual quaternion inner product
    // 3. Extract primal similarity and dual velocity
    // 4. If velocity > tau, drill down (recurse on children)
    // 5. Otherwise, use this entry's summary
    
    // Query memory banks for candidate entries
    int k_read = nav_params->max_nodes_per_level;
    int candidate_slots = 4;  // Default candidate slots
    
    // Allocate entries array
    aurora_memory_entry_t** entries = (aurora_memory_entry_t**)malloc(
        k_read * sizeof(aurora_memory_entry_t*)
    );
    if (!entries) {
        return NULL;
    }
    
    // Query verified bank
    // Extract primal and dual parts from query dual quaternion
    float query_primal[4];
    float query_dual[4];
    memcpy(query_primal, query, 4 * sizeof(float));
    memcpy(query_dual, query + 4, 4 * sizeof(float));
    
    int n_found = aurora_memory_bank_query_dual_complex(
        memory_banks->verified,
        query_primal,    // query_primal [4]
        query_dual,      // query_dual [4]
        candidate_slots,
        k_read,
        0.1f,            // dual_weight
        entries
    );
    
    // Prepare for batch processing (CPU or GPU)
    std::vector<float> query_batch;
    std::vector<float> key_batch;
    std::vector<int> entry_indices;
    
    // Collect valid entries for batch processing
    for (int i = 0; i < n_found && i < k_read; i++) {
        aurora_memory_entry_t* entry = entries[i];
        if (!entry || !entry->embedding) {
            continue;
        }
        
        // Prepare key dual quaternion
        float k[8];
        memcpy(k, entry->embedding, 4 * sizeof(float));
        if (entry->is_dual_complex && entry->embedding_dual) {
            memcpy(k + 4, entry->embedding_dual, 4 * sizeof(float));
        } else {
            memset(k + 4, 0, 4 * sizeof(float));
        }
        
        // Add to batch
        query_batch.insert(query_batch.end(), query, query + 8);
        key_batch.insert(key_batch.end(), k, k + 8);
        entry_indices.push_back(i);
    }
    
    int n_pairs = entry_indices.size();
    if (n_pairs == 0) {
        free(entries);
        return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 0);
    }
    
    // Allocate result arrays
    std::vector<float> scores_dq(n_pairs * 8);
    std::vector<float> similarities(n_pairs);
    std::vector<float> velocities(n_pairs);
    // Use uint8_t instead of bool for contiguous memory (std::vector<bool> is specialized and doesn't have .data())
    std::vector<uint8_t> should_drill_down(n_pairs);
    
#ifdef GGML_USE_CUDA
    // Use CUDA kernels for batch processing
    cudaStream_t stream = 0;  // Default stream
    
    // Allocate GPU memory
    float* d_queries = NULL;
    float* d_keys = NULL;
    int* d_query_indices = NULL;
    int* d_key_indices = NULL;
    float* d_scores_dq = NULL;
    float* d_similarities = NULL;
    float* d_velocities = NULL;
    bool* d_should_drill_down = NULL;
    
    cudaMalloc(&d_queries, n_pairs * 8 * sizeof(float));
    cudaMalloc(&d_keys, n_pairs * 8 * sizeof(float));
    cudaMalloc(&d_query_indices, n_pairs * sizeof(int));
    cudaMalloc(&d_key_indices, n_pairs * sizeof(int));
    cudaMalloc(&d_scores_dq, n_pairs * 8 * sizeof(float));
    cudaMalloc(&d_similarities, n_pairs * sizeof(float));
    cudaMalloc(&d_velocities, n_pairs * sizeof(float));
    cudaMalloc(&d_should_drill_down, n_pairs * sizeof(bool));
    
    // Prepare indices (each query-key pair)
    std::vector<int> query_indices_vec(n_pairs, 0);  // All use same query
    std::vector<int> key_indices_vec(n_pairs);
    for (int i = 0; i < n_pairs; i++) {
        key_indices_vec[i] = i;
    }
    
    // Time memory transfers
    float transfer_time = 0.0f;
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        cudaMemcpy(d_queries, query_batch.data(), n_pairs * 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_keys, key_batch.data(), n_pairs * 8 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_query_indices, query_indices_vec.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_key_indices, key_indices_vec.data(), n_pairs * sizeof(int), cudaMemcpyHostToDevice);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&transfer_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Time CUDA kernel execution
    float kernel_time = 0.0f;
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // Launch CUDA kernel
        ggml_navigation_batch_inner_product_gated_cuda_launch(
            d_queries,
            d_keys,
            d_query_indices,
            d_key_indices,
            d_scores_dq,
            d_similarities,
            d_velocities,
            d_should_drill_down,
            n_pairs,
            0.5f,  // similarity_threshold
            nav_params->tau,  // velocity_threshold
            stream
        );
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernel_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Time result copy back
    float copy_back_time = 0.0f;
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        cudaMemcpy(scores_dq.data(), d_scores_dq, n_pairs * 8 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(similarities.data(), d_similarities, n_pairs * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(velocities.data(), d_velocities, n_pairs * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(should_drill_down.data(), d_should_drill_down, n_pairs * sizeof(bool), cudaMemcpyDeviceToHost);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&copy_back_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Log GPU metrics (if enabled)
    float total_gpu_time = transfer_time + kernel_time + copy_back_time;
    ggml_cuda_log_metrics("FRACTAL_DRILL_DOWN", total_gpu_time, n_pairs * 8 * sizeof(float));
    ggml_cuda_log_metrics("DUAL_QUAT_OPS", kernel_time, n_pairs * 8 * sizeof(float));
    ggml_cuda_log_metrics("MEMORY_TRANSFER", transfer_time + copy_back_time, n_pairs * 8 * sizeof(float));
    
    // Free GPU memory
    cudaFree(d_queries);
    cudaFree(d_keys);
    cudaFree(d_query_indices);
    cudaFree(d_key_indices);
    cudaFree(d_scores_dq);
    cudaFree(d_similarities);
    cudaFree(d_velocities);
    cudaFree(d_should_drill_down);
#else
    // CPU fallback: process sequentially
    for (int i = 0; i < n_pairs; i++) {
        const float* q_ptr = &query_batch[i * 8];
        const float* k_ptr = &key_batch[i * 8];
        
        // Compute K* (conjugate)
        float k_conj[8];
        dual_quat_conjugate(k_ptr, k_conj);
        
        // Compute Q ⊗ K* (Hamilton product)
        float score_dq_result[8];
        dual_quat_hamilton_product(q_ptr, k_conj, score_dq_result);
        
        // Extract similarity and velocity
        similarities[i] = (score_dq_result[0] + 1.0f) / 2.0f;
        velocities[i] = sqrtf(
            score_dq_result[4] * score_dq_result[4] +
            score_dq_result[5] * score_dq_result[5] +
            score_dq_result[6] * score_dq_result[6] +
            score_dq_result[7] * score_dq_result[7]
        );
        
        // Policy gate (store as uint8_t: 0 = false, 1 = true)
        should_drill_down[i] = ((similarities[i] > 0.5f) && (velocities[i] > nav_params->tau)) ? 1 : 0;
        
        // Store score_dq
        memcpy(&scores_dq[i * 8], score_dq_result, 8 * sizeof(float));
    }
#endif
    
    // Process results and collect selected embeddings
    for (int i = 0; i < n_pairs; i++) {
        // Apply temporal decay if enabled
        float velocity = velocities[i];
        if (nav_params->apply_temporal_decay) {
            aurora_memory_entry_t* entry = entries[entry_indices[i]];
            if (entry && entry->embedding_dual) {
                float decay_lambda = 0.01f;
                float dt = nav_params->current_time;
                float decay_factor = expf(-decay_lambda * dt);
                velocity *= decay_factor;
            }
        }
        
        // RELEVANCE CHECK: Is this branch even related?
        if (similarities[i] > 0.5f) {
            // SCALING LOGIC: The "Policy Gate"
            // If velocity is high, drill down; else use summary
            if (velocity > nav_params->tau) {
                // Drill down: For now, we'll use the entry itself
                // In a full implementation, we would recurse on children
            }
            
            // Add entry embedding to selected embeddings
            aurora_memory_entry_t* entry = entries[entry_indices[i]];
            for (int j = 0; j < 4; j++) {
                selected_embeddings.push_back(entry->embedding[j]);
            }
            if (entry->is_dual_complex && entry->embedding_dual) {
                for (int j = 0; j < 4; j++) {
                    selected_embeddings.push_back(entry->embedding_dual[j]);
                }
            } else {
                // Pad dual with zeros
                for (int j = 0; j < 4; j++) {
                    selected_embeddings.push_back(0.0f);
                }
            }
        }
    }
    
    // Free entries array
    free(entries);
    
    // Create output tensor
    int k_selected = selected_embeddings.size() / 8;
    if (k_selected == 0) {
        // Return empty tensor
        return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 0);
    }
    
    struct ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, k_selected);
    if (result && selected_embeddings.size() > 0) {
        memcpy(result->data, selected_embeddings.data(), selected_embeddings.size() * sizeof(float));
    }
    
    return result;
}

// Hybrid Fractal Attention: Combines local window + global fractal context
struct ggml_tensor* llama_aurora_hybrid_fractal_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    int32_t window_size,
    aurora_memory_banks_t* memory_banks,
    const struct llama_aurora_navigation_params* nav_params
) {
    if (!ctx || !q || !k || !v || !memory_banks || !nav_params) {
        return NULL;
    }
    
    if (!nav_params->enable_navigation_paradigm) {
        // Fall back to standard window attention
        // Use existing bounded attention as fallback
        return NULL;
    }
    
    // Extract dimensions
    // GGML tensor layout: [head_dim, n_tokens, n_head]
    int head_dim = (int)q->ne[0];
    int n_tokens = (int)q->ne[1];
    int n_head = (int)q->ne[2];
    
    // 1. Local Attention: Standard dense attention on last W tokens
    int64_t w = (window_size < n_tokens) ? window_size : n_tokens;
    int64_t n_kv = k->ne[1];
    int64_t window_start = (n_kv > w) ? (n_kv - w) : 0;
    int64_t window_kv_size = n_kv - window_start;
    
    // Extract window from k and v: last W tokens
    struct ggml_tensor* k_window = NULL;
    struct ggml_tensor* v_window = NULL;
    
    if (window_start > 0) {
        // Create views of the windowed portion
        size_t offset = window_start * head_dim * sizeof(float);
        k_window = ggml_view_3d(ctx, k, head_dim, window_kv_size, n_head,
                                k->nb[1], k->nb[2], offset);
        v_window = ggml_view_3d(ctx, v, head_dim, window_kv_size, n_head,
                                v->nb[1], v->nb[2], offset);
    } else {
        k_window = k;
        v_window = v;
    }
    
    // Compute local attention: Q @ K_window^T
    struct ggml_tensor* kq_window = ggml_mul_mat(ctx, k_window, q);
    
    // Apply softmax
    struct ggml_tensor* attn_window = ggml_soft_max(ctx, kq_window);
    
    // Compute local output: attn_window @ V_window
    struct ggml_tensor* local_out = ggml_mul_mat(ctx, v_window, attn_window);
    
    // 2. Global Attention: Fractal memory retrieval
    // Extract query dual quaternion from last token
    // For dual quaternion, head_dim should be 8
    if (head_dim != 8) {
        // Not dual quaternion format, return local attention only
        return local_out;
    }
    
    // Create query tensor for fractal drill-down (last token, first head)
    // Extract [8] from last token
    struct ggml_tensor* query_dq = ggml_view_1d(
        ctx, q, 8,
        (n_tokens - 1) * q->nb[1]  // Last token offset
    );
    
    // Get global context via fractal drill-down
    struct ggml_tensor* global_context = llama_aurora_fractal_drill_down(
        ctx,
        query_dq,
        memory_banks,
        nav_params
    );
    
    if (!global_context || global_context->ne[1] == 0) {
        // No global context, return local attention only
        return local_out;
    }
    
    // 3. Global Cross-Attention: Q attends to global context
    // global_context shape: [8, k_selected] (dual quaternion format)
    int k_selected = (int)global_context->ne[1];
    
    if (k_selected == 0) {
        // No global context, return local attention only
        return local_out;
    }
    
    // For now, we'll use a simplified approach:
    // Use global_context as memory embeddings and compute cross-attention
    // In a full implementation, we'd properly reshape for multi-head attention
    
    // Reshape global_context to match attention format
    // global_context is [8, k_selected], we need [head_dim, k_selected, n_head]
    // For simplicity, we'll create a view and use it as both K and V
    struct ggml_tensor* k_global = global_context;  // [8, k_selected]
    struct ggml_tensor* v_global = global_context;  // Use same as K
    
    // Compute global attention: Q @ K_global^T
    // Note: This is simplified - proper implementation would handle multi-head correctly
    struct ggml_tensor* kq_global = ggml_mul_mat(ctx, k_global, q);
    
    // Apply softmax
    struct ggml_tensor* attn_global = ggml_soft_max(ctx, kq_global);
    
    // Compute global output: attn_global @ V_global
    struct ggml_tensor* global_out = ggml_mul_mat(ctx, v_global, attn_global);
    
    // 4. Combine local + global (weighted combination)
    // Weight: 70% local, 30% global (configurable)
    float local_weight = 0.7f;
    float global_weight = 0.3f;
    
    struct ggml_tensor* local_scaled = ggml_scale(ctx, local_out, local_weight);
    struct ggml_tensor* global_scaled = ggml_scale(ctx, global_out, global_weight);
    struct ggml_tensor* combined = ggml_add(ctx, local_scaled, global_scaled);
    
    return combined;
}
