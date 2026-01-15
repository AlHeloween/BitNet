#include "llama-aurora-attention.h"
#include "ggml-aurora-memory.h"
#include "ggml-dual-complex.h"
#include "aurora_memory_bank.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

void llama_aurora_attention_params_init(struct llama_aurora_attention_params* params) {
    if (!params) return;
    params->enable_aurora_memory = false;
    params->aurora_window_size = 2048;
    params->aurora_k_read = 8;
    params->aurora_candidate_slots = 4;
    params->memory_banks = NULL;
}

struct ggml_tensor* llama_aurora_extract_query_embedding(
    struct ggml_context* ctx,
    struct ggml_tensor* hidden_state,
    int32_t token_idx
) {
    if (!ctx || !hidden_state) {
        return NULL;
    }
    
    // Extract the embedding for the current token
    // hidden_state shape: [n_tokens, n_embd]
    // We want: [n_embd] for token at token_idx
    
    int64_t n_embd = hidden_state->ne[0];
    int64_t n_tokens = hidden_state->ne[1];
    
    if (token_idx < 0 || token_idx >= n_tokens) {
        return NULL;
    }
    
    // Create a view of the last token's embedding
    struct ggml_tensor* query_emb = ggml_view_1d(ctx, hidden_state, n_embd, token_idx * n_embd * sizeof(float));
    
    return query_emb;
}

void llama_aurora_window_kv_cache(
    struct llama_kv_cache* kv_cache,
    int32_t window_size
) {
    if (!kv_cache || window_size <= 0) {
        return;
    }
    
    // Get current KV cache size
    int32_t current_size = (int32_t)kv_cache->size;
    
    if (current_size <= window_size) {
        // No windowing needed
        return;
    }
    
    // Window KV cache: keep only last window_size tokens
    // This is a simplified implementation - in practice, we'd need to:
    // 1. Shift KV cache entries
    // 2. Update position indices
    // 3. Maintain cache consistency
    
    // For now, we'll just update the effective size
    // The actual windowing will be handled during attention computation
    kv_cache->size = window_size;
}

struct ggml_tensor* llama_aurora_bounded_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    int32_t window_size,
    aurora_memory_banks_t* memory_banks,
    int32_t k_read,
    int32_t candidate_slots
) {
    if (!ctx || !q || !k || !v) {
        return NULL;
    }
    
    // Get tensor dimensions
    int64_t n_head = q->ne[2];  // Assuming q shape: [head_dim, n_tokens, n_head]
    int64_t head_dim = q->ne[0];
    int64_t n_tokens = q->ne[1];
    int64_t n_kv = k->ne[1];
    
    // Apply windowing: only process last window_size tokens
    int64_t window_start = (n_kv > window_size) ? (n_kv - window_size) : 0;
    int64_t window_kv_size = n_kv - window_start;
    
    // Create windowed K and V tensors
    struct ggml_tensor* k_window = NULL;
    struct ggml_tensor* v_window = NULL;
    
    if (window_start > 0) {
        // Create views of the windowed portion
        size_t offset = window_start * head_dim * sizeof(float);
        k_window = ggml_view_3d(ctx, k, head_dim, window_kv_size, n_head, 
                                k->nb[0], k->nb[1], k->nb[2], offset);
        v_window = ggml_view_3d(ctx, v, head_dim, window_kv_size, n_head,
                                v->nb[0], v->nb[1], v->nb[2], offset);
    } else {
        k_window = k;
        v_window = v;
    }
    
    // Compute window attention: Q @ K_window^T
    struct ggml_tensor* kq_window = ggml_mul_mat(ctx, k_window, q);
    
    // Apply softmax
    struct ggml_tensor* attn_window = ggml_soft_max(ctx, kq_window);
    
    // Compute window output: attn_window @ V_window
    struct ggml_tensor* output_window = ggml_mul_mat(ctx, v_window, attn_window);
    
    // If memory banks are provided, augment with memory attention
    if (memory_banks && k_read > 0) {
        // Extract query embedding from current hidden state
        // For simplicity, we'll use the last token's query
        // In practice, we'd extract from the actual hidden state
        struct ggml_tensor* query_emb = ggml_view_1d(ctx, q, head_dim, 
                                                     (n_tokens - 1) * head_dim * sizeof(float));
        
        // Read from memory banks
        struct ggml_tensor* memory_embeddings = ggml_aurora_memory_read(
            ctx,
            query_emb,
            memory_banks,
            k_read,
            candidate_slots
        );
        
        if (memory_embeddings) {
            // Memory embeddings shape: [k_read, head_dim]
            // We need to reshape for attention: [n_head, k_read, head_dim]
            
            // Compute memory attention: Q @ memory_K^T
            // For simplicity, we'll use memory embeddings as both K and V
            struct ggml_tensor* memory_k = memory_embeddings;
            struct ggml_tensor* memory_v = memory_embeddings;
            
            // Reshape memory embeddings for multi-head attention
            // This is simplified - in practice, we'd need proper reshaping
            struct ggml_tensor* memory_kq = ggml_mul_mat(ctx, memory_k, q);
            struct ggml_tensor* attn_memory = ggml_soft_max(ctx, memory_kq);
            struct ggml_tensor* output_memory = ggml_mul_mat(ctx, memory_v, attn_memory);
            
            // Combine window and memory outputs
            // output = output_window + output_memory
            struct ggml_tensor* output = ggml_add(ctx, output_window, output_memory);
            
            return output;
        }
    }
    
    // Return window attention only
    return output_window;
}

// Dual-softmax implementation (JVP: Jacobian-vector product)
// p = softmax(scores_primal)
// dp = p * (scores_dual - sum(p * scores_dual))
static void dual_softmax(
    const float* scores_primal,
    const float* scores_dual,
    float* probs_primal,
    float* probs_dual,
    int n_scores
) {
    // Find max for numerical stability
    float max_score = scores_primal[0];
    for (int i = 1; i < n_scores; i++) {
        if (scores_primal[i] > max_score) {
            max_score = scores_primal[i];
        }
    }
    
    // Compute exp(scores - max) for primal
    float sum_exp = 0.0f;
    for (int i = 0; i < n_scores; i++) {
        float exp_val = expf(scores_primal[i] - max_score);
        probs_primal[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize to get softmax probabilities
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < n_scores; i++) {
        probs_primal[i] *= inv_sum;
    }
    
    // Compute dual probabilities: dp = p * (ds - sum(p*ds))
    float sum_p_ds = 0.0f;
    for (int i = 0; i < n_scores; i++) {
        sum_p_ds += probs_primal[i] * scores_dual[i];
    }
    
    for (int i = 0; i < n_scores; i++) {
        probs_dual[i] = probs_primal[i] * (scores_dual[i] - sum_p_ds);
    }
}

struct ggml_tensor* llama_aurora_bounded_attention_dual_complex(
    struct ggml_context* ctx,
    struct ggml_tensor* q_primal,
    struct ggml_tensor* q_dual,
    struct ggml_tensor* k_primal,
    struct ggml_tensor* k_dual,
    struct ggml_tensor* v_primal,
    struct ggml_tensor* v_dual,
    int32_t window_size,
    struct ggml_tensor* memory_embeddings_primal,
    struct ggml_tensor* memory_embeddings_dual,
    int32_t k_read
) {
    if (!ctx || !q_primal || !k_primal || !v_primal) {
        return NULL;
    }
    
    // Get tensor dimensions
    int64_t n_head = q_primal->ne[2];  // Assuming shape: [head_dim, n_tokens, n_head]
    int64_t head_dim = q_primal->ne[0];
    int64_t n_tokens = q_primal->ne[1];
    int64_t n_kv = k_primal->ne[1];
    
    // Apply windowing: only process last window_size tokens
    int64_t window_start = (n_kv > window_size) ? (n_kv - window_size) : 0;
    int64_t window_kv_size = n_kv - window_start;
    
    // For simplicity, we'll implement a basic version
    // In production, this would need proper tensor operations
    
    // Create windowed K and V tensors (primal)
    struct ggml_tensor* k_window_p = NULL;
    struct ggml_tensor* v_window_p = NULL;
    struct ggml_tensor* k_window_d = NULL;
    struct ggml_tensor* v_window_d = NULL;
    
    if (window_start > 0) {
        size_t offset = window_start * head_dim * sizeof(float);
        k_window_p = ggml_view_3d(ctx, k_primal, head_dim, window_kv_size, n_head,
                                  k_primal->nb[0], k_primal->nb[1], k_primal->nb[2], offset);
        v_window_p = ggml_view_3d(ctx, v_primal, head_dim, window_kv_size, n_head,
                                  v_primal->nb[0], v_primal->nb[1], v_primal->nb[2], offset);
        if (k_dual) {
            k_window_d = ggml_view_3d(ctx, k_dual, head_dim, window_kv_size, n_head,
                                      k_dual->nb[0], k_dual->nb[1], k_dual->nb[2], offset);
        }
        if (v_dual) {
            v_window_d = ggml_view_3d(ctx, v_dual, head_dim, window_kv_size, n_head,
                                      v_dual->nb[0], v_dual->nb[1], v_dual->nb[2], offset);
        }
    } else {
        k_window_p = k_primal;
        v_window_p = v_primal;
        k_window_d = k_dual;
        v_window_d = v_dual;
    }
    
    // Compute attention scores: Q @ K^T
    // Primal: scores_p = Qp @ Kp^T / sqrt(d)
    struct ggml_tensor* scores_primal = ggml_mul_mat(ctx, k_window_p, q_primal);
    // Scale by 1/sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);
    scores_primal = ggml_scale(ctx, scores_primal, scale);
    
    // Dual: scores_d = (Qd @ Kp^T + Qp @ Kd^T) / sqrt(d)
    struct ggml_tensor* scores_dual = NULL;
    if (q_dual && k_window_p) {
        struct ggml_tensor* scores_d1 = ggml_mul_mat(ctx, k_window_p, q_dual);
        scores_d1 = ggml_scale(ctx, scores_d1, scale);
        scores_dual = scores_d1;
    }
    if (q_primal && k_window_d) {
        struct ggml_tensor* scores_d2 = ggml_mul_mat(ctx, k_window_d, q_primal);
        scores_d2 = ggml_scale(ctx, scores_d2, scale);
        if (scores_dual) {
            scores_dual = ggml_add(ctx, scores_dual, scores_d2);
        } else {
            scores_dual = scores_d2;
        }
    }
    
    // Apply dual-softmax using ggml_dual_complex_softmax
    // Combine scores into dual-complex tensor
    struct ggml_tensor* scores_dual_safe = scores_dual;
    if (!scores_dual_safe) {
        scores_dual_safe = ggml_zeros_like(ctx, scores_primal);
    }
    struct ggml_tensor* scores_dc = ggml_dual_complex(ctx, scores_primal, scores_dual_safe);
    struct ggml_tensor* attn_dc = ggml_dual_complex_softmax(ctx, scores_dc);
    
    // Extract primal and dual attention weights
    struct ggml_tensor* attn_primal = ggml_dual_complex_primal(ctx, attn_dc);
    struct ggml_tensor* attn_dual = ggml_dual_complex_dual(ctx, attn_dc);
    
    // Compute attention output
    // Primal: op = attn_primal @ v_primal
    struct ggml_tensor* output_primal = ggml_mul_mat(ctx, v_window_p, attn_primal);
    
    // Dual: od = attn_primal @ v_dual + attn_dual @ v_primal
    struct ggml_tensor* output_dual = NULL;
    if (v_window_d) {
        struct ggml_tensor* od1 = ggml_mul_mat(ctx, v_window_d, attn_primal);
        output_dual = od1;
        
        // Add attn_dual @ v_primal term
        if (attn_dual) {
            struct ggml_tensor* od2 = ggml_mul_mat(ctx, v_window_p, attn_dual);
            output_dual = ggml_add(ctx, output_dual, od2);
        }
    } else if (attn_dual) {
        // If no v_dual, still compute attn_dual @ v_primal
        output_dual = ggml_mul_mat(ctx, v_window_p, attn_dual);
    }
    
    // Combine primal and dual into dual-complex tensor [n_head, n_tokens, head_dim, 2]
    // where [..., 0] = primal, [..., 1] = dual
    if (!output_dual) {
        // If no dual component, create zero dual
        output_dual = ggml_zeros_like(ctx, output_primal);
    }
    
    // Use ggml_dual_complex to stack primal and dual
    struct ggml_tensor* output_dc = ggml_dual_complex(ctx, output_primal, output_dual);
    
    return output_dc;  // Returns dual-complex tensor [..., 2]
}
