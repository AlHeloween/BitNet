#pragma once

#include "llama.h"
#include "ggml-aurora-memory.h"
#include "aurora_memory_bank.h"

#ifdef __cplusplus
extern "C" {
#endif

// Aurora Bounded Window Attention
// Implements O(w·Q) attention with memory backfill for compute reduction

// Configuration for Aurora attention
struct llama_aurora_attention_params {
    bool enable_aurora_memory;      // Enable Aurora memory integration
    int32_t aurora_window_size;     // Window size w (default: 2048)
    int32_t aurora_k_read;          // Number of memory embeddings to retrieve (default: 8)
    int32_t aurora_candidate_slots; // Number of candidate slots to search (default: 4)
    aurora_memory_banks_t* memory_banks; // Memory banks pointer
};

// Initialize Aurora attention parameters with defaults
void llama_aurora_attention_params_init(struct llama_aurora_attention_params* params);

// Apply bounded window attention with memory augmentation
// This modifies the attention computation to:
// 1. Process only last w tokens with full attention (O(w·Q))
// 2. Retrieve k_read embeddings from memory banks
// 3. Combine window attention with memory attention
//
// Inputs:
//   - ctx: GGML context
//   - q: Query tensor [n_head, n_tokens, head_dim]
//   - k: Key tensor [n_head, n_kv, head_dim] (may be larger than window)
//   - v: Value tensor [n_head, n_kv, head_dim]
//   - window_size: Window size w
//   - memory_banks: Aurora memory banks
//   - k_read: Number of memory embeddings to retrieve
//   - candidate_slots: Number of candidate slots
//
// Returns:
//   - Attention output tensor [n_head, n_tokens, head_dim]
struct ggml_tensor* llama_aurora_bounded_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    int32_t window_size,
    aurora_memory_banks_t* memory_banks,
    int32_t k_read,
    int32_t candidate_slots
);
// Dual-complex version: accepts dual-complex Q/K/V and memory embeddings
// Returns: [n_head, n_tokens, head_dim, 2] where [..., 0] = primal, [..., 1] = dual
struct ggml_tensor* llama_aurora_bounded_attention_dual_complex(
    struct ggml_context* ctx,
    struct ggml_tensor* q_primal,
    struct ggml_tensor* q_dual,
    struct ggml_tensor* k_primal,
    struct ggml_tensor* k_dual,
    struct ggml_tensor* v_primal,
    struct ggml_tensor* v_dual,
    int32_t window_size,
    struct ggml_tensor* memory_embeddings_primal,  // [k_read, dim] or NULL
    struct ggml_tensor* memory_embeddings_dual,    // [k_read, dim] or NULL
    int32_t k_read
);

// Apply windowing to KV cache (keep only last w tokens)
// This enforces the bounded window size in the KV cache
void llama_aurora_window_kv_cache(
    struct llama_kv_cache* kv_cache,
    int32_t window_size
);

// Extract query embedding from hidden state for memory lookup
// This extracts the current query embedding to use for memory retrieval
struct ggml_tensor* llama_aurora_extract_query_embedding(
    struct ggml_context* ctx,
    struct ggml_tensor* hidden_state,  // Current hidden state [n_tokens, n_embd]
    int32_t token_idx                  // Index of current token (usually last)
);

#ifdef __cplusplus
}
#endif
