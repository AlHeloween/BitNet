#pragma once

#include "llama.h"
#include "ggml-aurora-memory.h"
#include "aurora_memory_bank.h"

#ifdef __cplusplus
extern "C" {
#endif

// Navigation Paradigm: Velocity-based fractal drill-down
// Implements O(W·Q) + O(K·log N) scaling instead of O(Q²)

// Configuration for Navigation paradigm
struct llama_aurora_navigation_params {
    bool enable_navigation_paradigm;  // Enable Navigation paradigm (Stage 4)
    float tau;                         // Uncertainty threshold for drill-down (default: 0.1)
    int32_t max_depth;                 // Maximum fractal level to drill down (default: 7)
    int32_t max_nodes_per_level;       // Maximum nodes per level (hard cap, default: 4)
    bool apply_temporal_decay;          // Apply temporal decay to dual component
    float current_time;                // Current time for temporal decay (if apply_temporal_decay=true)
};

// Initialize Navigation paradigm parameters with defaults
void llama_aurora_navigation_params_init(struct llama_aurora_navigation_params* params);

// Recursive drill-down search using velocity-based gating
// This implements the Navigation paradigm algorithm:
// 1. Start at root (Level 0)
// 2. For each node:
//    a. Compute dual quaternion inner product ⟨q, k⟩dq
//    b. Extract primal similarity (relevance check)
//    c. Extract dual velocity ||zd|| (uncertainty signal)
//    d. If similarity > threshold AND velocity > tau: drill down
//    e. Else: use summary (stop recursion)
//
// Inputs:
//   - ctx: GGML context
//   - query_dq: Query dual quaternion [8] (primal + dual)
//   - memory_banks: Aurora memory banks with fractal structure
//   - nav_params: Navigation paradigm parameters
//
// Returns:
//   - Selected embeddings [K, 8] where K <= max_nodes_per_level
struct ggml_tensor* llama_aurora_fractal_drill_down(
    struct ggml_context* ctx,
    struct ggml_tensor* query_dq,  // [8] dual quaternion
    aurora_memory_banks_t* memory_banks,
    const struct llama_aurora_navigation_params* nav_params
);

// Hybrid Fractal Attention: Combines local window + global fractal context
// This implements the Navigation paradigm attention:
// - Clocal: Last W tokens with standard dense attention (O(W·Q))
// - Cglobal: Fractal memory via velocity-based drill-down (O(K·log N))
//
// Inputs:
//   - ctx: GGML context
//   - q: Query tensor [n_head, n_tokens, head_dim]
//   - k: Key tensor [n_head, n_kv, head_dim]
//   - v: Value tensor [n_head, n_kv, head_dim]
//   - window_size: Local window size W
//   - memory_banks: Aurora memory banks
//   - nav_params: Navigation paradigm parameters
//
// Returns:
//   - Attention output tensor [n_head, n_tokens, head_dim]
struct ggml_tensor* llama_aurora_hybrid_fractal_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    int32_t window_size,
    aurora_memory_banks_t* memory_banks,
    const struct llama_aurora_navigation_params* nav_params
);

// Extract dual velocity magnitude from dual quaternion inner product
// This is used for the "Policy Gate" decision (||zd|| > tau)
float llama_aurora_extract_dual_velocity(
    struct ggml_tensor* score_dq  // Dual quaternion inner product result [8]
);

// Extract primal similarity from dual quaternion inner product
// This is used for relevance check (similarity > threshold)
float llama_aurora_extract_primal_similarity(
    struct ggml_tensor* score_dq  // Dual quaternion inner product result [8]
);

#ifdef __cplusplus
}
#endif
