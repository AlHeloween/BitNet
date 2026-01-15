#pragma once

#include "llama.h"
#include "llama-aurora-attention.h"
#include "aurora_memory_bank.h"
#include "aurora_artifacts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Aurora Integration for llama.cpp Forward Pass
// This provides hooks to integrate Aurora memory into llama_decode()

// Add Aurora memory configuration to llama_context_params
// This extends the existing context params with Aurora-specific settings
struct llama_aurora_context_params {
    bool enable_aurora_memory;           // Enable Aurora memory integration
    int32_t aurora_window_size;         // Window size w (default: 2048)
    int32_t aurora_k_read;              // Number of memory embeddings to retrieve (default: 8)
    int32_t aurora_candidate_slots;     // Number of candidate slots (default: 4)
    aurora_memory_banks_t* memory_banks; // Memory banks pointer (managed externally)
    
    // Memory write settings
    bool enable_memory_write;            // Enable writing to memory after generation
    aurora_information_mark_t default_information_mark; // Default information mark for writes
    
    // Dual-complex settings
    bool use_dual_complex;               // Enable dual-complex mode (default: false)
    
    // Navigation paradigm settings (Stage 4)
    bool enable_navigation_paradigm;     // Enable Navigation paradigm (default: false)
    float nav_tau;                       // Uncertainty threshold for drill-down (default: 0.1)
    int32_t nav_max_depth;               // Maximum fractal level (default: 7)
    int32_t nav_max_nodes_per_level;     // Maximum nodes per level (default: 4)
    bool nav_apply_temporal_decay;       // Apply temporal decay (default: false)
    float nav_current_time;              // Current time for temporal decay (default: 0.0)
    
    // Artifact tracking
    aurora_run_metadata_t* run_metadata;  // Run metadata for artifact tracking (optional)
    int32_t seed;                        // Random seed for determinism (-1 to skip)
    bool enable_artifact_tracking;       // Enable artifact tracking (default: false)
    
    // GPU metrics collection (Stage 6)
    bool enable_gpu_metrics;             // Enable GPU metrics collection (default: false)
    void* gpu_metrics_ptr;                // Pointer to GPU metrics structure (internal use)
};

// GPU metrics structure for C++ CUDA runs
struct llama_aurora_gpu_metrics {
    // Kernel timings (milliseconds)
    float dual_quat_ops_time;            // Dual quaternion operations
    float fractal_drill_down_time;       // Fractal drill-down
    float attention_time;                // Attention computation
    float memory_transfer_time;          // Host↔device transfers
    float memory_query_time;             // Memory bank query time
    
    // Total GPU time
    float total_gpu_time;                 // Total GPU time per forward pass
    
    // Memory usage (bytes)
    size_t gpu_memory_allocated;         // GPU memory allocated
    size_t gpu_memory_used;              // GPU memory actually used
    
    // Throughput
    float tokens_per_second;             // Tokens processed per second
    int32_t tokens_processed;           // Number of tokens processed
    
    // Operation counts
    int32_t dual_quat_ops_count;         // Number of dual quaternion operations
    int32_t fractal_queries_count;       // Number of fractal queries
    int32_t memory_reads_count;          // Number of memory reads
};

// Initialize Aurora context params with defaults
void llama_aurora_context_params_init(struct llama_aurora_context_params* params);

// Hook function to be called before attention computation
// This extracts query embeddings and retrieves memory
// Returns: memory embeddings tensor or NULL if disabled
struct ggml_tensor* llama_aurora_pre_attention_hook(
    struct ggml_context* ctx,
    struct ggml_tensor* hidden_state,
    int32_t token_idx,
    const struct llama_aurora_context_params* params
);
// Dual-complex version: extracts both primal and dual components
struct ggml_tensor* llama_aurora_pre_attention_hook_dual_complex(
    struct ggml_context* ctx,
    struct ggml_tensor* hidden_state_primal,
    struct ggml_tensor* hidden_state_dual,  // Dual component (can be NULL if not available)
    int32_t token_idx,
    const struct llama_aurora_context_params* params
);

// Hook function to modify attention computation
// This applies bounded window attention with memory augmentation
// Returns: modified attention output tensor
struct ggml_tensor* llama_aurora_attention_hook(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    const struct llama_aurora_context_params* params
);

// Hook function to be called after forward pass
// This writes generated content to memory banks
// Returns: 0 on success, negative on error
int llama_aurora_post_forward_hook(
    struct llama_context* ctx,
    const struct ggml_tensor* hidden_state,
    int32_t token_idx,
    const char* generated_text,  // Generated text for this token (optional)
    const struct llama_aurora_context_params* params
);

// Apply KV cache windowing based on Aurora params
void llama_aurora_apply_kv_cache_windowing(
    struct llama_context* ctx,
    const struct llama_aurora_context_params* params
);

#ifdef __cplusplus
}
#endif
