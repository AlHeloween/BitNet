#include "llama-aurora-integration.h"
#include "llama-aurora-attention.h"
#include "llama-aurora-navigation.h"
#include "ggml-aurora-memory.h"
#include "aurora_memory_bank.h"
#include "aurora_artifacts.h"

#include <string.h>
#include <stdlib.h>

void llama_aurora_context_params_init(struct llama_aurora_context_params* params) {
    if (!params) return;
    params->enable_aurora_memory = false;
    params->aurora_window_size = 2048;
    params->aurora_k_read = 8;
    params->aurora_candidate_slots = 4;
    params->memory_banks = NULL;
    params->enable_memory_write = false;
    params->default_information_mark = AURORA_INFO_UNKNOWN;
    params->use_dual_complex = false;
    
    // Navigation paradigm defaults
    params->enable_navigation_paradigm = false;
    params->nav_tau = 0.1f;
    params->nav_max_depth = 7;
    params->nav_max_nodes_per_level = 4;
    params->nav_apply_temporal_decay = false;
    params->nav_current_time = 0.0f;
    
    params->run_metadata = NULL;
    params->seed = -1;
    params->enable_artifact_tracking = false;
    
    // GPU metrics defaults
    params->enable_gpu_metrics = false;
    params->gpu_metrics_ptr = NULL;
    
    // Set CUDA seed if provided
    if (params->seed >= 0) {
        aurora_cuda_set_seed(params->seed);
        aurora_cuda_set_deterministic(true);
    }
}

void llama_aurora_gpu_metrics_init(struct llama_aurora_gpu_metrics* metrics) {
    if (!metrics) return;
    memset(metrics, 0, sizeof(struct llama_aurora_gpu_metrics));
}

void llama_aurora_gpu_metrics_reset(struct llama_aurora_gpu_metrics* metrics) {
    if (!metrics) return;
    llama_aurora_gpu_metrics_init(metrics);
}

void llama_aurora_get_gpu_metrics(
    const struct llama_aurora_context_params* params,
    struct llama_aurora_gpu_metrics* metrics
) {
    if (!params || !metrics) return;
    
    if (params->enable_gpu_metrics && params->gpu_metrics_ptr) {
        struct llama_aurora_gpu_metrics* src = 
            (struct llama_aurora_gpu_metrics*)params->gpu_metrics_ptr;
        memcpy(metrics, src, sizeof(struct llama_aurora_gpu_metrics));
    } else {
        llama_aurora_gpu_metrics_init(metrics);
    }
}

struct ggml_tensor* llama_aurora_pre_attention_hook(
    struct ggml_context* ctx,
    struct ggml_tensor* hidden_state,
    int32_t token_idx,
    const struct llama_aurora_context_params* params
) {
    if (!ctx || !hidden_state || !params || !params->enable_aurora_memory || !params->memory_banks) {
        return NULL;
    }
    
    // Extract query embedding
    struct ggml_tensor* query_emb = llama_aurora_extract_query_embedding(ctx, hidden_state, token_idx);
    if (!query_emb) {
        return NULL;
    }
    
    // Read from memory banks
    struct ggml_tensor* memory_embeddings = ggml_aurora_memory_read(
        ctx,
        query_emb,
        params->memory_banks,
        params->aurora_k_read,
        params->aurora_candidate_slots
    );
    
    return memory_embeddings;
}

struct ggml_tensor* llama_aurora_pre_attention_hook_dual_complex(
    struct ggml_context* ctx,
    struct ggml_tensor* hidden_state_primal,
    struct ggml_tensor* hidden_state_dual,
    int32_t token_idx,
    const struct llama_aurora_context_params* params
) {
    if (!ctx || !hidden_state_primal || !params || !params->enable_aurora_memory || !params->memory_banks) {
        return NULL;
    }
    
    // Extract primal query embedding
    struct ggml_tensor* query_emb_primal = llama_aurora_extract_query_embedding(ctx, hidden_state_primal, token_idx);
    if (!query_emb_primal) {
        return NULL;
    }
    
    // Extract dual query embedding if available
    struct ggml_tensor* query_emb_dual = NULL;
    if (hidden_state_dual) {
        query_emb_dual = llama_aurora_extract_query_embedding(ctx, hidden_state_dual, token_idx);
    }
    
    // Read from memory banks using dual-complex similarity
    float dual_weight = 0.1f;  // Default weight for dual component
    struct ggml_tensor* memory_embeddings = ggml_aurora_memory_read_dual_complex(
        ctx,
        query_emb_primal,
        query_emb_dual,
        params->memory_banks,
        params->aurora_k_read,
        params->aurora_candidate_slots,
        dual_weight
    );
    
    return memory_embeddings;
}

struct ggml_tensor* llama_aurora_attention_hook(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    const struct llama_aurora_context_params* params
) {
    if (!ctx || !q || !k || !v || !params || !params->enable_aurora_memory) {
        return NULL;  // Use default attention
    }
    
    // Check if Navigation paradigm is enabled (Stage 4)
    if (params->enable_navigation_paradigm) {
        // Log that Navigation paradigm is active (for debugging/verification)
        // Note: This will log once per attention layer, which may be verbose
        // Consider adding a static flag to log only once if needed
        
        // Build Navigation paradigm parameters
        struct llama_aurora_navigation_params nav_params;
        llama_aurora_navigation_params_init(&nav_params);
        nav_params.enable_navigation_paradigm = true;
        nav_params.tau = params->nav_tau;
        nav_params.max_depth = params->nav_max_depth;
        nav_params.max_nodes_per_level = params->nav_max_nodes_per_level;
        nav_params.apply_temporal_decay = params->nav_apply_temporal_decay;
        nav_params.current_time = params->nav_current_time;
        
        // Use hybrid fractal attention
        struct ggml_tensor* output = llama_aurora_hybrid_fractal_attention(
            ctx,
            q,
            k,
            v,
            params->aurora_window_size,
            params->memory_banks,
            &nav_params
        );
        
        if (output) {
            return output;
        }
        // Fall through to standard attention if hybrid fails
    }
    
    // Check if dual-complex mode is enabled
    if (params->use_dual_complex) {
        // For dual-complex mode, we need dual components of Q/K/V
        // In the current implementation, hidden states are real, so we'll:
        // 1. Use real tensors as primal
        // 2. Initialize dual to zero (or extract from a separate dual state if available)
        // 3. Call dual-complex attention
        
        // TODO: Extract dual components from hidden state if available
        // For now, we'll use NULL for dual components (treated as zero)
        struct ggml_tensor* q_dual = NULL;
        struct ggml_tensor* k_dual = NULL;
        struct ggml_tensor* v_dual = NULL;
        
        // Get memory embeddings from pre-attention hook
        // In practice, this would be called before attention
        struct ggml_tensor* memory_primal = NULL;
        struct ggml_tensor* memory_dual = NULL;
        
        // For now, use standard bounded attention
        // Full dual-complex attention will be implemented when dual states are available
        struct ggml_tensor* output = llama_aurora_bounded_attention(
            ctx,
            q,
            k,
            v,
            params->aurora_window_size,
            params->memory_banks,
            params->aurora_k_read,
            params->aurora_candidate_slots
        );
        
        return output;
    } else {
        // Standard (real) bounded attention
        struct ggml_tensor* output = llama_aurora_bounded_attention(
            ctx,
            q,
            k,
            v,
            params->aurora_window_size,
            params->memory_banks,
            params->aurora_k_read,
            params->aurora_candidate_slots
        );
        
        return output;
    }
}

int llama_aurora_post_forward_hook(
    struct llama_context* ctx,
    const struct ggml_tensor* hidden_state,
    int32_t token_idx,
    const char* generated_text,
    const struct llama_aurora_context_params* params
) {
    if (!ctx || !params) {
        return 0;  // Not an error, just disabled
    }
    
    // Track artifact metadata if enabled
    if (params->enable_artifact_tracking && params->run_metadata) {
        // Artifact tracking is handled externally via run_metadata
        // This hook can be used to log token-level events if needed
    }
    
    // Write to memory banks if enabled
    if (params->enable_memory_write && params->memory_banks) {
        if (!generated_text || strlen(generated_text) == 0) {
            return 0;  // No text to write
        }
        
        // Write to memory banks
        bool success = aurora_memory_banks_add(
            params->memory_banks,
            generated_text,
            params->default_information_mark,
            NULL,  // Use same text for embedding
            NULL   // Compute MD5 automatically
        );
        
        return success ? 0 : -1;
    }
    
    return 0;
}

void llama_aurora_apply_kv_cache_windowing(
    struct llama_context* ctx,
    const struct llama_aurora_context_params* params
) {
    if (!ctx || !params || !params->enable_aurora_memory) {
        return;
    }
    
    // Apply windowing to KV cache
    llama_aurora_window_kv_cache(&ctx->kv_self, params->aurora_window_size);
}
