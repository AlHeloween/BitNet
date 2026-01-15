#include "ggml-aurora-memory.h"
#include "aurora_memory_bank.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Custom tensor type for Aurora memory operations
// We'll use a custom approach since we can't modify the GGML enum
// Store parameters in tensor->op_params (if available) or use a custom backend

// For now, we'll use a simpler approach: store parameters in a global registry
// In production, this should use a proper parameter storage mechanism
struct aurora_memory_registry_entry {
    struct ggml_tensor* tensor;
    aurora_memory_banks_t* memory_banks;
    int k_read;
    int candidate_slots;
    float dual_weight;  // For dual-complex queries
    bool is_dual_complex;  // Whether this is a dual-complex query
};

static aurora_memory_registry_entry* g_aurora_registry = NULL;
static int g_aurora_registry_size = 0;
static int g_aurora_registry_capacity = 0;

static void register_aurora_tensor(
    struct ggml_tensor* tensor,
    aurora_memory_banks_t* memory_banks,
    int k_read,
    int candidate_slots,
    float dual_weight,
    bool is_dual_complex
) {
    // Grow registry if needed
    if (g_aurora_registry_size >= g_aurora_registry_capacity) {
        int new_capacity = (g_aurora_registry_capacity == 0) ? 16 : g_aurora_registry_capacity * 2;
        g_aurora_registry = (aurora_memory_registry_entry*)realloc(
            g_aurora_registry,
            new_capacity * sizeof(aurora_memory_registry_entry)
        );
        g_aurora_registry_capacity = new_capacity;
    }
    
    g_aurora_registry[g_aurora_registry_size].tensor = tensor;
    g_aurora_registry[g_aurora_registry_size].memory_banks = memory_banks;
    g_aurora_registry[g_aurora_registry_size].k_read = k_read;
    g_aurora_registry[g_aurora_registry_size].candidate_slots = candidate_slots;
    g_aurora_registry[g_aurora_registry_size].dual_weight = dual_weight;
    g_aurora_registry[g_aurora_registry_size].is_dual_complex = is_dual_complex;
    g_aurora_registry_size++;
}

static aurora_memory_registry_entry* find_aurora_tensor(const struct ggml_tensor* tensor) {
    for (int i = 0; i < g_aurora_registry_size; i++) {
        if (g_aurora_registry[i].tensor == tensor) {
            return &g_aurora_registry[i];
        }
    }
    return NULL;
}

struct ggml_tensor * ggml_aurora_memory_read(
    struct ggml_context * ctx,
    struct ggml_tensor  * query_embedding,
    aurora_memory_banks_t* memory_banks,
    int k_read,
    int candidate_slots
) {
    // Validate inputs
    if (!ctx || !query_embedding || !memory_banks) {
        return NULL;
    }
    
    if (ggml_n_dims(query_embedding) != 1) {
        // Query embedding should be 1D: [dim]
        return NULL;
    }
    
    int dim = (int)query_embedding->ne[0];
    
    // Create output tensor: [k_read, dim]
    struct ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, k_read);
    
    if (!result) {
        return NULL;
    }
    
    // Store parameters in registry
    register_aurora_tensor(result, memory_banks, k_read, candidate_slots, 0.0f, false);
    
    // Set tensor name for debugging
    ggml_set_name(result, "aurora_memory_read");
    
    // Mark this tensor as requiring custom computation
    // We'll use a custom op type or handle it in the compute function
    // For now, we'll use GGML_OP_NONE and handle it specially
    
    return result;
}

void ggml_aurora_memory_read_impl(
    const struct ggml_tensor * query_embedding,
    struct ggml_tensor * retrieved_embeddings,
    aurora_memory_banks_t* memory_banks,
    int k_read,
    int candidate_slots
) {
    if (!query_embedding || !retrieved_embeddings || !memory_banks) {
        return;
    }
    
    // Get query embedding data
    float* query_data = (float*)ggml_get_data_f32((struct ggml_tensor*)query_embedding);
    if (!query_data) {
        return;
    }
    
    int dim = (int)query_embedding->ne[0];
    
    // Convert query embedding to text for memory lookup
    // In practice, we'd want to use the embedding directly, but the current
    // memory bank API uses text queries. For now, we'll create a dummy text.
    // TODO: Add embedding-based query to memory bank API
    char query_text[256];
    snprintf(query_text, sizeof(query_text), "embedding_%p", (void*)query_data);
    
    // Query memory banks
    aurora_memory_entry_t** entries = (aurora_memory_entry_t**)malloc(k_read * sizeof(aurora_memory_entry_t*));
    int count = aurora_memory_banks_query(
        memory_banks,
        query_text,
        candidate_slots,
        k_read,
        entries
    );
    
    // Get output data
    float* output_data = (float*)ggml_get_data_f32(retrieved_embeddings);
    if (!output_data) {
        free(entries);
        return;
    }
    
    // Copy retrieved embeddings to output tensor
    // Output shape: [k_read, dim]
    for (int i = 0; i < k_read; i++) {
        if (i < count && entries[i]) {
            // Copy embedding
            memcpy(&output_data[i * dim], entries[i]->embedding, dim * sizeof(float));
        } else {
            // Zero-pad if not enough entries
            memset(&output_data[i * dim], 0, dim * sizeof(float));
        }
    }
    
    free(entries);
}

struct ggml_tensor * ggml_aurora_memory_read_dual_complex(
    struct ggml_context * ctx,
    struct ggml_tensor  * query_primal,
    struct ggml_tensor  * query_dual,
    aurora_memory_banks_t* memory_banks,
    int k_read,
    int candidate_slots,
    float dual_weight
) {
    // Validate inputs
    if (!ctx || !query_primal || !memory_banks) {
        return NULL;
    }
    
    if (ggml_n_dims(query_primal) != 1) {
        return NULL;
    }
    
    int dim = (int)query_primal->ne[0];
    
    // Create output tensor: [k_read, dim, 2] where [..., 0] = primal, [..., 1] = dual
    int64_t ne[3] = {dim, k_read, 2};
    struct ggml_tensor* result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2]);
    
    if (!result) {
        return NULL;
    }
    
    // Store parameters in registry
    register_aurora_tensor(result, memory_banks, k_read, candidate_slots, dual_weight, true);
    
    // Set tensor name for debugging
    ggml_set_name(result, "aurora_memory_read_dual_complex");
    
    return result;
}

void ggml_aurora_memory_read_dual_complex_impl(
    const struct ggml_tensor * query_primal,
    const struct ggml_tensor * query_dual,
    struct ggml_tensor * retrieved_embeddings,
    aurora_memory_banks_t* memory_banks,
    int k_read,
    int candidate_slots,
    float dual_weight
) {
    if (!query_primal || !retrieved_embeddings || !memory_banks) {
        return;
    }
    
    // Get query embedding data
    float* query_primal_data = (float*)ggml_get_data_f32((struct ggml_tensor*)query_primal);
    if (!query_primal_data) {
        return;
    }
    
    float* query_dual_data = NULL;
    if (query_dual) {
        query_dual_data = (float*)ggml_get_data_f32((struct ggml_tensor*)query_dual);
    }
    
    int dim = (int)query_primal->ne[0];
    
    // Query memory banks using dual-complex similarity
    aurora_memory_entry_t** entries = (aurora_memory_entry_t**)malloc(k_read * sizeof(aurora_memory_entry_t*));
    int count = aurora_memory_bank_query_dual_complex(
        memory_banks->verified,  // Query verified bank
        query_primal_data,
        query_dual_data,
        candidate_slots,
        k_read,
        dual_weight,
        entries
    );
    
    // Get output data
    // Output shape: [k_read, dim, 2] where [i, j, 0] = primal, [i, j, 1] = dual
    float* output_data = (float*)ggml_get_data_f32(retrieved_embeddings);
    if (!output_data) {
        free(entries);
        return;
    }
    
    // Copy retrieved embeddings to output tensor
    for (int i = 0; i < k_read; i++) {
        if (i < count && entries[i]) {
            // Copy primal embedding: [i, :, 0]
            memcpy(&output_data[i * dim * 2], entries[i]->embedding, dim * sizeof(float));
            
            // Copy dual embedding: [i, :, 1]
            if (entries[i]->is_dual_complex && entries[i]->embedding_dual) {
                memcpy(&output_data[i * dim * 2 + dim], entries[i]->embedding_dual, dim * sizeof(float));
            } else {
                // Zero-pad dual if not available
                memset(&output_data[i * dim * 2 + dim], 0, dim * sizeof(float));
            }
        } else {
            // Zero-pad if not enough entries
            memset(&output_data[i * dim * 2], 0, dim * 2 * sizeof(float));
        }
    }
    
    free(entries);
}

struct ggml_aurora_memory_params * ggml_get_aurora_memory_params(
    const struct ggml_tensor * tensor
) {
    aurora_memory_registry_entry* entry = find_aurora_tensor(tensor);
    if (!entry) {
        return NULL;
    }
    
    static struct ggml_aurora_memory_params params;
    params.memory_banks = entry->memory_banks;
    params.k_read = entry->k_read;
    params.candidate_slots = entry->candidate_slots;
    
    return &params;
}

// Cleanup function (call when context is freed)
void ggml_aurora_memory_cleanup(void) {
    if (g_aurora_registry) {
        free(g_aurora_registry);
        g_aurora_registry = NULL;
        g_aurora_registry_size = 0;
        g_aurora_registry_capacity = 0;
    }
}
