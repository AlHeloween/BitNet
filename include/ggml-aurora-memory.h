#pragma once

#include "ggml.h"
#include "aurora_memory_bank.h"

#ifdef __cplusplus
extern "C" {
#endif

// GGML Aurora Memory Operation
// This operation reads from Aurora memory banks and returns retrieved embeddings

// Create a tensor that represents a memory read operation
// Input: query_embedding (shape: [dim]) - query embedding from hidden state
// Output: retrieved_embeddings (shape: [k_read, dim]) - top-k retrieved embeddings
// Parameters stored in tensor->op_params:
//   - memory_banks: pointer to aurora_memory_banks_t
//   - k_read: number of embeddings to retrieve
//   - candidate_slots: number of candidate slots to search
struct ggml_aurora_memory_params {
    aurora_memory_banks_t* memory_banks;
    int k_read;
    int candidate_slots;
};

// Create a memory read tensor in the compute graph
// This creates a tensor that will be populated during graph execution
GGML_API struct ggml_tensor * ggml_aurora_memory_read(
    struct ggml_context * ctx,
    struct ggml_tensor  * query_embedding,  // Input: [dim] query embedding
    aurora_memory_banks_t* memory_banks,    // Memory banks to query
    int k_read,                              // Number of embeddings to retrieve
    int candidate_slots                      // Number of candidate slots
);
// Dual-complex version: uses both primal and dual for similarity
// Returns: [k_read, dim, 2] where [..., 0] = primal, [..., 1] = dual
GGML_API struct ggml_tensor * ggml_aurora_memory_read_dual_complex(
    struct ggml_context * ctx,
    struct ggml_tensor  * query_primal,     // Input: [dim] primal query embedding
    struct ggml_tensor  * query_dual,       // Input: [dim] dual query embedding (can be NULL)
    aurora_memory_banks_t* memory_banks,    // Memory banks to query
    int k_read,                              // Number of embeddings to retrieve
    int candidate_slots,                     // Number of candidate slots
    float dual_weight                        // Weight for dual component in similarity (default: 0.1)
);

// Execute memory read operation (called during graph computation)
// This function is called by the GGML compute graph executor
GGML_API void ggml_aurora_memory_read_impl(
    const struct ggml_tensor * query_embedding,
    struct ggml_tensor * retrieved_embeddings,
    aurora_memory_banks_t* memory_banks,
    int k_read,
    int candidate_slots
);

// Get parameters from tensor (if it's a memory read operation)
GGML_API struct ggml_aurora_memory_params * ggml_get_aurora_memory_params(
    const struct ggml_tensor * tensor
);

#ifdef __cplusplus
}
#endif
