#pragma once

#include "ggml.h"
#include "ggml-dual-quaternion.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dual-quaternion attention
// Q/K/V: [batch, seq, 8] (dual quaternions)
// Attention scores: semantic (rotation) + contextual (translation)

struct ggml_tensor* ggml_dq_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,  // [batch, seq_q, 8]
    struct ggml_tensor* k,  // [batch, seq_k, 8]
    struct ggml_tensor* v,  // [batch, seq_k, 8]
    struct ggml_tensor* mask,  // [batch, seq_q, seq_k] or NULL
    float semantic_weight,   // Weight for rotation similarity (default: 0.7)
    float contextual_weight  // Weight for translation similarity (default: 0.3)
);

#ifdef __cplusplus
}
#endif
