#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dual-complex tensor operations for GGML
// Dual-complex: [..., 2] where [..., 0] = primal, [..., 1] = dual
// For embeddings: [batch, seq, dim, 2]

// Create dual-complex tensor from primal and dual components
struct ggml_tensor* ggml_dual_complex(
    struct ggml_context* ctx,
    struct ggml_tensor* primal,  // [..., dim]
    struct ggml_tensor* dual     // [..., dim]
);

// Extract primal component
struct ggml_tensor* ggml_dual_complex_primal(
    struct ggml_context* ctx,
    struct ggml_tensor* dc  // [..., dim, 2]
);

// Extract dual component
struct ggml_tensor* ggml_dual_complex_dual(
    struct ggml_context* ctx,
    struct ggml_tensor* dc  // [..., dim, 2]
);

// Dual-complex addition: dc1 + dc2
struct ggml_tensor* ggml_dual_complex_add(
    struct ggml_context* ctx,
    struct ggml_tensor* dc1,  // [..., dim, 2]
    struct ggml_tensor* dc2   // [..., dim, 2]
);

// Dual-complex multiplication: dc1 * dc2
// Formula: (a + εb) * (c + εd) = ac + ε(ad + bc)
struct ggml_tensor* ggml_dual_complex_mul(
    struct ggml_context* ctx,
    struct ggml_tensor* dc1,  // [..., dim, 2]
    struct ggml_tensor* dc2   // [..., dim, 2]
);

// Dual-complex conjugate
struct ggml_tensor* ggml_dual_complex_conjugate(
    struct ggml_context* ctx,
    struct ggml_tensor* dc  // [..., dim, 2]
);

// Dual-complex normalize
struct ggml_tensor* ggml_dual_complex_normalize(
    struct ggml_context* ctx,
    struct ggml_tensor* dc  // [..., dim, 2]
);

// Dual-complex softmax (dual-softmax)
// Primal: softmax(primal)
// Dual: softmax(primal) * (dual - mean(dual))
struct ggml_tensor* ggml_dual_complex_softmax(
    struct ggml_context* ctx,
    struct ggml_tensor* dc  // [..., dim, 2]
);

#ifdef __cplusplus
}
#endif
