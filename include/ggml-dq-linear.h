#pragma once

#include "ggml.h"
#include "ggml-dual-quaternion.h"

#ifdef __cplusplus
extern "C" {
#endif

// DQLinear: Hamilton product-based linear layer
// Forward: y = W * x (Hamilton product)
// Weight: [out_features, in_features, 8]
// Input: [batch, seq, in_features, 8]
// Output: [batch, seq, out_features, 8]

struct ggml_tensor* ggml_dq_linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // [batch, seq, in_features, 8]
    struct ggml_tensor* weight,   // [out_features, in_features, 8]
    struct ggml_tensor* bias      // [out_features, 8] or NULL
);

#ifdef __cplusplus
}
#endif
