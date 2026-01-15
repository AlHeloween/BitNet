#pragma once

#include "ggml.h"
#include "ggml-screw-theory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Temporal evolution on dual component via screw theory

// Evolve dual-complex tensor forward in time
// Input: z_primal [batch, seq, dim], z_dual [batch, seq, dim]
// Output: z_evolved_primal [batch, seq, dim], z_evolved_dual [batch, seq, dim]
struct ggml_tensor* ggml_temporal_evolve(
    struct ggml_context* ctx,
    struct ggml_tensor* z_primal,  // [batch, seq, dim]
    struct ggml_tensor* z_dual,     // [batch, seq, dim]
    struct ggml_tensor* dt,          // Time step (scalar or [batch, seq])
    struct ggml_tensor* screw_weights  // Optional learned weights for screw prediction
);

// Predict screw parameters from dual-complex state
void ggml_temporal_predict_screw(
    const float* z_primal,  // [dim]
    const float* z_dual,    // [dim]
    const float* weights,   // [6] (3 omega + 3 v) - learned weights
    struct screw_parameters* params_out
);

#ifdef __cplusplus
}
#endif
