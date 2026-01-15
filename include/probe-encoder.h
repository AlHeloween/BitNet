#pragma once

#include "ggml.h"
#include "ggml-dq-linear.h"
#include "ggml-dq-attention.h"

#ifdef __cplusplus
extern "C" {
#endif

// Probe encoder parameters
struct probe_encoder_params {
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    float dropout;
    bool enable_probe;
};

// Probe encoder forward pass
// Input: token embeddings [batch, seq, hidden_dim]
// Output: dual quaternion latents [batch, seq, hidden_size, 8]
struct ggml_tensor* probe_encode(
    struct ggml_context* ctx,
    struct ggml_tensor* input_embeddings,  // [batch, seq, hidden_dim]
    struct probe_encoder_params* params,
    struct ggml_tensor* probe_weights  // Encoder weights (loaded from file, can be NULL for now)
);

#ifdef __cplusplus
}
#endif
