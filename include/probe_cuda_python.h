#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// DQLinear forward pass
// Returns 0 on success, non-zero on error
int probe_cuda_dq_linear(
    const float* weight,      // [out_features, in_features, 8]
    const float* input,        // [batch, seq, in_features, 8]
    float* output,             // [batch, seq, out_features, 8]
    int batch_size,
    int seq_len,
    int in_features,
    int out_features
);

// DQAttention forward pass
// Returns 0 on success, non-zero on error
int probe_cuda_dq_attention_scores(
    const float* q,           // [batch, seq_q, 8]
    const float* k,           // [batch, seq_k, 8]
    float* scores,            // [batch, seq_q, seq_k]
    int batch_size,
    int seq_q,
    int seq_k,
    float semantic_weight,
    float contextual_weight
);

#ifdef __cplusplus
}
#endif
