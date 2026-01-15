#pragma once

#include "ggml.h"
#include "ggml-lattice-storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transcender: Triplet synthesis and hierarchical compression

// Triplet synthesis: (A, B, C) → higher-order node
struct ggml_tensor* ggml_transcender_synthesize_triplet(
    struct ggml_context* ctx,
    struct ggml_tensor* entry_a,  // [8] dual quaternion
    struct ggml_tensor* entry_b,  // [8] dual quaternion
    struct ggml_tensor* entry_c,  // [8] dual quaternion
    const char* method  // "mean", "max", "dual_quaternion"
);

// Apply rewrite rule to promote entries
int ggml_transcender_apply_rewrite(
    struct lattice_memory_storage* storage,
    const struct lattice_address* addr_a,
    const struct lattice_address* addr_b,
    const struct lattice_address* addr_c,
    const struct lattice_address* target_addr,
    const char* synthesis_method
);

// Check if promotion should occur
bool ggml_transcender_should_promote(
    struct lattice_memory_storage* storage,
    int level,
    int threshold,
    const char* trigger_type  // "count", "similarity"
);

// Compute compression metrics
void ggml_transcender_compute_metrics(
    struct lattice_memory_storage* storage,
    int level,
    float* information_loss_out,
    float* compression_ratio_out
);

#ifdef __cplusplus
}
#endif
