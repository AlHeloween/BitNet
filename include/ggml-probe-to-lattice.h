#pragma once

#include "ggml.h"
#include "ggml-ffe-quantization.h"
#include "ggml-lattice-addressing.h"
#include "ggml-lattice-storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Probe → FFE → Lattice pipeline

// Complete pipeline: Input → Probe → FFE → Lattice addresses
// Returns number of addresses stored
int ggml_probe_to_lattice_pipeline(
    struct ggml_context* ctx,
    struct ggml_tensor* input_embeddings,  // [batch, seq, hidden_dim]
    struct ggml_tensor* probe_weights,     // Probe encoder weights (optional)
    struct ggml_tensor* sierpinski_centroids,  // [n_levels, n_per_level, 8]
    struct lattice_memory_storage* storage,
    const char* text,  // Optional text
    struct lattice_address* addresses_out,  // Output addresses
    int max_addresses
);

#ifdef __cplusplus
}
#endif
