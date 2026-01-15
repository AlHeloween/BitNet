#pragma once

#include "ggml.h"
#include "ggml-ffe-quantization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sierpinski lattice addressing for discrete memory addresses

struct lattice_address {
    uint8_t level;      // 0-7 (3 bits)
    uint16_t index;     // 0-511 (9 bits)
    uint8_t sub_index;  // 0-3 (2 bits)
};

// Convert lattice address to FFE address
struct ffe_address lattice_to_ffe(const struct lattice_address* addr);

// Convert FFE address to lattice address
struct lattice_address ffe_to_lattice(const struct ffe_address* ffe_addr);

// Address to centroid mapping
struct ggml_tensor* ggml_lattice_address_to_centroid(
    struct ggml_context* ctx,
    const struct lattice_address* addr,
    struct ggml_tensor* sierpinski_centroids  // [n_levels, n_per_level, 8]
);

// Centroid to address mapping (find nearest)
struct lattice_address ggml_lattice_centroid_to_address(
    const float* centroid,  // [8]
    struct ggml_tensor* sierpinski_centroids  // [n_levels, n_per_level, 8]
);

// Get parent address (one level up)
struct lattice_address ggml_lattice_get_parent(
    const struct lattice_address* addr
);

// Get child addresses (one level down)
void ggml_lattice_get_children(
    const struct lattice_address* addr,
    int max_level,
    struct lattice_address* children_out,
    int* num_children_out
);

// Get sibling addresses
void ggml_lattice_get_siblings(
    const struct lattice_address* addr,
    struct lattice_address* siblings_out,
    int* num_siblings_out
);

// Get nearby addresses within radius
void ggml_lattice_get_nearby(
    const struct lattice_address* addr,
    int radius,
    struct lattice_address* nearby_out,
    int* num_nearby_out,
    int max_nearby
);

#ifdef __cplusplus
}
#endif
