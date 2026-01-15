#include "ggml-lattice-addressing.h"
#include "ggml.h"
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"

#include <cmath>
#include <cstring>

using namespace DynaMap::math;

struct ffe_address lattice_to_ffe(const struct lattice_address* addr) {
    struct ffe_address ffe;
    ffe.level = addr->level;
    ffe.index = addr->index;
    ffe.sub_index = addr->sub_index;
    return ffe;
}

struct lattice_address ffe_to_lattice(const struct ffe_address* ffe_addr) {
    struct lattice_address addr;
    addr.level = ffe_addr->level;
    addr.index = ffe_addr->index;
    addr.sub_index = ffe_addr->sub_index;
    return addr;
}

struct ggml_tensor* ggml_lattice_address_to_centroid(
    struct ggml_context* ctx,
    const struct lattice_address* addr,
    struct ggml_tensor* sierpinski_centroids
) {
    if (!ctx || !addr || !sierpinski_centroids) {
        return NULL;
    }
    
    GGML_ASSERT(sierpinski_centroids->n_dims == 3);
    GGML_ASSERT(addr->level < sierpinski_centroids->ne[0]);
    GGML_ASSERT(addr->index < sierpinski_centroids->ne[1]);
    
    // Create output tensor [8]
    struct ggml_tensor* result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    
    const float* centroids_data = (const float*)sierpinski_centroids->data;
    float* out_data = (float*)result->data;
    
    // Get centroid
    int n_per_level = (int)sierpinski_centroids->ne[1];
    const float* centroid = &centroids_data[(addr->level * n_per_level + addr->index) * 8];
    
    // Copy to output
    for (int j = 0; j < 8; j++) {
        out_data[j] = centroid[j];
    }
    
    return result;
}

struct lattice_address ggml_lattice_centroid_to_address(
    const float* centroid,
    struct ggml_tensor* sierpinski_centroids
) {
    struct lattice_address result = {0, 0, 0};
    
    if (!centroid || !sierpinski_centroids) {
        return result;
    }
    
    GGML_ASSERT(sierpinski_centroids->n_dims == 3);
    
    int n_levels = (int)sierpinski_centroids->ne[0];
    int n_per_level = (int)sierpinski_centroids->ne[1];
    
    // Find nearest centroid
    float best_distance = 1e9f;
    
    for (int level = 0; level < n_levels; level++) {
        for (int idx = 0; idx < n_per_level; idx++) {
            const float* c = &((const float*)sierpinski_centroids->data)[(level * n_per_level + idx) * 8];
            
            // Compute L2 distance
            float dist = 0.0f;
            for (int j = 0; j < 8; j++) {
                float diff = centroid[j] - c[j];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
            
            if (dist < best_distance) {
                best_distance = dist;
                result.level = (uint8_t)level;
                result.index = (uint16_t)idx;
                result.sub_index = 0;
            }
        }
    }
    
    return result;
}

struct lattice_address ggml_lattice_get_parent(const struct lattice_address* addr) {
    struct lattice_address parent = {0, 0, 0};
    
    if (addr->level == 0) {
        // Root level has no parent
        return parent;
    }
    
    parent.level = addr->level - 1;
    parent.index = addr->index / 2;  // Binary branching
    parent.sub_index = 0;
    
    return parent;
}

void ggml_lattice_get_children(
    const struct lattice_address* addr,
    int max_level,
    struct lattice_address* children_out,
    int* num_children_out
) {
    *num_children_out = 0;
    
    if (addr->level >= max_level) {
        return;
    }
    
    int child_level = addr->level + 1;
    int base_index = addr->index * 2;
    
    // Binary branching: 2 children
    for (int i = 0; i < 2; i++) {
        int child_index = base_index + i;
        // Check bounds (assuming n_per_level = 512)
        if (child_index < 512) {
            children_out[*num_children_out].level = (uint8_t)child_level;
            children_out[*num_children_out].index = (uint16_t)child_index;
            children_out[*num_children_out].sub_index = 0;
            (*num_children_out)++;
        }
    }
}

void ggml_lattice_get_siblings(
    const struct lattice_address* addr,
    struct lattice_address* siblings_out,
    int* num_siblings_out
) {
    *num_siblings_out = 0;
    
    // Siblings are at same level, with nearby indices
    for (int offset = -1; offset <= 1; offset += 2) {
        int sibling_index = (int)addr->index + offset;
        if (sibling_index >= 0 && sibling_index < 512) {
            siblings_out[*num_siblings_out].level = addr->level;
            siblings_out[*num_siblings_out].index = (uint16_t)sibling_index;
            siblings_out[*num_siblings_out].sub_index = addr->sub_index;
            (*num_siblings_out)++;
        }
    }
}

void ggml_lattice_get_nearby(
    const struct lattice_address* addr,
    int radius,
    struct lattice_address* nearby_out,
    int* num_nearby_out,
    int max_nearby
) {
    *num_nearby_out = 0;
    
    // Search in same level
    for (int offset = -radius; offset <= radius && *num_nearby_out < max_nearby; offset++) {
        int nearby_index = (int)addr->index + offset;
        if (nearby_index >= 0 && nearby_index < 512) {
            nearby_out[*num_nearby_out].level = addr->level;
            nearby_out[*num_nearby_out].index = (uint16_t)nearby_index;
            nearby_out[*num_nearby_out].sub_index = addr->sub_index;
            (*num_nearby_out)++;
        }
    }
}
