#include "ggml-ffe-quantization.h"
#include "ggml.h"
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"

#include <cmath>
#include <cstring>

using namespace DynaMap::math;

uint16_t ffe_address_encode(const struct ffe_address* addr) {
    return (uint16_t)((addr->level << 11) | (addr->index << 2) | addr->sub_index);
}

struct ffe_address ffe_address_decode(uint16_t bits) {
    struct ffe_address addr;
    addr.level = (uint8_t)((bits >> 11) & 0x7);
    addr.index = (uint16_t)((bits >> 2) & 0x1FF);
    addr.sub_index = (uint8_t)(bits & 0x3);
    return addr;
}

struct ggml_tensor* ggml_ffe_quantize(
    struct ggml_context* ctx,
    struct ggml_tensor* dual_quaternion,
    struct ggml_tensor* sierpinski_centroids
) {
    if (!ctx || !dual_quaternion || !sierpinski_centroids) {
        return NULL;
    }
    
    // Validate inputs
    GGML_ASSERT(sierpinski_centroids->n_dims == 3);
    GGML_ASSERT(sierpinski_centroids->ne[2] == 8);  // Last dim must be 8 (dual quaternion)
    
    int n_levels = (int)sierpinski_centroids->ne[0];
    int n_per_level = (int)sierpinski_centroids->ne[1];
    
    // Handle different input shapes
    bool is_scalar = false;
    int batch_size = 1;
    int seq_len = 1;
    
    if (dual_quaternion->n_dims == 1) {
        // Single dual quaternion [8]
        is_scalar = true;
        GGML_ASSERT(dual_quaternion->ne[0] == 8);
    } else if (dual_quaternion->n_dims == 2) {
        // [seq, 8]
        seq_len = (int)dual_quaternion->ne[0];
        GGML_ASSERT(dual_quaternion->ne[1] == 8);
    } else if (dual_quaternion->n_dims == 3) {
        // [batch, seq, 8]
        batch_size = (int)dual_quaternion->ne[0];
        seq_len = (int)dual_quaternion->ne[1];
        GGML_ASSERT(dual_quaternion->ne[2] == 8);
    } else {
        return NULL;
    }
    
    // Create output tensor: [batch, seq] or scalar
    struct ggml_tensor* result = NULL;
    if (is_scalar) {
        result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    } else if (dual_quaternion->n_dims == 2) {
        result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    } else {
        int64_t out_ne[2] = {batch_size, seq_len};
        result = ggml_new_tensor(ctx, GGML_TYPE_I32, 2, out_ne);
    }
    
    const float* dq_data = (const float*)dual_quaternion->data;
    const float* centroids_data = (const float*)sierpinski_centroids->data;
    int32_t* out_data = (int32_t*)result->data;
    
    int total_elements = batch_size * seq_len;
    
    // For each dual quaternion, find nearest centroid
    for (int i = 0; i < total_elements; i++) {
        // Load dual quaternion
        const float* dq = &dq_data[i * 8];
        Quaternion dq_r(dq[0], dq[1], dq[2], dq[3]);
        Quaternion dq_d(dq[4], dq[5], dq[6], dq[7]);
        dualQuat dq_obj(dq_r, dq_d);
        
        // Find nearest centroid across all levels
        int best_level = 0;
        int best_index = 0;
        float best_distance = 1e9f;
        
        for (int level = 0; level < n_levels; level++) {
            for (int idx = 0; idx < n_per_level; idx++) {
                // Load centroid
                const float* centroid = &centroids_data[(level * n_per_level + idx) * 8];
                Quaternion c_r(centroid[0], centroid[1], centroid[2], centroid[3]);
                Quaternion c_d(centroid[4], centroid[5], centroid[6], centroid[7]);
                dualQuat c_obj(c_r, c_d);
                
                // Compute distance (L2 norm of difference)
                float dist = 0.0f;
                for (int j = 0; j < 8; j++) {
                    float diff = dq[j] - centroid[j];
                    dist += diff * diff;
                }
                dist = sqrtf(dist);
                
                if (dist < best_distance) {
                    best_distance = dist;
                    best_level = level;
                    best_index = idx;
                }
            }
        }
        
        // Create FFE address
        struct ffe_address addr;
        addr.level = (uint8_t)best_level;
        addr.index = (uint16_t)best_index;
        addr.sub_index = 0;  // Default sub-index
        
        // Encode to 14-bit integer
        uint16_t addr_bits = ffe_address_encode(&addr);
        out_data[i] = (int32_t)addr_bits;
    }
    
    return result;
}

struct ggml_tensor* ggml_ffe_dequantize(
    struct ggml_context* ctx,
    struct ggml_tensor* addresses,
    struct ggml_tensor* sierpinski_centroids
) {
    if (!ctx || !addresses || !sierpinski_centroids) {
        return NULL;
    }
    
    GGML_ASSERT(sierpinski_centroids->n_dims == 3);
    GGML_ASSERT(sierpinski_centroids->ne[2] == 8);
    
    // Handle different input shapes
    bool is_scalar = false;
    int batch_size = 1;
    int seq_len = 1;
    
    if (addresses->n_dims == 0 || (addresses->n_dims == 1 && addresses->ne[0] == 1)) {
        is_scalar = true;
    } else if (addresses->n_dims == 1) {
        seq_len = (int)addresses->ne[0];
    } else if (addresses->n_dims == 2) {
        batch_size = (int)addresses->ne[0];
        seq_len = (int)addresses->ne[1];
    } else {
        return NULL;
    }
    
    // Create output tensor: [batch, seq, 8] or [8]
    struct ggml_tensor* result = NULL;
    if (is_scalar) {
        result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    } else if (addresses->n_dims == 1) {
        int64_t out_ne[2] = {seq_len, 8};
        result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, out_ne);
    } else {
        int64_t out_ne[3] = {batch_size, seq_len, 8};
        result = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, out_ne);
    }
    
    const int32_t* addr_data = (const int32_t*)addresses->data;
    const float* centroids_data = (const float*)sierpinski_centroids->data;
    float* out_data = (float*)result->data;
    
    int total_elements = batch_size * seq_len;
    
    // For each address, get corresponding centroid
    for (int i = 0; i < total_elements; i++) {
        uint16_t addr_bits = (uint16_t)addr_data[i];
        struct ffe_address addr = ffe_address_decode(addr_bits);
        
        // Get centroid
        const float* centroid = &centroids_data[(addr.level * (int)sierpinski_centroids->ne[1] + addr.index) * 8];
        
        // Copy to output
        float* out = &out_data[i * 8];
        for (int j = 0; j < 8; j++) {
            out[j] = centroid[j];
        }
    }
    
    return result;
}

struct ggml_tensor* ggml_ffe_get_centroid(
    struct ggml_context* ctx,
    struct ggml_tensor* address,
    struct ggml_tensor* sierpinski_centroids
) {
    if (!ctx || !address || !sierpinski_centroids) {
        return NULL;
    }
    
    // Address should be scalar
    GGML_ASSERT(address->n_dims == 0 || (address->n_dims == 1 && address->ne[0] == 1));
    
    int32_t addr_bits = ((const int32_t*)address->data)[0];
    struct ffe_address addr = ffe_address_decode((uint16_t)addr_bits);
    
    // Create output tensor [8]
    struct ggml_tensor* result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    
    const float* centroids_data = (const float*)sierpinski_centroids->data;
    float* out_data = (float*)result->data;
    
    // Get centroid
    const float* centroid = &centroids_data[(addr.level * (int)sierpinski_centroids->ne[1] + addr.index) * 8];
    
    // Copy to output
    for (int j = 0; j < 8; j++) {
        out_data[j] = centroid[j];
    }
    
    return result;
}
