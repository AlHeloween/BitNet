#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// FFE (Fractal Feature Encoding) 3-9-2 quantization
// Format: 3 bits level + 9 bits index + 2 bits sub_index = 14 bits total

// FFE address structure
struct ffe_address {
    uint8_t level;      // 0-7 (3 bits)
    uint16_t index;     // 0-511 (9 bits)
    uint8_t sub_index;  // 0-3 (2 bits)
};

// Encode FFE address to 14-bit integer
uint16_t ffe_address_encode(const struct ffe_address* addr);

// Decode 14-bit integer to FFE address
struct ffe_address ffe_address_decode(uint16_t bits);

// Quantize dual quaternion to FFE address
// Input: dual_quaternion [batch, seq, 8] or [8]
// Output: addresses [batch, seq] or scalar (14-bit integers)
struct ggml_tensor* ggml_ffe_quantize(
    struct ggml_context* ctx,
    struct ggml_tensor* dual_quaternion,  // [batch, seq, 8] or [8]
    struct ggml_tensor* sierpinski_centroids  // [n_levels, n_per_level, 8]
);

// Dequantize FFE addresses back to dual quaternions
// Input: addresses [batch, seq] or scalar (14-bit integers)
// Output: dual_quaternions [batch, seq, 8] or [8]
struct ggml_tensor* ggml_ffe_dequantize(
    struct ggml_context* ctx,
    struct ggml_tensor* addresses,  // [batch, seq] or scalar
    struct ggml_tensor* sierpinski_centroids  // [n_levels, n_per_level, 8]
);

// Get centroid for FFE address
struct ggml_tensor* ggml_ffe_get_centroid(
    struct ggml_context* ctx,
    struct ggml_tensor* address,  // Scalar (14-bit integer)
    struct ggml_tensor* sierpinski_centroids  // [n_levels, n_per_level, 8]
);

#ifdef __cplusplus
}
#endif
