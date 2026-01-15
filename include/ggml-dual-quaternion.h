#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dual quaternion tensor operations for GGML
// Dual quaternion: [..., 8] where [..., 0:4] = rotation, [..., 4:8] = translation
// Format: [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]

// Hamilton product for dual quaternions
// q1 * q2 where q1, q2 are [..., 8]
struct ggml_tensor* ggml_dual_quaternion_hamilton_product(
    struct ggml_context* ctx,
    struct ggml_tensor* q1,  // [..., 8]
    struct ggml_tensor* q2   // [..., 8]
);

// Dual quaternion conjugate
struct ggml_tensor* ggml_dual_quaternion_conjugate(
    struct ggml_context* ctx,
    struct ggml_tensor* dq  // [..., 8]
);

// Normalize dual quaternion (unit dual quaternion)
struct ggml_tensor* ggml_dual_quaternion_normalize(
    struct ggml_context* ctx,
    struct ggml_tensor* dq  // [..., 8]
);

// Dual quaternion inverse
struct ggml_tensor* ggml_dual_quaternion_inverse(
    struct ggml_context* ctx,
    struct ggml_tensor* dq  // [..., 8]
);

// Extract rotation quaternion [..., 4]
struct ggml_tensor* ggml_dual_quaternion_rotation(
    struct ggml_context* ctx,
    struct ggml_tensor* dq  // [..., 8]
);

// Extract translation quaternion [..., 4]
struct ggml_tensor* ggml_dual_quaternion_translation(
    struct ggml_context* ctx,
    struct ggml_tensor* dq  // [..., 8]
);

// Dual quaternion addition
struct ggml_tensor* ggml_dual_quaternion_add(
    struct ggml_context* ctx,
    struct ggml_tensor* dq1,  // [..., 8]
    struct ggml_tensor* dq2   // [..., 8]
);

// Scalar multiplication
struct ggml_tensor* ggml_dual_quaternion_scale(
    struct ggml_context* ctx,
    struct ggml_tensor* dq,   // [..., 8]
    float scalar
);

#ifdef __cplusplus
}
#endif
