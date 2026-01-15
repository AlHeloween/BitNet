#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Screw theory for temporal evolution
// Screw motion: rotation around axis + translation along axis

// Screw parameters structure
struct screw_parameters {
    float omega[3];  // Angular velocity [x, y, z]
    float v[3];      // Linear velocity [x, y, z]
};

// Compute screw motion transformation
// Returns dual quaternion [8]
struct ggml_tensor* ggml_screw_compute_motion(
    struct ggml_context* ctx,
    const struct screw_parameters* params,
    float dt  // Time step
);

// Convert screw parameters to dual quaternion
struct ggml_tensor* ggml_screw_to_dual_quaternion(
    struct ggml_context* ctx,
    const float* omega,  // [3]
    const float* v,      // [3]
    float dt
);

// Extract screw parameters from dual quaternion
void ggml_dual_quaternion_to_screw(
    const float* dq,  // [8] dual quaternion
    float dt,
    struct screw_parameters* params_out
);

#ifdef __cplusplus
}
#endif
