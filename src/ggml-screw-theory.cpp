#include "ggml-screw-theory.h"
#include "ggml.h"
#include "../../lib/math_dualQuat.h"
#include "../../lib/math_Quat.h"

#include <cmath>
#include <cstring>

using namespace DynaMap::math;

struct ggml_tensor* ggml_screw_compute_motion(
    struct ggml_context* ctx,
    const struct screw_parameters* params,
    float dt
) {
    if (!ctx || !params) {
        return NULL;
    }
    
    // Create output tensor [8]
    struct ggml_tensor* result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    float* out_data = (float*)result->data;
    
    // Compute rotation quaternion from omega
    // Formula: q = exp(omega * dt / 2)
    // For small angles: q ≈ [1, omega*dt/2]
    float omega_dt[3] = {
        params->omega[0] * dt,
        params->omega[1] * dt,
        params->omega[2] * dt,
    };
    
    float omega_norm = sqrtf(omega_dt[0] * omega_dt[0] + omega_dt[1] * omega_dt[1] + omega_dt[2] * omega_dt[2]);
    
    float half_angle = omega_norm / 2.0f;
    float cos_half = cosf(half_angle);
    float sin_half = sinf(half_angle);
    
    // Quaternion: [w, x, y, z] = [cos(θ/2), sin(θ/2) * axis]
    if (omega_norm > 1e-6f) {
        out_data[0] = cos_half;  // w
        out_data[1] = sin_half * omega_dt[0] / omega_norm;  // x
        out_data[2] = sin_half * omega_dt[1] / omega_norm;  // y
        out_data[3] = sin_half * omega_dt[2] / omega_norm;  // z
    } else {
        // Identity quaternion
        out_data[0] = 1.0f;
        out_data[1] = 0.0f;
        out_data[2] = 0.0f;
        out_data[3] = 0.0f;
    }
    
    // Compute translation quaternion
    // Translation: t = v * dt
    float t[3] = {
        params->v[0] * dt,
        params->v[1] * dt,
        params->v[2] * dt,
    };
    
    // Dual quaternion: real part = rotation, dual part = 0.5 * t * q_real
    out_data[4] = 0.0f;  // dual w
    out_data[5] = 0.5f * t[0];  // dual x
    out_data[6] = 0.5f * t[1];  // dual y
    out_data[7] = 0.5f * t[2];  // dual z
    
    return result;
}

struct ggml_tensor* ggml_screw_to_dual_quaternion(
    struct ggml_context* ctx,
    const float* omega,
    const float* v,
    float dt
) {
    if (!ctx || !omega || !v) {
        return NULL;
    }
    
    struct screw_parameters params;
    params.omega[0] = omega[0];
    params.omega[1] = omega[1];
    params.omega[2] = omega[2];
    params.v[0] = v[0];
    params.v[1] = v[1];
    params.v[2] = v[2];
    
    return ggml_screw_compute_motion(ctx, &params, dt);
}

void ggml_dual_quaternion_to_screw(
    const float* dq,
    float dt,
    struct screw_parameters* params_out
) {
    if (!dq || !params_out || dt <= 0.0f) {
        return;
    }
    
    // Extract real and dual parts
    float q_w = dq[0];
    float q_x = dq[1];
    float q_y = dq[2];
    float q_z = dq[3];
    
    float q_dual_w = dq[4];
    float q_dual_x = dq[5];
    float q_dual_y = dq[6];
    float q_dual_z = dq[7];
    
    // Extract rotation axis and angle from quaternion
    float half_angle = acosf(fmaxf(-1.0f, fminf(1.0f, q_w)));
    float angle = 2.0f * half_angle;
    
    float sin_half = sinf(half_angle);
    if (sin_half > 1e-6f) {
        params_out->omega[0] = (angle * q_x / sin_half) / dt;
        params_out->omega[1] = (angle * q_y / sin_half) / dt;
        params_out->omega[2] = (angle * q_z / sin_half) / dt;
    } else {
        params_out->omega[0] = 0.0f;
        params_out->omega[1] = 0.0f;
        params_out->omega[2] = 0.0f;
    }
    
    // Extract translation from dual part
    // t = 2 * q_dual[1:4]
    params_out->v[0] = (2.0f * q_dual_x) / dt;
    params_out->v[1] = (2.0f * q_dual_y) / dt;
    params_out->v[2] = (2.0f * q_dual_z) / dt;
}
