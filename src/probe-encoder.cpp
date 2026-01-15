#include "probe-encoder.h"
#include "ggml-dq-linear.h"
#include "ggml-dq-attention.h"
#include "ggml-dual-quaternion.h"
#include "ggml.h"
#include "../../../lib/math_dualQuat.h"
#include "../../../lib/math_Quat.h"

#include <cmath>

using namespace DynaMap::math;

struct ggml_tensor* probe_encode(
    struct ggml_context* ctx,
    struct ggml_tensor* input_embeddings,
    struct probe_encoder_params* params,
    struct ggml_tensor* probe_weights
) {
    // input_embeddings: [batch, seq, hidden_dim]
    // output: [batch, seq, hidden_size, 8] (dual quaternion latents)
    
    GGML_ASSERT(ggml_n_dims(input_embeddings) == 3);
    
    int batch_size = input_embeddings->ne[0];
    int seq_len = input_embeddings->ne[1];
    int hidden_dim = input_embeddings->ne[2];
    int hidden_size = params->hidden_size;
    
    // Step 1: Project input embeddings to dual quaternion space
    // For each embedding dimension, create a dual quaternion
    // Simplified: replicate embedding 8 times and initialize as dual quaternion
    // Rotation: normalized embedding (first 4 dims), Translation: small random (last 4 dims)
    
    // Create dual quaternion tensor: [batch, seq, hidden_size, 8]
    int64_t dq_ne[4] = {batch_size, seq_len, hidden_size, 8};
    struct ggml_tensor* dq_latents = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, dq_ne);
    
    const float* emb_data = (const float*)input_embeddings->data;
    float* dq_data = (float*)dq_latents->data;
    
    // Initialize dual quaternions from embeddings
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < hidden_size; h++) {
                // Get embedding value (use modulo to handle different sizes)
                float emb_val = emb_data[(b * seq_len + s) * hidden_dim + (h % hidden_dim)];
                
                // Initialize rotation quaternion as unit quaternion
                // Use embedding value to create rotation
                float angle = emb_val * 0.1f;  // Scale down
                Quaternion q_r(cosf(angle), sinf(angle), 0.0f, 0.0f);
                q_r = q_r.normalize();
                
                // Initialize translation quaternion with small values
                Quaternion q_d(0.0f, emb_val * 0.01f, 0.0f, 0.0f);
                
                dualQuat dq(q_r, q_d);
                dq = dualQuat::normalize(dq);
                
                // Store: [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]
                int idx = ((b * seq_len + s) * hidden_size + h) * 8;
                dq_data[idx + 0] = dq.real.w;
                dq_data[idx + 1] = dq.real.x;
                dq_data[idx + 2] = dq.real.y;
                dq_data[idx + 3] = dq.real.z;
                dq_data[idx + 4] = dq.dual.w;
                dq_data[idx + 5] = dq.dual.x;
                dq_data[idx + 6] = dq.dual.y;
                dq_data[idx + 7] = dq.dual.z;
            }
        }
    }
    
    // Step 2: Apply Probe encoder layers (DQLinear + DQAttention)
    // For now, we'll do a simplified version
    // In full implementation, we'd have:
    // - Multiple layers of DQLinear + DQAttention
    // - Residual connections
    // - Normalization
    
    // For now, just normalize the dual quaternions
    struct ggml_tensor* dq_normalized = ggml_dual_quaternion_normalize(ctx, dq_latents);
    
    return dq_normalized;
}
