#include "ggml-probe-to-lattice.h"
#include "ggml.h"
#include "probe-encoder.h"
#include "ggml-ffe-quantization.h"
#include "ggml-lattice-addressing.h"
#include "ggml-lattice-storage.h"

#include <cstring>

int ggml_probe_to_lattice_pipeline(
    struct ggml_context* ctx,
    struct ggml_tensor* input_embeddings,
    struct ggml_tensor* probe_weights,
    struct ggml_tensor* sierpinski_centroids,
    struct lattice_memory_storage* storage,
    const char* text,
    struct lattice_address* addresses_out,
    int max_addresses
) {
    if (!ctx || !input_embeddings || !sierpinski_centroids || !storage || !addresses_out) {
        return 0;
    }
    
    // 1. Encode with Probe (simplified - in production would use full Probe encoder)
    // For now, we'll assume input_embeddings are already dual quaternions or use a simple projection
    // In production, this would call probe_encode()
    
    // Assume input is already in dual quaternion format [batch, seq, 8]
    // Or convert from embeddings to dual quaternions
    struct ggml_tensor* dq_latents = NULL;
    
    if (ggml_n_dims(input_embeddings) == 3 && input_embeddings->ne[2] == 8) {
        // Already dual quaternions
        dq_latents = input_embeddings;
    } else {
        // Need conversion (simplified - in production would use Probe encoder)
        // For now, create dummy dual quaternions
        int batch_size = (int)input_embeddings->ne[0];
        int seq_len = (int)input_embeddings->ne[1];
        int64_t dq_ne[3] = {batch_size, seq_len, 8};
        dq_latents = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, dq_ne);
        
        // Initialize with zeros (in production, would use Probe encoder)
        float* dq_data = (float*)dq_latents->data;
        int total = batch_size * seq_len * 8;
        for (int i = 0; i < total; i++) {
            dq_data[i] = 0.0f;
        }
    }
    
    // 2. Quantize to FFE addresses
    struct ggml_tensor* addresses = ggml_ffe_quantize(ctx, dq_latents, sierpinski_centroids);
    if (!addresses) {
        return 0;
    }
    
    // 3. Convert to lattice addresses and write to storage
    const int32_t* addr_data = (const int32_t*)addresses->data;
    const float* dq_data = (const float*)dq_latents->data;
    
    int batch_size = (int)dq_latents->ne[0];
    int seq_len = (int)dq_latents->ne[1];
    int total_elements = batch_size * seq_len;
    
    int count = 0;
    for (int i = 0; i < total_elements && count < max_addresses; i++) {
        // Decode FFE address
        uint16_t addr_bits = (uint16_t)addr_data[i];
        struct ffe_address ffe_addr = ffe_address_decode(addr_bits);
        
        // Convert to lattice address
        struct lattice_address lattice_addr = ffe_to_lattice(&ffe_addr);
        
        // Get dual quaternion
        const float* dq = &dq_data[i * 8];
        
        // Create memory entry
        struct lattice_memory_entry entry;
        memcpy(entry.embedding, dq, 8 * sizeof(float));
        
        if (text) {
            size_t text_len = strlen(text);
            entry.text = new char[text_len + 1];
            strcpy(entry.text, text);
        } else {
            entry.text = NULL;
        }
        
        entry.metadata = NULL;
        
        // Write to storage
        if (ggml_lattice_storage_write_leaf(storage, &lattice_addr, &entry)) {
            addresses_out[count] = lattice_addr;
            count++;
        }
        
        // Free text if allocated
        if (entry.text) {
            delete[] entry.text;
        }
    }
    
    return count;
}
