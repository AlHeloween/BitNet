// CUDA kernels for TKG (Temporal Knowledge Graph) scoring operations
// Stage 2: GPU acceleration for triple scoring and negative sampling

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

// Score triples using ComplEx-style scoring: Re(sum(e_s * e_r * conj(e_o)))
// Inputs:
//   entity_emb: (n_entities, dim) complex embeddings
//   relation_emb: (n_relations, dim) complex embeddings
//   s: (batch_size,) subject entity IDs
//   r: (batch_size,) relation IDs
//   o: (batch_size,) object entity IDs
// Output:
//   scores: (batch_size,) real scores
__global__ void tkg_score_triples_complex_kernel(
    const float2* entity_emb,      // Complex as float2 (real, imag)
    const float2* relation_emb,
    const int32_t* s,
    const int32_t* r,
    const int32_t* o,
    float* scores,
    int batch_size,
    int dim,
    int n_entities,
    int n_relations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int s_idx = s[idx];
    int r_idx = r[idx];
    int o_idx = o[idx];
    
    if (s_idx < 0 || s_idx >= n_entities || r_idx < 0 || r_idx >= n_relations || o_idx < 0 || o_idx >= n_entities) {
        scores[idx] = 0.0f;
        return;
    }
    
    const float2* es = &entity_emb[s_idx * dim];
    const float2* er = &relation_emb[r_idx * dim];
    const float2* eo = &entity_emb[o_idx * dim];
    
    float sum_real = 0.0f;
    for (int d = 0; d < dim; d++) {
        // es * er * conj(eo) = (es.real + i*es.imag) * (er.real + i*er.imag) * (eo.real - i*eo.imag)
        // = (es.real*er.real - es.imag*er.imag + i*(es.real*er.imag + es.imag*er.real)) * (eo.real - i*eo.imag)
        // Real part: (es.real*er.real - es.imag*er.imag)*eo.real + (es.real*er.imag + es.imag*er.real)*eo.imag
        float es_real = es[d].x;
        float es_imag = es[d].y;
        float er_real = er[d].x;
        float er_imag = er[d].y;
        float eo_real = eo[d].x;
        float eo_imag = eo[d].y;
        
        float prod_real = es_real * er_real - es_imag * er_imag;
        float prod_imag = es_real * er_imag + es_imag * er_real;
        
        sum_real += prod_real * eo_real + prod_imag * eo_imag;
    }
    
    scores[idx] = sum_real;
}

// Score all tails for a batch of (s, r) pairs
// Output: scores (batch_size, n_entities)
__global__ void tkg_score_all_tails_kernel(
    const float2* entity_emb,
    const float2* relation_emb,
    const int32_t* s,
    const int32_t* r,
    float* scores,
    int batch_size,
    int dim,
    int n_entities,
    int n_relations
) {
    int batch_idx = blockIdx.x;
    int entity_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || entity_idx >= n_entities) return;
    
    int s_idx = s[batch_idx];
    int r_idx = r[batch_idx];
    
    if (s_idx < 0 || s_idx >= n_entities || r_idx < 0 || r_idx >= n_relations) {
        scores[batch_idx * n_entities + entity_idx] = 0.0f;
        return;
    }
    
    const float2* es = &entity_emb[s_idx * dim];
    const float2* er = &relation_emb[r_idx * dim];
    const float2* eo = &entity_emb[entity_idx * dim];
    
    float sum_real = 0.0f;
    for (int d = 0; d < dim; d++) {
        float es_real = es[d].x;
        float es_imag = es[d].y;
        float er_real = er[d].x;
        float er_imag = er[d].y;
        float eo_real = eo[d].x;
        float eo_imag = eo[d].y;
        
        float prod_real = es_real * er_real - es_imag * er_imag;
        float prod_imag = es_real * er_imag + es_imag * er_real;
        
        sum_real += prod_real * eo_real + prod_imag * eo_imag;
    }
    
    scores[batch_idx * n_entities + entity_idx] = sum_real;
}

// Launch functions
extern "C" {
    void tkg_score_triples_complex_cuda_launch(
        const float2* entity_emb,
        const float2* relation_emb,
        const int32_t* s,
        const int32_t* r,
        const int32_t* o,
        float* scores,
        int batch_size,
        int dim,
        int n_entities,
        int n_relations
    ) {
        dim3 block(256);
        dim3 grid((batch_size + block.x - 1) / block.x);
        tkg_score_triples_complex_kernel<<<grid, block>>>(
            entity_emb, relation_emb, s, r, o, scores,
            batch_size, dim, n_entities, n_relations
        );
    }
    
    void tkg_score_all_tails_cuda_launch(
        const float2* entity_emb,
        const float2* relation_emb,
        const int32_t* s,
        const int32_t* r,
        float* scores,
        int batch_size,
        int dim,
        int n_entities,
        int n_relations
    ) {
        dim3 block(1, 256);
        dim3 grid(batch_size, (n_entities + block.y - 1) / block.y);
        tkg_score_all_tails_kernel<<<grid, block>>>(
            entity_emb, relation_emb, s, r, scores,
            batch_size, dim, n_entities, n_relations
        );
    }
}
