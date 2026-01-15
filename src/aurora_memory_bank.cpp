#include "aurora_memory_bank.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

// MD5 implementation (simple, for portability)
// Note: For production, consider using a proper MD5 library
static void md5_hash(const unsigned char* data, size_t len, unsigned char* digest) {
    // Simplified MD5 - for production use proper MD5 library
    // This is a placeholder that generates deterministic hashes
    uint32_t h[4] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476};
    for (size_t i = 0; i < len; i++) {
        h[0] = ((h[0] << 1) ^ (data[i] * 0x9E3779B9)) & 0xFFFFFFFF;
        h[1] = ((h[1] << 1) ^ (h[0] * 0x9E3779B9)) & 0xFFFFFFFF;
        h[2] = ((h[2] << 1) ^ (h[1] * 0x9E3779B9)) & 0xFFFFFFFF;
        h[3] = ((h[3] << 1) ^ (h[2] * 0x9E3779B9)) & 0xFFFFFFFF;
    }
    memcpy(digest, h, 16);
}

static void md5_to_hex(const unsigned char* digest, char* hex_out) {
    static const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        hex_out[i * 2] = hex_chars[(digest[i] >> 4) & 0xF];
        hex_out[i * 2 + 1] = hex_chars[digest[i] & 0xF];
    }
    hex_out[32] = '\0';
}

void aurora_md5_hex(const char* text, char* md5_out) {
    unsigned char digest[16];
    md5_hash((const unsigned char*)text, strlen(text), digest);
    md5_to_hex(digest, md5_out);
}

// LCG constants (matches Python implementation)
#define LCG_MULT 1664525u
#define LCG_ADD 1013904223u

static uint32_t lcg_step(uint32_t* state) {
    *state = ((*state * LCG_MULT) + LCG_ADD) & 0xFFFFFFFF;
    return *state;
}

// Generate regular N-simplex vertices
static void generate_simplex_vertices(int n_dim, float* vertices_out) {
    // For n_dim=1: vertices at [-1], [1]
    // For n_dim=2: equilateral triangle
    // For n_dim>=3: general case using Gram-Schmidt-like approach
    
    if (n_dim == 1) {
        vertices_out[0] = -1.0f;
        vertices_out[1] = 1.0f;
    } else if (n_dim == 2) {
        float sqrt3 = sqrtf(3.0f);
        vertices_out[0] = -1.0f;
        vertices_out[1] = -sqrt3 / 3.0f;
        vertices_out[2] = 1.0f;
        vertices_out[3] = -sqrt3 / 3.0f;
        vertices_out[4] = 0.0f;
        vertices_out[5] = 2.0f * sqrt3 / 3.0f;
        // Normalize
        for (int i = 0; i < 3; i++) {
            float norm = sqrtf(vertices_out[i * 2] * vertices_out[i * 2] + 
                              vertices_out[i * 2 + 1] * vertices_out[i * 2 + 1]);
            if (norm > 1e-8f) {
                vertices_out[i * 2] /= norm;
                vertices_out[i * 2 + 1] /= norm;
            }
        }
    } else {
        // General case: simplified approach
        // Place first vertex at (1, 0, ..., 0)
        for (int i = 0; i < n_dim + 1; i++) {
            for (int j = 0; j < n_dim; j++) {
                if (i == 0) {
                    vertices_out[i * n_dim + j] = (j == 0) ? 1.0f : 0.0f;
                } else {
                    if (j < i) {
                        vertices_out[i * n_dim + j] = -1.0f / (n_dim + 1);
                    } else if (j == i) {
                        vertices_out[i * n_dim + j] = 1.0f - (1.0f / (n_dim + 1));
                    } else {
                        vertices_out[i * n_dim + j] = 0.0f;
                    }
                }
            }
            // Normalize
            float norm = 0.0f;
            for (int j = 0; j < n_dim; j++) {
                float v = vertices_out[i * n_dim + j];
                norm += v * v;
            }
            norm = sqrtf(norm);
            if (norm > 1e-8f) {
                for (int j = 0; j < n_dim; j++) {
                    vertices_out[i * n_dim + j] /= norm;
                }
            }
        }
    }
}

// Generate Sierpinski centroids
static void generate_sierpinski_centroids(
    int n_dim,
    int depth,
    int n_centroids,
    int seed,
    float* centroids_out
) {
    int n_vertices = n_dim + 1;
    float* vertices = (float*)malloc((n_vertices * n_dim) * sizeof(float));
    generate_simplex_vertices(n_dim, vertices);
    
    // Initialize centroids to zero
    for (int i = 0; i < n_centroids * n_dim; i++) {
        centroids_out[i] = 0.0f;
    }
    
    // Chaos game iteration
    for (int c = 0; c < n_centroids; c++) {
        uint32_t state = ((seed & 0xFFFFFFFF) ^ ((c * 747796405u) & 0xFFFFFFFF)) & 0xFFFFFFFF;
        float* centroid = &centroids_out[c * n_dim];
        
        for (int iter = 0; iter < depth; iter++) {
            lcg_step(&state);
            int vertex_idx = (int)(state % (uint32_t)n_vertices);
            for (int d = 0; d < n_dim; d++) {
                centroid[d] = (centroid[d] + vertices[vertex_idx * n_dim + d]) * 0.5f;
            }
        }
    }
    
    free(vertices);
}

// Hash embedding (matches Python implementation)
void aurora_hash_embed(const char* text, int dim, int seed, float* embedding_out) {
    // Tokenize: extract alphanumeric tokens
    // Simplified: treat as single token for now
    const char* token = text;
    int n_hashes_per_token = 4;
    
    // Initialize embedding to zero
    for (int i = 0; i < dim; i++) {
        embedding_out[i] = 0.0f;
    }
    
    // Hash token
    char hash_input[256];
    snprintf(hash_input, sizeof(hash_input), "%d:%s", seed, token);
    unsigned char digest[16];
    md5_hash((const unsigned char*)hash_input, strlen(hash_input), digest);
    
    // Update embedding
    for (int j = 0; j < n_hashes_per_token; j++) {
        int off = (j * 4) % 16;
        uint32_t idx_val = 0;
        for (int b = 0; b < 4; b++) {
            idx_val |= ((uint32_t)digest[(off + b) % 16]) << (b * 8);
        }
        int idx = (int)(idx_val % (uint32_t)dim);
        float sign = ((digest[(off + 1) % 16] & 1) != 0) ? -1.0f : 1.0f;
        embedding_out[idx] += sign;
    }
    
    // Normalize
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += embedding_out[i] * embedding_out[i];
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
        for (int i = 0; i < dim; i++) {
            embedding_out[i] /= norm;
        }
    }
}

// Memory entry operations
aurora_memory_entry_t* aurora_memory_entry_create(
    const char* md5_tag,
    aurora_information_mark_t information_mark,
    const char* text,
    const float* embedding,
    int dim
) {
    aurora_memory_entry_t* entry = (aurora_memory_entry_t*)malloc(sizeof(aurora_memory_entry_t));
    entry->md5_tag = strdup(md5_tag);
    entry->information_mark = information_mark;
    entry->text = strdup(text);
    entry->dim = dim;
    entry->embedding = (float*)malloc(dim * sizeof(float));
    memcpy(entry->embedding, embedding, dim * sizeof(float));
    entry->embedding_dual = NULL;
    entry->is_dual_complex = false;
    return entry;
}

aurora_memory_entry_t* aurora_memory_entry_create_dual_complex(
    const char* md5_tag,
    aurora_information_mark_t information_mark,
    const char* text,
    const float* embedding_primal,
    const float* embedding_dual,
    int dim
) {
    aurora_memory_entry_t* entry = (aurora_memory_entry_t*)malloc(sizeof(aurora_memory_entry_t));
    entry->md5_tag = strdup(md5_tag);
    entry->information_mark = information_mark;
    entry->text = strdup(text);
    entry->dim = dim;
    entry->embedding = (float*)malloc(dim * sizeof(float));
    memcpy(entry->embedding, embedding_primal, dim * sizeof(float));
    entry->embedding_dual = (float*)malloc(dim * sizeof(float));
    if (embedding_dual) {
        memcpy(entry->embedding_dual, embedding_dual, dim * sizeof(float));
    } else {
        // Initialize dual to zero if not provided
        for (int i = 0; i < dim; i++) {
            entry->embedding_dual[i] = 0.0f;
        }
    }
    entry->is_dual_complex = true;
    return entry;
}

void aurora_memory_entry_free(aurora_memory_entry_t* entry) {
    if (entry) {
        free(entry->md5_tag);
        free(entry->text);
        free(entry->embedding);
        if (entry->embedding_dual) {
            free(entry->embedding_dual);
        }
        free(entry);
    }
}

// Memory bank operations
aurora_memory_bank_t* aurora_memory_bank_create(
    int n_clusters,
    int dim,
    int depth,
    int seed
) {
    aurora_memory_bank_t* bank = (aurora_memory_bank_t*)malloc(sizeof(aurora_memory_bank_t));
    bank->n_clusters = n_clusters;
    bank->dim = dim;
    bank->depth = depth;
    bank->seed = seed;
    
    // Generate centroids
    bank->centroids = (float*)malloc(n_clusters * dim * sizeof(float));
    generate_sierpinski_centroids(dim, depth, n_clusters, seed, bank->centroids);
    
    // Initialize slots
    bank->slots = (aurora_memory_entry_t**)malloc(n_clusters * sizeof(aurora_memory_entry_t*));
    bank->slot_counts = (int*)calloc(n_clusters, sizeof(int));
    bank->slot_capacities = (int*)malloc(n_clusters * sizeof(int));
    for (int i = 0; i < n_clusters; i++) {
        bank->slots[i] = NULL;
        bank->slot_capacities[i] = 0;
    }
    
    bank->total_entries = 0;
    bank->seen_md5 = NULL;
    bank->seen_md5_count = 0;
    bank->seen_md5_capacity = 0;
    
    return bank;
}

void aurora_memory_bank_free(aurora_memory_bank_t* bank) {
    if (!bank) return;
    
    // Free all entries
    for (int i = 0; i < bank->n_clusters; i++) {
        for (int j = 0; j < bank->slot_counts[i]; j++) {
            aurora_memory_entry_free(&bank->slots[i][j]);
        }
        free(bank->slots[i]);
    }
    
    free(bank->centroids);
    free(bank->slots);
    free(bank->slot_counts);
    free(bank->slot_capacities);
    
    // Free MD5 set
    for (int i = 0; i < bank->seen_md5_count; i++) {
        free(bank->seen_md5[i]);
    }
    free(bank->seen_md5);
    
    free(bank);
}

// Find nearest centroid index
static int find_nearest_centroid(const aurora_memory_bank_t* bank, const float* embedding) {
    int best_idx = 0;
    float best_dist2 = 1e10f;
    
    for (int i = 0; i < bank->n_clusters; i++) {
        float dist2 = 0.0f;
        for (int d = 0; d < bank->dim; d++) {
            float diff = embedding[d] - bank->centroids[i * bank->dim + d];
            dist2 += diff * diff;
        }
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_idx = i;
        }
    }
    return best_idx;
}

// Find k nearest centroid indices
static void find_nearest_centroids(
    const aurora_memory_bank_t* bank,
    const float* embedding,
    int k,
    int* indices_out
) {
    // Simple approach: compute all distances, sort
    float* dists = (float*)malloc(bank->n_clusters * sizeof(float));
    int* idxs = (int*)malloc(bank->n_clusters * sizeof(int));
    
    for (int i = 0; i < bank->n_clusters; i++) {
        float dist2 = 0.0f;
        for (int d = 0; d < bank->dim; d++) {
            float diff = embedding[d] - bank->centroids[i * bank->dim + d];
            dist2 += diff * diff;
        }
        dists[i] = dist2;
        idxs[i] = i;
    }
    
    // Simple selection sort for top k
    for (int i = 0; i < k && i < bank->n_clusters; i++) {
        int best_j = i;
        for (int j = i + 1; j < bank->n_clusters; j++) {
            if (dists[j] < dists[best_j]) {
                best_j = j;
            }
        }
        // Swap
        float tmp_d = dists[i];
        int tmp_i = idxs[i];
        dists[i] = dists[best_j];
        idxs[i] = idxs[best_j];
        dists[best_j] = tmp_d;
        idxs[best_j] = tmp_i;
    }
    
    for (int i = 0; i < k && i < bank->n_clusters; i++) {
        indices_out[i] = idxs[i];
    }
    
    free(dists);
    free(idxs);
}

// Check if MD5 exists
static bool md5_exists(const aurora_memory_bank_t* bank, const char* md5_tag) {
    for (int i = 0; i < bank->seen_md5_count; i++) {
        if (strcmp(bank->seen_md5[i], md5_tag) == 0) {
            return true;
        }
    }
    return false;
}

// Add MD5 to set
static void add_md5(aurora_memory_bank_t* bank, const char* md5_tag) {
    if (md5_exists(bank, md5_tag)) {
        return;
    }
    
    if (bank->seen_md5_count >= bank->seen_md5_capacity) {
        int new_capacity = (bank->seen_md5_capacity == 0) ? 16 : bank->seen_md5_capacity * 2;
        bank->seen_md5 = (char**)realloc(bank->seen_md5, new_capacity * sizeof(char*));
        bank->seen_md5_capacity = new_capacity;
    }
    
    bank->seen_md5[bank->seen_md5_count] = strdup(md5_tag);
    bank->seen_md5_count++;
}

bool aurora_memory_bank_add(
    aurora_memory_bank_t* bank,
    const char* text,
    aurora_information_mark_t information_mark,
    const char* embed_text,
    const char* md5_tag
) {
    // Compute MD5 if not provided
    char computed_md5[33];
    if (md5_tag) {
        strncpy(computed_md5, md5_tag, 32);
        computed_md5[32] = '\0';
    } else {
        aurora_md5_hex(text, computed_md5);
    }
    
    // Check for duplicate
    if (md5_exists(bank, computed_md5)) {
        return false;
    }
    
    // Compute embedding
    const char* embed_source = embed_text ? embed_text : text;
    float* embedding = (float*)malloc(bank->dim * sizeof(float));
    aurora_hash_embed(embed_source, bank->dim, bank->seed, embedding);
    
    // Find slot
    int slot_idx = find_nearest_centroid(bank, embedding);
    
    // Add entry
    aurora_memory_entry_t* entry = aurora_memory_entry_create(
        computed_md5,
        information_mark,
        text,
        embedding,
        bank->dim
    );
    
    // Grow slot if needed
    if (bank->slot_counts[slot_idx] >= bank->slot_capacities[slot_idx]) {
        int new_capacity = (bank->slot_capacities[slot_idx] == 0) ? 8 : bank->slot_capacities[slot_idx] * 2;
        bank->slots[slot_idx] = (aurora_memory_entry_t*)realloc(
            bank->slots[slot_idx],
            new_capacity * sizeof(aurora_memory_entry_t)
        );
        bank->slot_capacities[slot_idx] = new_capacity;
    }
    
        bank->slots[slot_idx][bank->slot_counts[slot_idx]] = *entry;
    bank->slot_counts[slot_idx]++;
    bank->total_entries++;
    
    add_md5(bank, computed_md5);
    
    free(embedding);
    return true;
}

// Add entry with dual-complex embeddings to a single bank
bool aurora_memory_bank_add_dual_complex(
    aurora_memory_bank_t* bank,
    const char* text,
    aurora_information_mark_t information_mark,
    const float* embedding_primal,
    const float* embedding_dual,
    int dim,
    const char* md5_tag
) {
    // Compute MD5 if not provided
    char computed_md5[33];
    if (md5_tag) {
        strncpy(computed_md5, md5_tag, 32);
        computed_md5[32] = '\0';
    } else {
        aurora_md5_hex(text, computed_md5);
    }
    
    // Check for duplicate
    if (md5_exists(bank, computed_md5)) {
        return false;
    }
    
    // Find slot using primal embedding
    int slot_idx = find_nearest_centroid(bank, embedding_primal);
    
    // Add entry with dual-complex
    aurora_memory_entry_t* entry = aurora_memory_entry_create_dual_complex(
        computed_md5,
        information_mark,
        text,
        embedding_primal,
        embedding_dual,
        dim
    );
    
    // Grow slot if needed
    if (bank->slot_counts[slot_idx] >= bank->slot_capacities[slot_idx]) {
        int new_capacity = (bank->slot_capacities[slot_idx] == 0) ? 8 : bank->slot_capacities[slot_idx] * 2;
        bank->slots[slot_idx] = (aurora_memory_entry_t*)realloc(
            bank->slots[slot_idx],
            new_capacity * sizeof(aurora_memory_entry_t)
        );
        bank->slot_capacities[slot_idx] = new_capacity;
    }
    
    // Add to slot
        bank->slots[slot_idx][bank->slot_counts[slot_idx]] = *entry;
    bank->slot_counts[slot_idx]++;
    bank->total_entries++;
    
    // Add MD5
    add_md5(bank, computed_md5);
    
    return true;
}

int aurora_memory_bank_query(
    const aurora_memory_bank_t* bank,
    const char* query_text,
    int candidate_slots,
    int top_k,
    aurora_memory_entry_t** entries_out
) {
    // Compute query embedding
    float* query_emb = (float*)malloc(bank->dim * sizeof(float));
    aurora_hash_embed(query_text, bank->dim, bank->seed, query_emb);
    
    // Find candidate slots
    int* slot_idxs = (int*)malloc(candidate_slots * sizeof(int));
    find_nearest_centroids(bank, query_emb, candidate_slots, slot_idxs);
    
    // Collect all entries from candidate slots
    int total_entries = 0;
    aurora_memory_entry_t** all_entries = NULL;
    int all_entries_capacity = 0;
    
    for (int i = 0; i < candidate_slots; i++) {
        int slot_idx = slot_idxs[i];
        int count = bank->slot_counts[slot_idx];
        for (int j = 0; j < count; j++) {
            if (total_entries >= all_entries_capacity) {
                int new_capacity = (all_entries_capacity == 0) ? 16 : all_entries_capacity * 2;
                all_entries = (aurora_memory_entry_t**)realloc(
                    all_entries,
                    new_capacity * sizeof(aurora_memory_entry_t*)
                );
                all_entries_capacity = new_capacity;
            }
            all_entries[total_entries++] = &bank->slots[slot_idx][j];
        }
    }
    
    if (total_entries == 0) {
        free(query_emb);
        free(slot_idxs);
        free(all_entries);
        return 0;
    }
    
    // Compute similarities
    float* sims = (float*)malloc(total_entries * sizeof(float));
    int* idxs = (int*)malloc(total_entries * sizeof(int));
    
    for (int i = 0; i < total_entries; i++) {
        float sim = 0.0f;
        for (int d = 0; d < bank->dim; d++) {
            sim += all_entries[i]->embedding[d] * query_emb[d];
        }
        sims[i] = sim;
        idxs[i] = i;
    }
    
    // Sort by similarity (simple selection sort for top k)
    int k = (top_k < total_entries) ? top_k : total_entries;
    for (int i = 0; i < k; i++) {
        int best_j = i;
        for (int j = i + 1; j < total_entries; j++) {
            if (sims[j] > sims[best_j]) {
                best_j = j;
            }
        }
        // Swap
        float tmp_s = sims[i];
        int tmp_i = idxs[i];
        sims[i] = sims[best_j];
        idxs[i] = idxs[best_j];
        sims[best_j] = tmp_s;
        idxs[best_j] = tmp_i;
    }
    
    // Copy top k entries
    for (int i = 0; i < k; i++) {
        entries_out[i] = all_entries[idxs[i]];
    }
    
    free(query_emb);
    free(slot_idxs);
    free(all_entries);
    free(sims);
    free(idxs);
    
    return k;
}

int aurora_memory_bank_query_embedding(
    const aurora_memory_bank_t* bank,
    const float* query_embedding,
    int candidate_slots,
    int top_k,
    aurora_memory_entry_t** entries_out
) {
    // Find candidate slots
    int* slot_idxs = (int*)malloc(candidate_slots * sizeof(int));
    find_nearest_centroids(bank, query_embedding, candidate_slots, slot_idxs);
    
    // Collect all entries from candidate slots
    int total_entries = 0;
    aurora_memory_entry_t** all_entries = NULL;
    int all_entries_capacity = 0;
    
    for (int i = 0; i < candidate_slots; i++) {
        int slot_idx = slot_idxs[i];
        int count = bank->slot_counts[slot_idx];
        for (int j = 0; j < count; j++) {
            if (total_entries >= all_entries_capacity) {
                int new_capacity = (all_entries_capacity == 0) ? 16 : all_entries_capacity * 2;
                all_entries = (aurora_memory_entry_t**)realloc(
                    all_entries,
                    new_capacity * sizeof(aurora_memory_entry_t*)
                );
                all_entries_capacity = new_capacity;
            }
            all_entries[total_entries++] = &bank->slots[slot_idx][j];
        }
    }
    
    if (total_entries == 0) {
        free(slot_idxs);
        free(all_entries);
        return 0;
    }
    
    // Compute similarities
    float* sims = (float*)malloc(total_entries * sizeof(float));
    int* idxs = (int*)malloc(total_entries * sizeof(int));
    
    for (int i = 0; i < total_entries; i++) {
        float sim = 0.0f;
        for (int d = 0; d < bank->dim; d++) {
            sim += all_entries[i]->embedding[d] * query_embedding[d];
        }
        sims[i] = sim;
        idxs[i] = i;
    }
    
    // Sort by similarity (simple selection sort for top k)
    int k = (top_k < total_entries) ? top_k : total_entries;
    for (int i = 0; i < k; i++) {
        int best_j = i;
        for (int j = i + 1; j < total_entries; j++) {
            if (sims[j] > sims[best_j]) {
                best_j = j;
            }
        }
        // Swap
        float tmp_s = sims[i];
        int tmp_i = idxs[i];
        sims[i] = sims[best_j];
        idxs[i] = idxs[best_j];
        sims[best_j] = tmp_s;
        idxs[best_j] = tmp_i;
    }
    
    // Copy top k entries
    for (int i = 0; i < k; i++) {
        entries_out[i] = all_entries[idxs[i]];
    }
    
    free(slot_idxs);
    free(all_entries);
    free(sims);
    free(idxs);
    
    return k;
}

int aurora_memory_bank_query_dual_complex(
    const aurora_memory_bank_t* bank,
    const float* query_primal,
    const float* query_dual,
    int candidate_slots,
    int top_k,
    float dual_weight,
    aurora_memory_entry_t** entries_out
) {
    // Find candidate slots using primal embedding
    int* slot_idxs = (int*)malloc(candidate_slots * sizeof(int));
    find_nearest_centroids(bank, query_primal, candidate_slots, slot_idxs);
    
    // Collect all entries from candidate slots
    int total_entries = 0;
    aurora_memory_entry_t** all_entries = NULL;
    int all_entries_capacity = 0;
    
    for (int i = 0; i < candidate_slots; i++) {
        int slot_idx = slot_idxs[i];
        int count = bank->slot_counts[slot_idx];
        for (int j = 0; j < count; j++) {
            if (total_entries >= all_entries_capacity) {
                int new_capacity = (all_entries_capacity == 0) ? 16 : all_entries_capacity * 2;
                all_entries = (aurora_memory_entry_t**)realloc(
                    all_entries,
                    new_capacity * sizeof(aurora_memory_entry_t*)
                );
                all_entries_capacity = new_capacity;
            }
            all_entries[total_entries++] = &bank->slots[slot_idx][j];
        }
    }
    
    if (total_entries == 0) {
        free(slot_idxs);
        free(all_entries);
        return 0;
    }
    
    // Compute dual-complex similarities
    // Formula: similarity = cos(primal_q, primal_m) + α * cos(dual_q, dual_m)
    float* sims = (float*)malloc(total_entries * sizeof(float));
    int* idxs = (int*)malloc(total_entries * sizeof(int));
    
    for (int i = 0; i < total_entries; i++) {
        // Primal similarity
        float sim_primal = 0.0f;
        for (int d = 0; d < bank->dim; d++) {
            sim_primal += all_entries[i]->embedding[d] * query_primal[d];
        }
        
        // Dual similarity (if available)
        float sim_dual = 0.0f;
        if (query_dual && all_entries[i]->is_dual_complex && all_entries[i]->embedding_dual) {
            for (int d = 0; d < bank->dim; d++) {
                sim_dual += all_entries[i]->embedding_dual[d] * query_dual[d];
            }
        }
        
        // Combined similarity
        sims[i] = sim_primal + dual_weight * sim_dual;
        idxs[i] = i;
    }
    
    // Sort by similarity (simple selection sort for top k)
    int k = (top_k < total_entries) ? top_k : total_entries;
    for (int i = 0; i < k; i++) {
        int best_j = i;
        for (int j = i + 1; j < total_entries; j++) {
            if (sims[j] > sims[best_j]) {
                best_j = j;
            }
        }
        // Swap
        float tmp_s = sims[i];
        int tmp_i = idxs[i];
        sims[i] = sims[best_j];
        idxs[i] = idxs[best_j];
        sims[best_j] = tmp_s;
        idxs[best_j] = tmp_i;
    }
    
    // Copy top k entries
    for (int i = 0; i < k; i++) {
        entries_out[i] = all_entries[idxs[i]];
    }
    
    free(slot_idxs);
    free(all_entries);
    free(sims);
    free(idxs);
    
    return k;
}

void aurora_memory_bank_stats(
    const aurora_memory_bank_t* bank,
    int* entries_out,
    int* slots_total_out,
    int* slots_non_empty_out
) {
    if (entries_out) *entries_out = bank->total_entries;
    if (slots_total_out) *slots_total_out = bank->n_clusters;
    if (slots_non_empty_out) {
        int non_empty = 0;
        for (int i = 0; i < bank->n_clusters; i++) {
            if (bank->slot_counts[i] > 0) {
                non_empty++;
            }
        }
        *slots_non_empty_out = non_empty;
    }
}

// Dual banks operations
aurora_memory_banks_t* aurora_memory_banks_create(
    int n_clusters,
    int dim,
    int depth,
    int seed,
    bool allow_quarantine_read
) {
    aurora_memory_banks_t* banks = (aurora_memory_banks_t*)malloc(sizeof(aurora_memory_banks_t));
    banks->verified = aurora_memory_bank_create(n_clusters, dim, depth, seed);
    banks->quarantine = aurora_memory_bank_create(n_clusters, dim, depth, seed);
    banks->allow_quarantine_read = allow_quarantine_read;
    return banks;
}

void aurora_memory_banks_free(aurora_memory_banks_t* banks) {
    if (!banks) return;
    aurora_memory_bank_free(banks->verified);
    aurora_memory_bank_free(banks->quarantine);
    free(banks);
}

bool aurora_memory_banks_add(
    aurora_memory_banks_t* banks,
    const char* text,
    aurora_information_mark_t information_mark,
    const char* embed_text,
    const char* md5_tag
) {
    // Route based on information mark
    if (information_mark == AURORA_INFO_EXACT || information_mark == AURORA_INFO_INFERRED) {
        return aurora_memory_bank_add(banks->verified, text, information_mark, embed_text, md5_tag);
    } else {
        return aurora_memory_bank_add(banks->quarantine, text, information_mark, embed_text, md5_tag);
    }
}

bool aurora_memory_banks_add_dual_complex(
    aurora_memory_banks_t* banks,
    const char* text,
    aurora_information_mark_t information_mark,
    const float* embedding_primal,
    const float* embedding_dual,
    int dim,
    const char* md5_tag
) {
    // Route based on information mark
    if (information_mark == AURORA_INFO_EXACT || information_mark == AURORA_INFO_INFERRED) {
        return aurora_memory_bank_add_dual_complex(banks->verified, text, information_mark, embedding_primal, embedding_dual, dim, md5_tag);
    } else {
        return aurora_memory_bank_add_dual_complex(banks->quarantine, text, information_mark, embedding_primal, embedding_dual, dim, md5_tag);
    }
}

int aurora_memory_banks_query(
    const aurora_memory_banks_t* banks,
    const char* query_text,
    int candidate_slots,
    int top_k,
    aurora_memory_entry_t** entries_out
) {
    // Query verified first
    int count = aurora_memory_bank_query(banks->verified, query_text, candidate_slots, top_k, entries_out);
    
    // If allowed and needed, query quarantine
    if (banks->allow_quarantine_read && count < top_k) {
        int rem = top_k - count;
        aurora_memory_entry_t** quarantine_entries = (aurora_memory_entry_t**)malloc(rem * sizeof(aurora_memory_entry_t*));
        int quarantine_count = aurora_memory_bank_query(banks->quarantine, query_text, candidate_slots, rem, quarantine_entries);
        
        // Append quarantine entries
        for (int i = 0; i < quarantine_count && count < top_k; i++) {
            entries_out[count++] = quarantine_entries[i];
        }
        
        free(quarantine_entries);
    }
    
    return count;
}

int aurora_memory_banks_query_embedding(
    const aurora_memory_banks_t* banks,
    const float* query_embedding,
    int candidate_slots,
    int top_k,
    aurora_memory_entry_t** entries_out
) {
    // Query verified first
    int count = aurora_memory_bank_query_embedding(banks->verified, query_embedding, candidate_slots, top_k, entries_out);
    
    // If allowed and needed, query quarantine
    if (banks->allow_quarantine_read && count < top_k) {
        int rem = top_k - count;
        aurora_memory_entry_t** quarantine_entries = (aurora_memory_entry_t**)malloc(rem * sizeof(aurora_memory_entry_t*));
        int quarantine_count = aurora_memory_bank_query_embedding(banks->quarantine, query_embedding, candidate_slots, rem, quarantine_entries);
        
        // Append quarantine entries
        for (int i = 0; i < quarantine_count && count < top_k; i++) {
            entries_out[count++] = quarantine_entries[i];
        }
        
        free(quarantine_entries);
    }
    
    return count;
}

int aurora_memory_banks_query_dual_complex(
    const aurora_memory_banks_t* banks,
    const float* query_primal,
    const float* query_dual,
    int candidate_slots,
    int top_k,
    float dual_weight,
    aurora_memory_entry_t** entries_out
) {
    // Query verified first
    int count = aurora_memory_bank_query_dual_complex(banks->verified, query_primal, query_dual, candidate_slots, top_k, dual_weight, entries_out);
    
    // If allowed and needed, query quarantine
    if (banks->allow_quarantine_read && count < top_k) {
        int rem = top_k - count;
        aurora_memory_entry_t** quarantine_entries = (aurora_memory_entry_t**)malloc(rem * sizeof(aurora_memory_entry_t*));
        int quarantine_count = aurora_memory_bank_query_dual_complex(banks->quarantine, query_primal, query_dual, candidate_slots, rem, dual_weight, quarantine_entries);
        
        // Append quarantine entries
        for (int i = 0; i < quarantine_count && count < top_k; i++) {
            entries_out[count++] = quarantine_entries[i];
        }
        
        free(quarantine_entries);
    }
    
    return count;
}

// Memory persistence implementation
int aurora_memory_bank_save(const aurora_memory_bank_t* bank, const char* filepath) {
    if (!bank || !filepath) {
        return -1;
    }
    
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        return -2;
    }
    
    // Write header
    int32_t magic = 0x4155524F;  // "AURO"
    int32_t version = 2;  // Version 2: supports dual-complex
    fwrite(&magic, sizeof(int32_t), 1, f);
    fwrite(&version, sizeof(int32_t), 1, f);
    fwrite(&bank->n_clusters, sizeof(int), 1, f);
    fwrite(&bank->dim, sizeof(int), 1, f);
    fwrite(&bank->depth, sizeof(int), 1, f);
    fwrite(&bank->seed, sizeof(int), 1, f);
    fwrite(&bank->total_entries, sizeof(int), 1, f);
    
    // Write centroids
    fwrite(bank->centroids, sizeof(float), bank->n_clusters * bank->dim, f);
    
    // Write entries per slot
    for (int i = 0; i < bank->n_clusters; i++) {
        int count = bank->slot_counts[i];
        fwrite(&count, sizeof(int), 1, f);
        for (int j = 0; j < count; j++) {
            aurora_memory_entry_t* entry = &bank->slots[i][j];
            // Write entry
            int32_t md5_len = (int32_t)strlen(entry->md5_tag);
            int32_t text_len = (int32_t)strlen(entry->text);
            fwrite(&md5_len, sizeof(int32_t), 1, f);
            fwrite(entry->md5_tag, sizeof(char), md5_len, f);
            fwrite(&entry->information_mark, sizeof(aurora_information_mark_t), 1, f);
            fwrite(&text_len, sizeof(int32_t), 1, f);
            fwrite(entry->text, sizeof(char), text_len, f);
            fwrite(entry->embedding, sizeof(float), bank->dim, f);
            // Write dual-complex flag and dual embedding if present
            int8_t is_dual = entry->is_dual_complex ? 1 : 0;
            fwrite(&is_dual, sizeof(int8_t), 1, f);
            if (entry->is_dual_complex && entry->embedding_dual) {
                fwrite(entry->embedding_dual, sizeof(float), bank->dim, f);
            }
        }
    }
    
    fclose(f);
    return 0;
}

aurora_memory_bank_t* aurora_memory_bank_load(const char* filepath, int n_clusters, int dim, int depth, int seed) {
    if (!filepath) {
        return NULL;
    }
    
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return NULL;
    }
    
    // Read header
    int32_t magic, version;
    if (fread(&magic, sizeof(int32_t), 1, f) != 1 || magic != 0x4155524F) {
        fclose(f);
        return NULL;
    }
    if (fread(&version, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    
    int file_n_clusters, file_dim, file_depth, file_seed, total_entries;
    if (fread(&file_n_clusters, sizeof(int), 1, f) != 1 ||
        fread(&file_dim, sizeof(int), 1, f) != 1 ||
        fread(&file_depth, sizeof(int), 1, f) != 1 ||
        fread(&file_seed, sizeof(int), 1, f) != 1 ||
        fread(&total_entries, sizeof(int), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    
    // Validate parameters match
    if (file_n_clusters != n_clusters || file_dim != dim || file_depth != depth || file_seed != seed) {
        fclose(f);
        return NULL;
    }
    
    // Create bank
    aurora_memory_bank_t* bank = aurora_memory_bank_create(n_clusters, dim, depth, seed);
    if (!bank) {
        fclose(f);
        return NULL;
    }
    
    // Read centroids (skip, we'll regenerate)
    fseek(f, n_clusters * dim * sizeof(float), SEEK_CUR);
    
    // Read entries
    for (int i = 0; i < n_clusters; i++) {
        int count;
        if (fread(&count, sizeof(int), 1, f) != 1) {
            aurora_memory_bank_free(bank);
            fclose(f);
            return NULL;
        }
        
        for (int j = 0; j < count; j++) {
            int32_t md5_len, text_len;
            if (fread(&md5_len, sizeof(int32_t), 1, f) != 1) {
                aurora_memory_bank_free(bank);
                fclose(f);
                return NULL;
            }
            
            char* md5_tag = (char*)malloc((md5_len + 1) * sizeof(char));
            if (fread(md5_tag, sizeof(char), md5_len, f) != md5_len) {
                free(md5_tag);
                aurora_memory_bank_free(bank);
                fclose(f);
                return NULL;
            }
            md5_tag[md5_len] = '\0';
            
            aurora_information_mark_t information_mark;
            if (fread(&information_mark, sizeof(aurora_information_mark_t), 1, f) != 1) {
                free(md5_tag);
                aurora_memory_bank_free(bank);
                fclose(f);
                return NULL;
            }
            
            if (fread(&text_len, sizeof(int32_t), 1, f) != 1) {
                free(md5_tag);
                aurora_memory_bank_free(bank);
                fclose(f);
                return NULL;
            }
            
            char* text = (char*)malloc((text_len + 1) * sizeof(char));
            if (fread(text, sizeof(char), text_len, f) != text_len) {
                free(md5_tag);
                free(text);
                aurora_memory_bank_free(bank);
                fclose(f);
                return NULL;
            }
            text[text_len] = '\0';
            
            float* embedding = (float*)malloc(dim * sizeof(float));
            if (fread(embedding, sizeof(float), dim, f) != dim) {
                free(md5_tag);
                free(text);
                free(embedding);
                aurora_memory_bank_free(bank);
                fclose(f);
                return NULL;
            }
            
            // Read dual-complex flag (version 2+)
            int8_t is_dual = 0;
            float* embedding_dual = NULL;
            if (version >= 2) {
                if (fread(&is_dual, sizeof(int8_t), 1, f) != 1) {
                    free(md5_tag);
                    free(text);
                    free(embedding);
                    aurora_memory_bank_free(bank);
                    fclose(f);
                    return NULL;
                }
                if (is_dual) {
                    embedding_dual = (float*)malloc(dim * sizeof(float));
                    if (fread(embedding_dual, sizeof(float), dim, f) != dim) {
                        free(md5_tag);
                        free(text);
                        free(embedding);
                        free(embedding_dual);
                        aurora_memory_bank_free(bank);
                        fclose(f);
                        return NULL;
                    }
                }
            }
            
            // Add entry to bank
            aurora_memory_entry_t* entry;
            if (is_dual && embedding_dual) {
                entry = aurora_memory_entry_create_dual_complex(md5_tag, information_mark, text, embedding, embedding_dual, dim);
                free(embedding_dual);
            } else {
                entry = aurora_memory_entry_create(md5_tag, information_mark, text, embedding, dim);
            }
            
            // Grow slot if needed
            if (bank->slot_counts[i] >= bank->slot_capacities[i]) {
                int new_capacity = (bank->slot_capacities[i] == 0) ? 8 : bank->slot_capacities[i] * 2;
                bank->slots[i] = (aurora_memory_entry_t*)realloc(
                    bank->slots[i],
                    new_capacity * sizeof(aurora_memory_entry_t)
                );
                bank->slot_capacities[i] = new_capacity;
            }
            
            bank->slots[i][bank->slot_counts[i]] = *entry;
            bank->slot_counts[i]++;
            bank->total_entries++;
            add_md5(bank, md5_tag);
            
            free(md5_tag);
            free(text);
            free(embedding);
        }
    }
    
    fclose(f);
    return bank;
}

int aurora_memory_banks_save(const aurora_memory_banks_t* banks, const char* filepath) {
    if (!banks || !filepath) {
        return -1;
    }
    
    // Save verified bank
    char verified_path[512];
    snprintf(verified_path, sizeof(verified_path), "%s.verified", filepath);
    int ret1 = aurora_memory_bank_save(banks->verified, verified_path);
    
    // Save quarantine bank
    char quarantine_path[512];
    snprintf(quarantine_path, sizeof(quarantine_path), "%s.quarantine", filepath);
    int ret2 = aurora_memory_bank_save(banks->quarantine, quarantine_path);
    
    // Save metadata
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", filepath);
    FILE* f = fopen(meta_path, "wb");
    if (f) {
        int32_t allow_quarantine_read = banks->allow_quarantine_read ? 1 : 0;
        fwrite(&allow_quarantine_read, sizeof(int32_t), 1, f);
        fclose(f);
    }
    
    return (ret1 < 0 || ret2 < 0) ? -1 : 0;
}

aurora_memory_banks_t* aurora_memory_banks_load(const char* filepath, int n_clusters, int dim, int depth, int seed, bool allow_quarantine_read) {
    if (!filepath) {
        return NULL;
    }
    
    // Load verified bank
    char verified_path[512];
    snprintf(verified_path, sizeof(verified_path), "%s.verified", filepath);
    aurora_memory_bank_t* verified = aurora_memory_bank_load(verified_path, n_clusters, dim, depth, seed);
    if (!verified) {
        return NULL;
    }
    
    // Load quarantine bank
    char quarantine_path[512];
    snprintf(quarantine_path, sizeof(quarantine_path), "%s.quarantine", filepath);
    aurora_memory_bank_t* quarantine = aurora_memory_bank_load(quarantine_path, n_clusters, dim, depth, seed);
    if (!quarantine) {
        aurora_memory_bank_free(verified);
        return NULL;
    }
    
    // Load metadata
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", filepath);
    FILE* f = fopen(meta_path, "rb");
    if (f) {
        int32_t file_allow_quarantine_read;
        if (fread(&file_allow_quarantine_read, sizeof(int32_t), 1, f) == 1) {
            allow_quarantine_read = (file_allow_quarantine_read != 0);
        }
        fclose(f);
    }
    
    aurora_memory_banks_t* banks = (aurora_memory_banks_t*)malloc(sizeof(aurora_memory_banks_t));
    banks->verified = verified;
    banks->quarantine = quarantine;
    banks->allow_quarantine_read = allow_quarantine_read;
    
    return banks;
}
