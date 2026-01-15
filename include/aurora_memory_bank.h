#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Information mark enum (matches Python InformationMark)
typedef enum {
    AURORA_INFO_EXACT = 0,
    AURORA_INFO_INFERRED = 1,
    AURORA_INFO_HYPOTHETICAL = 2,
    AURORA_INFO_GUESS = 3,
    AURORA_INFO_UNKNOWN = 4
} aurora_information_mark_t;

// Forward declarations
typedef struct aurora_memory_entry aurora_memory_entry_t;
typedef struct aurora_memory_bank aurora_memory_bank_t;
typedef struct aurora_memory_banks aurora_memory_banks_t;

// Memory entry structure
struct aurora_memory_entry {
    char* md5_tag;              // MD5 hash (32 hex chars + null terminator)
    aurora_information_mark_t information_mark;
    char* text;                  // Text content (null-terminated)
    float* embedding;            // Embedding vector (dim elements) - primal component
    float* embedding_dual;       // Dual component (dim elements) - NULL if not dual-complex
    int dim;                     // Embedding dimension
    bool is_dual_complex;        // Whether this entry uses dual-complex (has embedding_dual)
};

// Memory bank structure
struct aurora_memory_bank {
    int n_clusters;              // Number of Sierpinski clusters
    int dim;                     // Embedding dimension
    int depth;                   // Sierpinski depth
    int seed;                    // Random seed
    float* centroids;            // Sierpinski centroids (n_clusters * dim)
    aurora_memory_entry_t** slots;  // Array of entry lists per cluster
    int* slot_counts;           // Number of entries per slot
    int* slot_capacities;        // Capacity per slot
    int total_entries;            // Total number of entries
    char** seen_md5;             // MD5 deduplication set
    int seen_md5_count;          // Number of unique MD5s
    int seen_md5_capacity;        // Capacity of MD5 set
};

// Dual memory banks (verified/quarantine)
struct aurora_memory_banks {
    aurora_memory_bank_t* verified;
    aurora_memory_bank_t* quarantine;
    bool allow_quarantine_read;
};

// API Functions

// Memory entry operations
aurora_memory_entry_t* aurora_memory_entry_create(
    const char* md5_tag,
    aurora_information_mark_t information_mark,
    const char* text,
    const float* embedding,
    int dim
);
// Create entry with dual-complex support
aurora_memory_entry_t* aurora_memory_entry_create_dual_complex(
    const char* md5_tag,
    aurora_information_mark_t information_mark,
    const char* text,
    const float* embedding_primal,
    const float* embedding_dual,
    int dim
);
void aurora_memory_entry_free(aurora_memory_entry_t* entry);

// Memory bank operations
aurora_memory_bank_t* aurora_memory_bank_create(
    int n_clusters,
    int dim,
    int depth,
    int seed
);
void aurora_memory_bank_free(aurora_memory_bank_t* bank);

// Add entry to bank (returns true if added, false if duplicate)
bool aurora_memory_bank_add(
    aurora_memory_bank_t* bank,
    const char* text,
    aurora_information_mark_t information_mark,
    const char* embed_text,  // Optional: different text for embedding
    const char* md5_tag       // Optional: precomputed MD5
);
// Add entry with dual-complex embeddings
bool aurora_memory_bank_add_dual_complex(
    aurora_memory_bank_t* bank,
    const char* text,
    aurora_information_mark_t information_mark,
    const float* embedding_primal,
    const float* embedding_dual,
    int dim,
    const char* md5_tag
);

// Query bank (returns number of entries found, fills entries array)
int aurora_memory_bank_query(
    const aurora_memory_bank_t* bank,
    const char* query_text,
    int candidate_slots,
    int top_k,
    aurora_memory_entry_t** entries_out  // Pre-allocated array of size top_k
);
// Query bank using embedding (returns number of entries found)
int aurora_memory_bank_query_embedding(
    const aurora_memory_bank_t* bank,
    const float* query_embedding,
    int candidate_slots,
    int top_k,
    aurora_memory_entry_t** entries_out
);
// Query bank using dual-complex embeddings with dual-weighted similarity
int aurora_memory_bank_query_dual_complex(
    const aurora_memory_bank_t* bank,
    const float* query_primal,
    const float* query_dual,  // Can be NULL
    int candidate_slots,
    int top_k,
    float dual_weight,  // Weight for dual component in similarity (default: 0.1)
    aurora_memory_entry_t** entries_out
);

// Get bank statistics
void aurora_memory_bank_stats(
    const aurora_memory_bank_t* bank,
    int* entries_out,
    int* slots_total_out,
    int* slots_non_empty_out
);

// Dual banks operations
aurora_memory_banks_t* aurora_memory_banks_create(
    int n_clusters,
    int dim,
    int depth,
    int seed,
    bool allow_quarantine_read
);
void aurora_memory_banks_free(aurora_memory_banks_t* banks);

// Add entry to dual banks (routes based on information mark)
bool aurora_memory_banks_add(
    aurora_memory_banks_t* banks,
    const char* text,
    aurora_information_mark_t information_mark,
    const char* embed_text,
    const char* md5_tag
);
// Add entry with dual-complex embeddings
bool aurora_memory_banks_add_dual_complex(
    aurora_memory_banks_t* banks,
    const char* text,
    aurora_information_mark_t information_mark,
    const float* embedding_primal,
    const float* embedding_dual,
    int dim,
    const char* md5_tag
);

// Query dual banks (searches verified first, then quarantine if allowed)
int aurora_memory_banks_query(
    const aurora_memory_banks_t* banks,
    const char* query_text,
    int candidate_slots,
    int top_k,
    aurora_memory_entry_t** entries_out
);

// Memory persistence: save/load banks to/from file
// File format: binary (simple format for now)
// Returns: 0 on success, negative on error
int aurora_memory_bank_save(const aurora_memory_bank_t* bank, const char* filepath);
aurora_memory_bank_t* aurora_memory_bank_load(const char* filepath, int n_clusters, int dim, int depth, int seed);

int aurora_memory_banks_save(const aurora_memory_banks_t* banks, const char* filepath);
aurora_memory_banks_t* aurora_memory_banks_load(const char* filepath, int n_clusters, int dim, int depth, int seed, bool allow_quarantine_read);

// Utility functions
void aurora_md5_hex(const char* text, char* md5_out);  // 32 chars + null
void aurora_hash_embed(const char* text, int dim, int seed, float* embedding_out);

#ifdef __cplusplus
}
#endif
