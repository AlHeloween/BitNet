#pragma once

#include "ggml.h"
#include "ggml-lattice-addressing.h"

#ifdef __cplusplus
extern "C" {
#endif

// Persistent storage using Sierpinski lattice addressing

// Memory entry stored in lattice
struct lattice_memory_entry {
    float embedding[8];  // Dual quaternion [8]
    char* text;           // Optional text (NULL if not set)
    void* metadata;       // Optional metadata (NULL if not set)
};

// Lattice memory storage
struct lattice_memory_storage {
    void* storage;  // Internal storage (dict: address -> entries)
    int n_levels;
    int n_per_level;
};

// Initialize lattice storage
struct lattice_memory_storage* ggml_lattice_storage_init(
    int n_levels,
    int n_per_level
);

// Free lattice storage
void ggml_lattice_storage_free(struct lattice_memory_storage* storage);

// Write entry to leaf node
bool ggml_lattice_storage_write_leaf(
    struct lattice_memory_storage* storage,
    const struct lattice_address* addr,
    const struct lattice_memory_entry* entry
);

// Read entries from leaf node
// Returns number of entries found
int ggml_lattice_storage_read_leaf(
    struct lattice_memory_storage* storage,
    const struct lattice_address* addr,
    struct lattice_memory_entry* entries_out,
    int max_entries
);

// Check if leaf exists
bool ggml_lattice_storage_has_leaf(
    struct lattice_memory_storage* storage,
    const struct lattice_address* addr
);

// Save storage to file
bool ggml_lattice_storage_save(
    struct lattice_memory_storage* storage,
    const char* path
);

// Load storage from file
bool ggml_lattice_storage_load(
    struct lattice_memory_storage* storage,
    const char* path
);

// Get statistics
void ggml_lattice_storage_get_stats(
    struct lattice_memory_storage* storage,
    int* total_addresses_out,
    int* total_entries_out
);

#ifdef __cplusplus
}
#endif
