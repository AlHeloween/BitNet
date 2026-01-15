#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// TKG (Temporal Knowledge Graph) dataset structures and operations
// For Stage 2: Real dataset track integration

// TKG example: (subject, relation, object, time)
typedef struct {
    int32_t s;      // Subject entity ID
    int32_t r;      // Relation ID
    int32_t o;      // Object entity ID
    int32_t t;      // Time ID
    float tau;      // Normalized time scalar [0, 1]
} tkg_example_t;

// TKG dataset structure
typedef struct {
    int32_t n_entities;
    int32_t n_relations;
    int32_t n_times;
    int32_t n_examples;
    tkg_example_t* examples;  // Array of examples
    char** entity_names;      // Array of entity name strings (n_entities)
    char** relation_names;    // Array of relation name strings (n_relations)
    char** time_names;        // Array of time name strings (n_times)
} tkg_dataset_t;

// Load TKG dataset from TSV file
// Returns NULL on failure
tkg_dataset_t* tkg_dataset_load_tsv(
    const char* path,
    const char delimiter  // '\t' for tab, ' ' for space
);

// Load TKG dataset from split directory (train.tsv, valid.tsv, test.tsv)
// Returns NULL on failure
// Outputs: train, valid, test datasets (caller must free)
int tkg_dataset_load_splits(
    const char* dir_path,
    const char delimiter,
    tkg_dataset_t** train_out,
    tkg_dataset_t** valid_out,
    tkg_dataset_t** test_out
);

// Free TKG dataset
void tkg_dataset_free(tkg_dataset_t* dataset);

// Get entity ID by name (returns -1 if not found)
int32_t tkg_dataset_entity_id(const tkg_dataset_t* dataset, const char* entity_name);

// Get relation ID by name (returns -1 if not found)
int32_t tkg_dataset_relation_id(const tkg_dataset_t* dataset, const char* relation_name);

// Get time ID by name (returns -1 if not found)
int32_t tkg_dataset_time_id(const tkg_dataset_t* dataset, const char* time_name);

// Normalize time values to [0, 1] range
// Modifies tau values in-place
void tkg_dataset_normalize_time(tkg_dataset_t* dataset);

#ifdef __cplusplus
}
#endif
