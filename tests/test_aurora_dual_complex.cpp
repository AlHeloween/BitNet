#include "aurora_memory_bank.h"
#include "ggml-aurora-memory.h"

#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>

// Test dual-complex memory entry creation
void test_dual_complex_entry_creation() {
    std::cout << "Testing dual-complex entry creation...\n";
    
    const int dim = 128;
    float* emb_primal = (float*)malloc(dim * sizeof(float));
    float* emb_dual = (float*)malloc(dim * sizeof(float));
    
    // Initialize test embeddings
    for (int i = 0; i < dim; i++) {
        emb_primal[i] = 0.1f * (float)i;
        emb_dual[i] = 0.01f * (float)i;
    }
    
    aurora_memory_entry_t* entry = aurora_memory_entry_create_dual_complex(
        "test_md5_12345678901234567890123456789012",
        AURORA_INFO_EXACT,
        "Test text entry",
        emb_primal,
        emb_dual,
        dim
    );
    
    assert(entry != NULL);
    assert(entry->is_dual_complex == true);
    assert(entry->embedding_dual != NULL);
    assert(entry->dim == dim);
    
    // Verify embeddings are stored correctly
    for (int i = 0; i < dim; i++) {
        assert(fabs(entry->embedding[i] - emb_primal[i]) < 1e-6f);
        assert(fabs(entry->embedding_dual[i] - emb_dual[i]) < 1e-6f);
    }
    
    aurora_memory_entry_free(entry);
    free(emb_primal);
    free(emb_dual);
    
    std::cout << "  ✓ Dual-complex entry creation works\n";
}

// Test dual-complex memory bank add
void test_dual_complex_bank_add() {
    std::cout << "Testing dual-complex bank add...\n";
    
    const int n_clusters = 64;
    const int dim = 128;
    const int depth = 3;
    const int seed = 1234;
    
    aurora_memory_bank_t* bank = aurora_memory_bank_create(n_clusters, dim, depth, seed);
    
    float* emb_primal = (float*)malloc(dim * sizeof(float));
    float* emb_dual = (float*)malloc(dim * sizeof(float));
    
    // Initialize test embeddings
    for (int i = 0; i < dim; i++) {
        emb_primal[i] = 0.1f * (float)i;
        emb_dual[i] = 0.01f * (float)i;
    }
    
    bool success = aurora_memory_bank_add_dual_complex(
        bank,
        "Test entry with dual-complex",
        AURORA_INFO_EXACT,
        emb_primal,
        emb_dual,
        dim,
        NULL
    );
    
    assert(success == true);
    
    int entries, slots_total, slots_non_empty;
    aurora_memory_bank_stats(bank, &entries, &slots_total, &slots_non_empty);
    assert(entries == 1);
    
    aurora_memory_bank_free(bank);
    free(emb_primal);
    free(emb_dual);
    
    std::cout << "  ✓ Dual-complex bank add works\n";
}

// Test dual-complex query
void test_dual_complex_query() {
    std::cout << "Testing dual-complex query...\n";
    
    const int n_clusters = 64;
    const int dim = 128;
    const int depth = 3;
    const int seed = 1234;
    
    aurora_memory_bank_t* bank = aurora_memory_bank_create(n_clusters, dim, depth, seed);
    
    // Add several entries with dual-complex
    for (int e = 0; e < 5; e++) {
        float* emb_primal = (float*)malloc(dim * sizeof(float));
        float* emb_dual = (float*)malloc(dim * sizeof(float));
        
        for (int i = 0; i < dim; i++) {
            emb_primal[i] = 0.1f * (float)(e * dim + i);
            emb_dual[i] = 0.01f * (float)(e * dim + i);
        }
        
        char text[256];
        snprintf(text, sizeof(text), "Entry %d", e);
        
        aurora_memory_bank_add_dual_complex(
            bank,
            text,
            AURORA_INFO_EXACT,
            emb_primal,
            emb_dual,
            dim,
            NULL
        );
        
        free(emb_primal);
        free(emb_dual);
    }
    
    // Query with dual-complex
    float* query_primal = (float*)malloc(dim * sizeof(float));
    float* query_dual = (float*)malloc(dim * sizeof(float));
    
    // Use similar embedding to first entry
    for (int i = 0; i < dim; i++) {
        query_primal[i] = 0.1f * (float)i;
        query_dual[i] = 0.01f * (float)i;
    }
    
    aurora_memory_entry_t** entries = (aurora_memory_entry_t**)malloc(3 * sizeof(aurora_memory_entry_t*));
    int count = aurora_memory_bank_query_dual_complex(
        bank,
        query_primal,
        query_dual,
        4,  // candidate_slots
        3,  // top_k
        0.1f,  // dual_weight
        entries
    );
    
    assert(count > 0);
    assert(count <= 3);
    
    // Verify retrieved entries have dual components
    for (int i = 0; i < count; i++) {
        assert(entries[i] != NULL);
        assert(entries[i]->is_dual_complex == true);
        assert(entries[i]->embedding_dual != NULL);
    }
    
    free(entries);
    free(query_primal);
    free(query_dual);
    aurora_memory_bank_free(bank);
    
    std::cout << "  ✓ Dual-complex query works\n";
}

// Test dual-complex persistence
void test_dual_complex_persistence() {
    std::cout << "Testing dual-complex persistence...\n";
    
    const int n_clusters = 64;
    const int dim = 128;
    const int depth = 3;
    const int seed = 1234;
    
    // Create bank and add dual-complex entry
    aurora_memory_bank_t* bank1 = aurora_memory_bank_create(n_clusters, dim, depth, seed);
    
    float* emb_primal = (float*)malloc(dim * sizeof(float));
    float* emb_dual = (float*)malloc(dim * sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        emb_primal[i] = 0.1f * (float)i;
        emb_dual[i] = 0.01f * (float)i;
    }
    
    aurora_memory_bank_add_dual_complex(
        bank1,
        "Persistent test entry",
        AURORA_INFO_EXACT,
        emb_primal,
        emb_dual,
        dim,
        NULL
    );
    
    // Save
    int ret = aurora_memory_bank_save(bank1, "test_dual_complex.bin");
    assert(ret == 0);
    
    // Load
    aurora_memory_bank_t* bank2 = aurora_memory_bank_load("test_dual_complex.bin", n_clusters, dim, depth, seed);
    assert(bank2 != NULL);
    
    int entries1, entries2;
    aurora_memory_bank_stats(bank1, &entries1, NULL, NULL);
    aurora_memory_bank_stats(bank2, &entries2, NULL, NULL);
    assert(entries1 == entries2);
    
    // Verify loaded entry has dual component
    // (We'd need to query to verify, but this tests the save/load path)
    
    aurora_memory_bank_free(bank1);
    aurora_memory_bank_free(bank2);
    free(emb_primal);
    free(emb_dual);
    
    std::cout << "  ✓ Dual-complex persistence works\n";
}

int main() {
    std::cout << "Running dual-complex memory tests...\n\n";
    
    test_dual_complex_entry_creation();
    test_dual_complex_bank_add();
    test_dual_complex_query();
    test_dual_complex_persistence();
    
    std::cout << "\nAll dual-complex tests passed!\n";
    return 0;
}
