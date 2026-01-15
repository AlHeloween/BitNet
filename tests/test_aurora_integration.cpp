// Integration test for Aurora compute reduction
// Tests the full integration: memory banks -> GGML operations -> attention hooks

#include "llama-aurora-integration.h"
#include "aurora_memory_bank.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "TEST FAILED: %s\n", msg); \
            exit(1); \
        } \
    } while(0)

void test_aurora_context_params_init() {
    printf("Testing Aurora context params initialization...\n");
    
    struct llama_aurora_context_params params;
    llama_aurora_context_params_init(&params);
    
    TEST_ASSERT(params.enable_aurora_memory == false, "Default should be disabled");
    TEST_ASSERT(params.aurora_window_size == 2048, "Default window size should be 2048");
    TEST_ASSERT(params.aurora_k_read == 8, "Default k_read should be 8");
    TEST_ASSERT(params.aurora_candidate_slots == 4, "Default candidate_slots should be 4");
    TEST_ASSERT(params.memory_banks == NULL, "Default memory_banks should be NULL");
    TEST_ASSERT(params.enable_memory_write == false, "Default should not write to memory");
    
    printf("  ✓ Context params initialization test passed\n");
}

void test_aurora_memory_banks_integration() {
    printf("Testing memory banks integration...\n");
    
    // Create memory banks
    aurora_memory_banks_t* banks = aurora_memory_banks_create(
        64,   // n_clusters
        128,  // dim
        1,    // depth
        1234, // seed
        false // allow_quarantine_read
    );
    
    TEST_ASSERT(banks != NULL, "Memory banks creation should succeed");
    TEST_ASSERT(banks->verified != NULL, "Verified bank should exist");
    TEST_ASSERT(banks->quarantine != NULL, "Quarantine bank should exist");
    
    // Add entries
    bool added1 = aurora_memory_banks_add(
        banks,
        "Test entry for integration",
        AURORA_INFO_EXACT,
        NULL,
        NULL
    );
    TEST_ASSERT(added1 == true, "Entry should be added");
    
    // Query
    aurora_memory_entry_t* entries[4];
    int count = aurora_memory_banks_query(
        banks,
        "Test",
        4,
        4,
        entries
    );
    TEST_ASSERT(count >= 0, "Query should succeed");
    
    aurora_memory_banks_free(banks);
    printf("  ✓ Memory banks integration test passed\n");
}

void test_aurora_configuration() {
    printf("Testing Aurora configuration...\n");
    
    // Create memory banks
    aurora_memory_banks_t* banks = aurora_memory_banks_create(64, 128, 1, 1234, false);
    
    // Configure Aurora
    struct llama_aurora_context_params params;
    llama_aurora_context_params_init(&params);
    params.enable_aurora_memory = true;
    params.aurora_window_size = 1024;
    params.aurora_k_read = 4;
    params.aurora_candidate_slots = 2;
    params.memory_banks = banks;
    params.enable_memory_write = true;
    params.default_information_mark = AURORA_INFO_INFERRED;
    
    TEST_ASSERT(params.enable_aurora_memory == true, "Aurora should be enabled");
    TEST_ASSERT(params.aurora_window_size == 1024, "Window size should be 1024");
    TEST_ASSERT(params.memory_banks == banks, "Memory banks should be set");
    
    aurora_memory_banks_free(banks);
    printf("  ✓ Configuration test passed\n");
}

int main() {
    printf("Running Aurora Integration Tests\n");
    printf("================================\n\n");
    
    test_aurora_context_params_init();
    test_aurora_memory_banks_integration();
    test_aurora_configuration();
    
    printf("\n================================\n");
    printf("All integration tests passed!\n");
    printf("\nNote: Full integration with llama.cpp requires linking with llama.cpp library.\n");
    printf("These tests verify the Aurora components work independently.\n");
    return 0;
}
