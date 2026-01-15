// Test suite for Aurora memory bank C++ implementation
// Compile with: g++ -std=c++11 -I../include -I../3rdparty/llama.cpp/ggml/include test_aurora_memory_cpp.cpp ../src/aurora_memory_bank.cpp -o test_aurora_memory_cpp

#include "aurora_memory_bank.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "TEST FAILED: %s\n", msg); \
            exit(1); \
        } \
    } while(0)

void test_memory_bank_create() {
    printf("Testing memory bank creation...\n");
    
    aurora_memory_bank_t* bank = aurora_memory_bank_create(16, 64, 1, 1234);
    TEST_ASSERT(bank != NULL, "Bank creation failed");
    
    int entries, slots_total, slots_non_empty;
    aurora_memory_bank_stats(bank, &entries, &slots_total, &slots_non_empty);
    TEST_ASSERT(entries == 0, "New bank should have 0 entries");
    TEST_ASSERT(slots_total == 16, "Bank should have 16 slots");
    TEST_ASSERT(slots_non_empty == 0, "New bank should have 0 non-empty slots");
    
    aurora_memory_bank_free(bank);
    printf("  ✓ Memory bank creation test passed\n");
}

void test_memory_bank_add() {
    printf("Testing memory bank add operations...\n");
    
    aurora_memory_bank_t* bank = aurora_memory_bank_create(16, 64, 1, 1234);
    
    // Add first entry
    bool added1 = aurora_memory_bank_add(
        bank,
        "Test entry 1",
        AURORA_INFO_EXACT,
        NULL,
        NULL
    );
    TEST_ASSERT(added1 == true, "First add should succeed");
    
    int entries;
    aurora_memory_bank_stats(bank, &entries, NULL, NULL);
    TEST_ASSERT(entries == 1, "Bank should have 1 entry");
    
    // Try to add duplicate (same text)
    bool added2 = aurora_memory_bank_add(
        bank,
        "Test entry 1",
        AURORA_INFO_EXACT,
        NULL,
        NULL
    );
    TEST_ASSERT(added2 == false, "Duplicate add should fail");
    
    aurora_memory_bank_stats(bank, &entries, NULL, NULL);
    TEST_ASSERT(entries == 1, "Bank should still have 1 entry");
    
    // Add second entry
    bool added3 = aurora_memory_bank_add(
        bank,
        "Test entry 2",
        AURORA_INFO_INFERRED,
        NULL,
        NULL
    );
    TEST_ASSERT(added3 == true, "Second add should succeed");
    
    aurora_memory_bank_stats(bank, &entries, NULL, NULL);
    TEST_ASSERT(entries == 2, "Bank should have 2 entries");
    
    aurora_memory_bank_free(bank);
    printf("  ✓ Memory bank add test passed\n");
}

void test_memory_bank_query() {
    printf("Testing memory bank query operations...\n");
    
    aurora_memory_bank_t* bank = aurora_memory_bank_create(16, 64, 1, 1234);
    
    // Add several entries
    aurora_memory_bank_add(bank, "Python programming language", AURORA_INFO_EXACT, NULL, NULL);
    aurora_memory_bank_add(bank, "Machine learning algorithms", AURORA_INFO_INFERRED, NULL, NULL);
    aurora_memory_bank_add(bank, "Neural network architecture", AURORA_INFO_EXACT, NULL, NULL);
    aurora_memory_bank_add(bank, "Deep learning models", AURORA_INFO_INFERRED, NULL, NULL);
    
    // Query for similar content
    aurora_memory_entry_t* entries[4];
    int count = aurora_memory_bank_query(bank, "programming", 4, 2, entries);
    TEST_ASSERT(count >= 0 && count <= 2, "Query should return at most 2 entries");
    
    // Query should find at least one entry
    if (count > 0) {
        TEST_ASSERT(entries[0] != NULL, "First entry should not be NULL");
        TEST_ASSERT(entries[0]->text != NULL, "Entry text should not be NULL");
    }
    
    aurora_memory_bank_free(bank);
    printf("  ✓ Memory bank query test passed\n");
}

void test_memory_banks_adid_routing() {
    printf("Testing ADID routing in dual banks...\n");
    
    aurora_memory_banks_t* banks = aurora_memory_banks_create(16, 64, 1, 1234, false);
    
    // EXACT should go to verified
    bool added1 = aurora_memory_banks_add(
        banks,
        "Exact information",
        AURORA_INFO_EXACT,
        NULL,
        NULL
    );
    TEST_ASSERT(added1 == true, "EXACT entry should be added");
    
    int verified_entries, quarantine_entries;
    aurora_memory_bank_stats(banks->verified, &verified_entries, NULL, NULL);
    aurora_memory_bank_stats(banks->quarantine, &quarantine_entries, NULL, NULL);
    TEST_ASSERT(verified_entries == 1, "Verified bank should have 1 entry");
    TEST_ASSERT(quarantine_entries == 0, "Quarantine bank should have 0 entries");
    
    // INFERRED should go to verified
    bool added2 = aurora_memory_banks_add(
        banks,
        "Inferred information",
        AURORA_INFO_INFERRED,
        NULL,
        NULL
    );
    TEST_ASSERT(added2 == true, "INFERRED entry should be added");
    
    aurora_memory_bank_stats(banks->verified, &verified_entries, NULL, NULL);
    aurora_memory_bank_stats(banks->quarantine, &quarantine_entries, NULL, NULL);
    TEST_ASSERT(verified_entries == 2, "Verified bank should have 2 entries");
    TEST_ASSERT(quarantine_entries == 0, "Quarantine bank should still have 0 entries");
    
    // HYPOTHETICAL should go to quarantine
    bool added3 = aurora_memory_banks_add(
        banks,
        "Hypothetical information",
        AURORA_INFO_HYPOTHETICAL,
        NULL,
        NULL
    );
    TEST_ASSERT(added3 == true, "HYPOTHETICAL entry should be added");
    
    aurora_memory_bank_stats(banks->verified, &verified_entries, NULL, NULL);
    aurora_memory_bank_stats(banks->quarantine, &quarantine_entries, NULL, NULL);
    TEST_ASSERT(verified_entries == 2, "Verified bank should still have 2 entries");
    TEST_ASSERT(quarantine_entries == 1, "Quarantine bank should have 1 entry");
    
    aurora_memory_banks_free(banks);
    printf("  ✓ ADID routing test passed\n");
}

void test_memory_persistence() {
    printf("Testing memory persistence...\n");
    
    aurora_memory_bank_t* bank1 = aurora_memory_bank_create(16, 64, 1, 1234);
    
    // Add some entries
    aurora_memory_bank_add(bank1, "Entry 1", AURORA_INFO_EXACT, NULL, NULL);
    aurora_memory_bank_add(bank1, "Entry 2", AURORA_INFO_INFERRED, NULL, NULL);
    aurora_memory_bank_add(bank1, "Entry 3", AURORA_INFO_HYPOTHETICAL, NULL, NULL);
    
    int entries1;
    aurora_memory_bank_stats(bank1, &entries1, NULL, NULL);
    TEST_ASSERT(entries1 == 3, "Bank1 should have 3 entries");
    
    // Save bank
    int save_result = aurora_memory_bank_save(bank1, "test_memory.bin");
    TEST_ASSERT(save_result == 0, "Save should succeed");
    
    aurora_memory_bank_free(bank1);
    
    // Load bank
    aurora_memory_bank_t* bank2 = aurora_memory_bank_load("test_memory.bin", 16, 64, 1, 1234);
    TEST_ASSERT(bank2 != NULL, "Load should succeed");
    
    int entries2;
    aurora_memory_bank_stats(bank2, &entries2, NULL, NULL);
    TEST_ASSERT(entries2 == 3, "Bank2 should have 3 entries");
    
    // Query to verify entries
    aurora_memory_entry_t* query_entries[3];
    int query_count = aurora_memory_bank_query(bank2, "Entry", 4, 3, query_entries);
    TEST_ASSERT(query_count >= 0, "Query should succeed");
    
    aurora_memory_bank_free(bank2);
    
    // Cleanup
    remove("test_memory.bin");
    
    printf("  ✓ Memory persistence test passed\n");
}

int main() {
    printf("Running Aurora Memory Bank C++ Tests\n");
    printf("=====================================\n\n");
    
    test_memory_bank_create();
    test_memory_bank_add();
    test_memory_bank_query();
    test_memory_banks_adid_routing();
    test_memory_persistence();
    
    printf("\n=====================================\n");
    printf("All tests passed!\n");
    return 0;
}
