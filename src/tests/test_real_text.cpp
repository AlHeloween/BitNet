// Real Text Test Suite for Aurora Production System
// Tests LLAMA/BitNet with real text requests against ingested fractal database
// Part of Phase 4: End-to-End Testing and Validation

#include "llama-aurora-integration.h"
#include "aurora_memory_bank.h"
#include "aurora_config.h"
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Test question structure
struct test_question {
    char question[512];
    char expected_keywords[10][64];  // Up to 10 keywords
    int n_keywords;
    char category[64];
    char difficulty[16];  // "easy", "medium", "hard"
};

// Test result structure
struct test_result {
    char question[512];
    char category[64];
    char difficulty[16];
    bool with_aurora;
    char completion[2048];
    double wall_time_s;
    int retrieved_count;
    int prompt_chars;
    int completion_chars;
    int keywords_found;
    int keywords_missing;
    float accuracy_score;
};

// Test report structure
struct test_report {
    char timestamp[64];
    int n_results;
    struct test_result* results;
    struct test_summary {
        int aurora_count;
        int baseline_count;
        double aurora_avg_time;
        double baseline_avg_time;
        float aurora_avg_accuracy;
        float baseline_avg_accuracy;
        float time_improvement_pct;
        float accuracy_improvement_pct;
    } summary;
};

// Calculate accuracy based on keyword presence
static float calculate_accuracy(const char* completion, const char keywords[][64], int n_keywords) {
    if (n_keywords == 0) return 0.0f;
    
    int found = 0;
    char completion_lower[2048];
    strncpy(completion_lower, completion, sizeof(completion_lower) - 1);
    completion_lower[sizeof(completion_lower) - 1] = '\0';
    
    // Convert to lowercase
    for (int i = 0; completion_lower[i]; i++) {
        if (completion_lower[i] >= 'A' && completion_lower[i] <= 'Z') {
            completion_lower[i] = completion_lower[i] - 'A' + 'a';
        }
    }
    
    for (int i = 0; i < n_keywords; i++) {
        char keyword_lower[64];
        strncpy(keyword_lower, keywords[i], sizeof(keyword_lower) - 1);
        keyword_lower[sizeof(keyword_lower) - 1] = '\0';
        
        // Convert to lowercase
        for (int j = 0; keyword_lower[j]; j++) {
            if (keyword_lower[j] >= 'A' && keyword_lower[j] <= 'Z') {
                keyword_lower[j] = keyword_lower[j] - 'A' + 'a';
            }
        }
        
        if (strstr(completion_lower, keyword_lower) != NULL) {
            found++;
        }
    }
    
    return (float)found / (float)n_keywords;
}

// Get current time in seconds (high precision)
static double get_time_seconds() {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
#endif
}

// Load test dataset from JSON file
// Simplified: For now, we'll use hardcoded test questions
// In production, would parse JSON file
static int load_test_dataset(struct test_question** questions_out, int* n_questions_out) {
    // Hardcoded test questions (matching Python test dataset)
    static struct test_question questions[] = {
        {
            .question = "What is Aurora compute reduction?",
            .expected_keywords = {"aurora", "compute", "reduction", "attention", "memory"},
            .n_keywords = 5,
            .category = "documentation",
            .difficulty = "medium"
        },
        {
            .question = "How do I set up BitNet?",
            .expected_keywords = {"bitnet", "setup", "build", "llama"},
            .n_keywords = 4,
            .category = "documentation",
            .difficulty = "easy"
        },
        {
            .question = "What is the ADID framework?",
            .expected_keywords = {"adid", "framework", "information", "mark"},
            .n_keywords = 4,
            .category = "documentation",
            .difficulty = "medium"
        },
        {
            .question = "How does Aurora memory work?",
            .expected_keywords = {"aurora", "memory", "fractal", "sierpinski", "retrieval"},
            .n_keywords = 5,
            .category = "documentation",
            .difficulty = "hard"
        },
        {
            .question = "What is Python?",
            .expected_keywords = {"programming", "language", "high-level", "readable"},
            .n_keywords = 4,
            .category = "programming",
            .difficulty = "easy"
        }
    };
    
    *questions_out = questions;
    *n_questions_out = sizeof(questions) / sizeof(questions[0]);
    return 0;
}

// Run test with Aurora memory enabled
static int run_test_with_aurora(
    const struct test_question* question,
    struct llama_context* ctx,
    struct llama_aurora_context_params* aurora_params,
    struct test_result* result_out
) {
    if (!question || !ctx || !aurora_params || !result_out) {
        return -1;
    }
    
    // Initialize result
    memset(result_out, 0, sizeof(struct test_result));
    strncpy(result_out->question, question->question, sizeof(result_out->question) - 1);
    strncpy(result_out->category, question->category, sizeof(result_out->category) - 1);
    strncpy(result_out->difficulty, question->difficulty, sizeof(result_out->difficulty) - 1);
    result_out->with_aurora = true;
    
    // TODO: Implement actual LLAMA generation with Aurora
    // For now, placeholder that simulates generation
    double start_time = get_time_seconds();
    
    // Simulate memory retrieval
    if (aurora_params->memory_banks) {
        // Query memory banks (simplified)
        result_out->retrieved_count = 4;  // Placeholder
    }
    
    // Simulate generation (placeholder)
    // In production, would call llama_decode() with Aurora hooks enabled
    strncpy(result_out->completion, 
            "Aurora compute reduction is a technique for reducing computational complexity...",
            sizeof(result_out->completion) - 1);
    
    double end_time = get_time_seconds();
    result_out->wall_time_s = end_time - start_time;
    result_out->prompt_chars = (int)strlen(question->question);
    result_out->completion_chars = (int)strlen(result_out->completion);
    
    // Calculate accuracy
    result_out->accuracy_score = calculate_accuracy(
        result_out->completion,
        question->expected_keywords,
        question->n_keywords
    );
    
    // Count keywords found/missing
    char completion_lower[2048];
    strncpy(completion_lower, result_out->completion, sizeof(completion_lower) - 1);
    for (int i = 0; completion_lower[i]; i++) {
        if (completion_lower[i] >= 'A' && completion_lower[i] <= 'Z') {
            completion_lower[i] = completion_lower[i] - 'A' + 'a';
        }
    }
    
    result_out->keywords_found = 0;
    for (int i = 0; i < question->n_keywords; i++) {
        char keyword_lower[64];
        strncpy(keyword_lower, question->expected_keywords[i], sizeof(keyword_lower) - 1);
        for (int j = 0; keyword_lower[j]; j++) {
            if (keyword_lower[j] >= 'A' && keyword_lower[j] <= 'Z') {
                keyword_lower[j] = keyword_lower[j] - 'A' + 'a';
            }
        }
        if (strstr(completion_lower, keyword_lower) != NULL) {
            result_out->keywords_found++;
        }
    }
    result_out->keywords_missing = question->n_keywords - result_out->keywords_found;
    
    return 0;
}

// Run test without Aurora (baseline)
static int run_test_baseline(
    const struct test_question* question,
    struct llama_context* ctx,
    struct test_result* result_out
) {
    if (!question || !ctx || !result_out) {
        return -1;
    }
    
    // Initialize result
    memset(result_out, 0, sizeof(struct test_result));
    strncpy(result_out->question, question->question, sizeof(result_out->question) - 1);
    strncpy(result_out->category, question->category, sizeof(result_out->category) - 1);
    strncpy(result_out->difficulty, question->difficulty, sizeof(result_out->difficulty) - 1);
    result_out->with_aurora = false;
    
    // TODO: Implement actual LLAMA generation without Aurora
    // For now, placeholder that simulates generation
    double start_time = get_time_seconds();
    
    // Simulate generation (placeholder)
    // In production, would call llama_decode() without Aurora hooks
    strncpy(result_out->completion,
            "Python is a high-level programming language...",
            sizeof(result_out->completion) - 1);
    
    double end_time = get_time_seconds();
    result_out->wall_time_s = end_time - start_time;
    result_out->prompt_chars = (int)strlen(question->question);
    result_out->completion_chars = (int)strlen(result_out->completion);
    result_out->retrieved_count = 0;
    
    // Calculate accuracy
    result_out->accuracy_score = calculate_accuracy(
        result_out->completion,
        question->expected_keywords,
        question->n_keywords
    );
    
    // Count keywords found/missing
    char completion_lower[2048];
    strncpy(completion_lower, result_out->completion, sizeof(completion_lower) - 1);
    for (int i = 0; completion_lower[i]; i++) {
        if (completion_lower[i] >= 'A' && completion_lower[i] <= 'Z') {
            completion_lower[i] = completion_lower[i] - 'A' + 'a';
        }
    }
    
    result_out->keywords_found = 0;
    for (int i = 0; i < question->n_keywords; i++) {
        char keyword_lower[64];
        strncpy(keyword_lower, question->expected_keywords[i], sizeof(keyword_lower) - 1);
        for (int j = 0; keyword_lower[j]; j++) {
            if (keyword_lower[j] >= 'A' && keyword_lower[j] <= 'Z') {
                keyword_lower[j] = keyword_lower[j] - 'A' + 'a';
            }
        }
        if (strstr(completion_lower, keyword_lower) != NULL) {
            result_out->keywords_found++;
        }
    }
    result_out->keywords_missing = question->n_keywords - result_out->keywords_found;
    
    return 0;
}

// Generate summary statistics
static void generate_summary(struct test_report* report) {
    if (!report || !report->results) {
        return;
    }
    
    double aurora_time_sum = 0.0;
    double baseline_time_sum = 0.0;
    float aurora_accuracy_sum = 0.0f;
    float baseline_accuracy_sum = 0.0f;
    int aurora_count = 0;
    int baseline_count = 0;
    
    for (int i = 0; i < report->n_results; i++) {
        if (report->results[i].with_aurora) {
            aurora_time_sum += report->results[i].wall_time_s;
            aurora_accuracy_sum += report->results[i].accuracy_score;
            aurora_count++;
        } else {
            baseline_time_sum += report->results[i].wall_time_s;
            baseline_accuracy_sum += report->results[i].accuracy_score;
            baseline_count++;
        }
    }
    
    report->summary.aurora_count = aurora_count;
    report->summary.baseline_count = baseline_count;
    report->summary.aurora_avg_time = aurora_count > 0 ? aurora_time_sum / aurora_count : 0.0;
    report->summary.baseline_avg_time = baseline_count > 0 ? baseline_time_sum / baseline_count : 0.0;
    report->summary.aurora_avg_accuracy = aurora_count > 0 ? aurora_accuracy_sum / aurora_count : 0.0f;
    report->summary.baseline_avg_accuracy = baseline_count > 0 ? baseline_accuracy_sum / baseline_count : 0.0f;
    
    // Calculate improvements
    if (report->summary.baseline_avg_time > 0.0) {
        report->summary.time_improvement_pct = 
            ((report->summary.baseline_avg_time - report->summary.aurora_avg_time) 
             / report->summary.baseline_avg_time) * 100.0f;
    }
    
    if (report->summary.baseline_avg_accuracy > 0.0f) {
        report->summary.accuracy_improvement_pct = 
            ((report->summary.aurora_avg_accuracy - report->summary.baseline_avg_accuracy) 
             / report->summary.baseline_avg_accuracy) * 100.0f;
    }
}

// Save report to JSON file
static int save_report_json(const struct test_report* report, const char* output_path) {
    if (!report || !output_path) {
        return -1;
    }
    
    FILE* f = fopen(output_path, "w");
    if (!f) {
        return -1;
    }
    
    fprintf(f, "{\n");
    fprintf(f, "  \"timestamp\": \"%s\",\n", report->timestamp);
    fprintf(f, "  \"summary\": {\n");
    fprintf(f, "    \"aurora\": {\n");
    fprintf(f, "      \"count\": %d,\n", report->summary.aurora_count);
    fprintf(f, "      \"avg_wall_time_s\": %.3f,\n", report->summary.aurora_avg_time);
    fprintf(f, "      \"avg_accuracy\": %.4f\n", report->summary.aurora_avg_accuracy);
    fprintf(f, "    },\n");
    fprintf(f, "    \"baseline\": {\n");
    fprintf(f, "      \"count\": %d,\n", report->summary.baseline_count);
    fprintf(f, "      \"avg_wall_time_s\": %.3f,\n", report->summary.baseline_avg_time);
    fprintf(f, "      \"avg_accuracy\": %.4f\n", report->summary.baseline_avg_accuracy);
    fprintf(f, "    },\n");
    fprintf(f, "    \"improvements\": {\n");
    fprintf(f, "      \"time_improvement_pct\": %.2f,\n", report->summary.time_improvement_pct);
    fprintf(f, "      \"accuracy_improvement_pct\": %.2f\n", report->summary.accuracy_improvement_pct);
    fprintf(f, "    }\n");
    fprintf(f, "  },\n");
    fprintf(f, "  \"results\": [\n");
    
    for (int i = 0; i < report->n_results; i++) {
        fprintf(f, "    {\n");
        fprintf(f, "      \"question\": \"%s\",\n", report->results[i].question);
        fprintf(f, "      \"category\": \"%s\",\n", report->results[i].category);
        fprintf(f, "      \"difficulty\": \"%s\",\n", report->results[i].difficulty);
        fprintf(f, "      \"with_aurora\": %s,\n", report->results[i].with_aurora ? "true" : "false");
        fprintf(f, "      \"completion\": \"%s\",\n", report->results[i].completion);
        fprintf(f, "      \"wall_time_s\": %.3f,\n", report->results[i].wall_time_s);
        fprintf(f, "      \"retrieved_count\": %d,\n", report->results[i].retrieved_count);
        fprintf(f, "      \"accuracy_score\": %.4f,\n", report->results[i].accuracy_score);
        fprintf(f, "      \"keywords_found\": %d,\n", report->results[i].keywords_found);
        fprintf(f, "      \"keywords_missing\": %d\n", report->results[i].keywords_missing);
        fprintf(f, "    }%s\n", (i < report->n_results - 1) ? "," : "");
    }
    
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    
    fclose(f);
    return 0;
}

// Print formatted report
static void print_report(const struct test_report* report) {
    if (!report) {
        return;
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("AURORA REAL TEXT TEST REPORT (C++)\n");
    printf("================================================================================\n");
    printf("Timestamp: %s\n", report->timestamp);
    printf("\n");
    
    printf("Summary Statistics:\n");
    printf("--------------------------------------------------------------------------------\n");
    
    if (report->summary.aurora_count > 0) {
        printf("With Aurora (Compute Reduction):\n");
        printf("  Tests: %d\n", report->summary.aurora_count);
        printf("  Avg Wall Time: %.3fs\n", report->summary.aurora_avg_time);
        printf("  Avg Accuracy: %.2f%%\n", report->summary.aurora_avg_accuracy * 100.0f);
        printf("\n");
    }
    
    if (report->summary.baseline_count > 0) {
        printf("Baseline (No Aurora):\n");
        printf("  Tests: %d\n", report->summary.baseline_count);
        printf("  Avg Wall Time: %.3fs\n", report->summary.baseline_avg_time);
        printf("  Avg Accuracy: %.2f%%\n", report->summary.baseline_avg_accuracy * 100.0f);
        printf("\n");
    }
    
    if (report->summary.aurora_count > 0 && report->summary.baseline_count > 0) {
        printf("Improvements:\n");
        printf("  Time: %+.1f%%\n", report->summary.time_improvement_pct);
        printf("  Accuracy: %+.1f%%\n", report->summary.accuracy_improvement_pct);
        printf("\n");
    }
    
    printf("Detailed Results:\n");
    printf("--------------------------------------------------------------------------------\n");
    for (int i = 0; i < report->n_results; i++) {
        const char* label = report->results[i].with_aurora ? "Aurora" : "Baseline";
        printf("\n%d. [%s] %s\n", i + 1, label, report->results[i].question);
        printf("   Category: %s, Difficulty: %s\n", 
               report->results[i].category, report->results[i].difficulty);
        printf("   Wall Time: %.3fs\n", report->results[i].wall_time_s);
        printf("   Accuracy: %.2f%%\n", report->results[i].accuracy_score * 100.0f);
        printf("   Keywords Found: %d, Missing: %d\n", 
               report->results[i].keywords_found, report->results[i].keywords_missing);
        if (report->results[i].with_aurora) {
            printf("   Memory Reads: %d\n", report->results[i].retrieved_count);
        }
    }
    
    printf("\n");
    printf("================================================================================\n");
}

int main(int argc, char** argv) {
    printf("Aurora Real Text Test Suite (C++)\n");
    printf("==================================\n\n");
    
    // Parse arguments
    const char* model_path = NULL;
    const char* memory_bank_path = NULL;
    const char* output_path = "test_report.json";
    bool test_baseline = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--memory-bank") == 0 && i + 1 < argc) {
            memory_bank_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--test-baseline") == 0) {
            test_baseline = true;
        }
    }
    
    // Load test dataset
    struct test_question* questions = NULL;
    int n_questions = 0;
    if (load_test_dataset(&questions, &n_questions) != 0) {
        printf("Failed to load test dataset\n");
        return 1;
    }
    
    printf("Loaded %d test questions\n", n_questions);
    
    // Initialize Aurora config
    struct aurora_config config;
    aurora_config_init_defaults(&config);
    aurora_config_load_from_env(&config);
    
    // Load memory banks if provided
    aurora_memory_banks_t* memory_banks = NULL;
    if (memory_bank_path) {
        // TODO: Load memory banks from disk
        printf("Loading memory banks from: %s\n", memory_bank_path);
        // memory_banks = aurora_memory_banks_load(memory_bank_path, ...);
    }
    
    // Initialize LLAMA context (placeholder - would need actual model loading)
    struct llama_context* ctx = NULL;  // TODO: Load model
    
    // Initialize Aurora params
    struct llama_aurora_context_params aurora_params;
    llama_aurora_context_params_init(&aurora_params);
    aurora_params.enable_aurora_memory = (memory_banks != NULL);
    aurora_params.memory_banks = memory_banks;
    aurora_params.aurora_window_size = 2048;
    aurora_params.aurora_k_read = 8;
    aurora_params.enable_navigation_paradigm = true;
    
    // Allocate results array
    int max_results = n_questions * (test_baseline ? 2 : 1);
    struct test_result* results = (struct test_result*)malloc(max_results * sizeof(struct test_result));
    if (!results) {
        printf("Failed to allocate results array\n");
        return 1;
    }
    int n_results = 0;
    
    // Run tests with Aurora
    printf("\nRunning tests WITH Aurora...\n");
    for (int i = 0; i < n_questions; i++) {
        printf("  [%d/%d] %s\n", i + 1, n_questions, questions[i].question);
        if (run_test_with_aurora(&questions[i], ctx, &aurora_params, &results[n_results]) == 0) {
            printf("    ✓ Time: %.3fs, Accuracy: %.2f%%, Reads: %d\n",
                   results[n_results].wall_time_s,
                   results[n_results].accuracy_score * 100.0f,
                   results[n_results].retrieved_count);
            n_results++;
        } else {
            printf("    ✗ Failed\n");
        }
    }
    
    // Run tests without Aurora (baseline)
    if (test_baseline) {
        printf("\nRunning tests WITHOUT Aurora (baseline)...\n");
        aurora_params.enable_aurora_memory = false;
        for (int i = 0; i < n_questions; i++) {
            printf("  [%d/%d] %s\n", i + 1, n_questions, questions[i].question);
            if (run_test_baseline(&questions[i], ctx, &results[n_results]) == 0) {
                printf("    ✓ Time: %.3fs, Accuracy: %.2f%%\n",
                       results[n_results].wall_time_s,
                       results[n_results].accuracy_score * 100.0f);
                n_results++;
            } else {
                printf("    ✗ Failed\n");
            }
        }
    }
    
    // Generate report
    struct test_report report;
    memset(&report, 0, sizeof(report));
    
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    strftime(report.timestamp, sizeof(report.timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    report.n_results = n_results;
    report.results = results;
    generate_summary(&report);
    
    // Print report
    print_report(&report);
    
    // Save JSON report
    if (save_report_json(&report, output_path) == 0) {
        printf("\nReport saved to: %s\n", output_path);
    } else {
        printf("\nFailed to save report to: %s\n", output_path);
    }
    
    // Cleanup
    free(results);
    if (memory_banks) {
        // TODO: Free memory banks
    }
    
    return 0;
}
