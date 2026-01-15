#include "tkg_dataset.h"

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <string>

// Helper: trim whitespace
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Helper: split string by delimiter
static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

// Helper: parse time scalar (integer or ISO8601)
static float parse_time_scalar(const std::string& s) {
    // Try integer first
    try {
        return static_cast<float>(std::stoi(s));
    } catch (...) {
        // Try ISO8601 (simplified: extract year)
        // Format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
        if (s.length() >= 4) {
            try {
                return static_cast<float>(std::stoi(s.substr(0, 4)));
            } catch (...) {
                return 0.0f;
            }
        }
        return 0.0f;
    }
}

// Load TKG dataset from TSV file
tkg_dataset_t* tkg_dataset_load_tsv(const char* path, char delimiter) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return nullptr;
    }
    
    std::vector<std::string> entities;
    std::vector<std::string> relations;
    std::vector<std::string> times;
    std::vector<std::tuple<std::string, std::string, std::string, std::string>> rows;
    std::vector<float> time_values;
    
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;  // Skip empty lines and comments
        }
        
        auto tokens = split(line, delimiter);
        if (tokens.size() < 4) {
            continue;  // Skip malformed lines
        }
        
        std::string s = tokens[0];
        std::string r = tokens[1];
        std::string o = tokens[2];
        std::string t = tokens[3];
        
        entities.push_back(s);
        entities.push_back(o);
        relations.push_back(r);
        times.push_back(t);
        rows.push_back({s, r, o, t});
        time_values.push_back(parse_time_scalar(t));
    }
    file.close();
    
    // Build vocabularies (unique, sorted)
    std::map<std::string, int32_t> entity_to_id;
    std::map<std::string, int32_t> relation_to_id;
    std::map<std::string, int32_t> time_to_id;
    
    std::vector<std::string> unique_entities(entities.begin(), entities.end());
    std::sort(unique_entities.begin(), unique_entities.end());
    unique_entities.erase(std::unique(unique_entities.begin(), unique_entities.end()), unique_entities.end());
    
    std::vector<std::string> unique_relations(relations.begin(), relations.end());
    std::sort(unique_relations.begin(), unique_relations.end());
    unique_relations.erase(std::unique(unique_relations.begin(), unique_relations.end()), unique_relations.end());
    
    std::vector<std::string> unique_times(times.begin(), times.end());
    std::sort(unique_times.begin(), unique_times.end());
    unique_times.erase(std::unique(unique_times.begin(), unique_times.end()), unique_times.end());
    
    for (size_t i = 0; i < unique_entities.size(); i++) {
        entity_to_id[unique_entities[i]] = static_cast<int32_t>(i);
    }
    for (size_t i = 0; i < unique_relations.size(); i++) {
        relation_to_id[unique_relations[i]] = static_cast<int32_t>(i);
    }
    for (size_t i = 0; i < unique_times.size(); i++) {
        time_to_id[unique_times[i]] = static_cast<int32_t>(i);
    }
    
    // Normalize time values
    float time_min = *std::min_element(time_values.begin(), time_values.end());
    float time_max = *std::max_element(time_values.begin(), time_values.end());
    float time_span = time_max - time_min;
    
    // Allocate dataset
    tkg_dataset_t* dataset = static_cast<tkg_dataset_t*>(malloc(sizeof(tkg_dataset_t)));
    if (!dataset) return nullptr;
    
    memset(dataset, 0, sizeof(tkg_dataset_t));
    dataset->n_entities = static_cast<int32_t>(unique_entities.size());
    dataset->n_relations = static_cast<int32_t>(unique_relations.size());
    dataset->n_times = static_cast<int32_t>(unique_times.size());
    dataset->n_examples = static_cast<int32_t>(rows.size());
    
    // Allocate and copy entity/relation/time names
    dataset->entity_names = static_cast<char**>(malloc(dataset->n_entities * sizeof(char*)));
    dataset->relation_names = static_cast<char**>(malloc(dataset->n_relations * sizeof(char*)));
    dataset->time_names = static_cast<char**>(malloc(dataset->n_times * sizeof(char*)));
    
    if (!dataset->entity_names || !dataset->relation_names || !dataset->time_names) {
        tkg_dataset_free(dataset);
        return nullptr;
    }
    
    for (int32_t i = 0; i < dataset->n_entities; i++) {
        dataset->entity_names[i] = strdup(unique_entities[i].c_str());
    }
    for (int32_t i = 0; i < dataset->n_relations; i++) {
        dataset->relation_names[i] = strdup(unique_relations[i].c_str());
    }
    for (int32_t i = 0; i < dataset->n_times; i++) {
        dataset->time_names[i] = strdup(unique_times[i].c_str());
    }
    
    // Allocate and fill examples
    dataset->examples = static_cast<tkg_example_t*>(malloc(dataset->n_examples * sizeof(tkg_example_t)));
    if (!dataset->examples) {
        tkg_dataset_free(dataset);
        return nullptr;
    }
    
    for (size_t i = 0; i < rows.size(); i++) {
        const auto& row = rows[i];
        dataset->examples[i].s = entity_to_id[std::get<0>(row)];
        dataset->examples[i].r = relation_to_id[std::get<1>(row)];
        dataset->examples[i].o = entity_to_id[std::get<2>(row)];
        dataset->examples[i].t = time_to_id[std::get<3>(row)];
        
        // Normalize tau
        float tv = time_values[i];
        dataset->examples[i].tau = (time_span > 0.0f) ? ((tv - time_min) / time_span) : 0.0f;
    }
    
    return dataset;
}

// Load TKG dataset from split directory
int tkg_dataset_load_splits(
    const char* dir_path,
    char delimiter,
    tkg_dataset_t** train_out,
    tkg_dataset_t** valid_out,
    tkg_dataset_t** test_out
) {
    // Try common filenames
    const char* train_candidates[] = {"train.tsv", "train.txt", "train.csv", nullptr};
    const char* valid_candidates[] = {"valid.tsv", "valid.txt", "valid.csv", "dev.tsv", "dev.txt", "dev.csv", nullptr};
    const char* test_candidates[] = {"test.tsv", "test.txt", "test.csv", nullptr};
    
    std::string dir(dir_path);
    std::string train_path, valid_path, test_path;
    
    // Find train file
    for (int i = 0; train_candidates[i]; i++) {
        std::string candidate = dir + "/" + train_candidates[i];
        std::ifstream f(candidate);
        if (f.good()) {
            train_path = candidate;
            f.close();
            break;
        }
    }
    
    // Find valid file
    for (int i = 0; valid_candidates[i]; i++) {
        std::string candidate = dir + "/" + valid_candidates[i];
        std::ifstream f(candidate);
        if (f.good()) {
            valid_path = candidate;
            f.close();
            break;
        }
    }
    
    // Find test file
    for (int i = 0; test_candidates[i]; i++) {
        std::string candidate = dir + "/" + test_candidates[i];
        std::ifstream f(candidate);
        if (f.good()) {
            test_path = candidate;
            f.close();
            break;
        }
    }
    
    if (train_path.empty() || valid_path.empty() || test_path.empty()) {
        return -1;  // Files not found
    }
    
    *train_out = tkg_dataset_load_tsv(train_path.c_str(), delimiter);
    *valid_out = tkg_dataset_load_tsv(valid_path.c_str(), delimiter);
    *test_out = tkg_dataset_load_tsv(test_path.c_str(), delimiter);
    
    if (!*train_out || !*valid_out || !*test_out) {
        if (*train_out) tkg_dataset_free(*train_out);
        if (*valid_out) tkg_dataset_free(*valid_out);
        if (*test_out) tkg_dataset_free(*test_out);
        return -1;
    }
    
    return 0;
}

// Free TKG dataset
void tkg_dataset_free(tkg_dataset_t* dataset) {
    if (!dataset) return;
    
    if (dataset->entity_names) {
        for (int32_t i = 0; i < dataset->n_entities; i++) {
            if (dataset->entity_names[i]) free(dataset->entity_names[i]);
        }
        free(dataset->entity_names);
    }
    
    if (dataset->relation_names) {
        for (int32_t i = 0; i < dataset->n_relations; i++) {
            if (dataset->relation_names[i]) free(dataset->relation_names[i]);
        }
        free(dataset->relation_names);
    }
    
    if (dataset->time_names) {
        for (int32_t i = 0; i < dataset->n_times; i++) {
            if (dataset->time_names[i]) free(dataset->time_names[i]);
        }
        free(dataset->time_names);
    }
    
    if (dataset->examples) {
        free(dataset->examples);
    }
    
    free(dataset);
}

// Get entity ID by name
int32_t tkg_dataset_entity_id(const tkg_dataset_t* dataset, const char* entity_name) {
    if (!dataset || !entity_name) return -1;
    for (int32_t i = 0; i < dataset->n_entities; i++) {
        if (strcmp(dataset->entity_names[i], entity_name) == 0) {
            return i;
        }
    }
    return -1;
}

// Get relation ID by name
int32_t tkg_dataset_relation_id(const tkg_dataset_t* dataset, const char* relation_name) {
    if (!dataset || !relation_name) return -1;
    for (int32_t i = 0; i < dataset->n_relations; i++) {
        if (strcmp(dataset->relation_names[i], relation_name) == 0) {
            return i;
        }
    }
    return -1;
}

// Get time ID by name
int32_t tkg_dataset_time_id(const tkg_dataset_t* dataset, const char* time_name) {
    if (!dataset || !time_name) return -1;
    for (int32_t i = 0; i < dataset->n_times; i++) {
        if (strcmp(dataset->time_names[i], time_name) == 0) {
            return i;
        }
    }
    return -1;
}

// Normalize time values (already done in load_tsv, but can be called separately)
void tkg_dataset_normalize_time(tkg_dataset_t* dataset) {
    if (!dataset || dataset->n_examples == 0) return;
    
    // Find min/max tau
    float min_tau = dataset->examples[0].tau;
    float max_tau = dataset->examples[0].tau;
    for (int32_t i = 1; i < dataset->n_examples; i++) {
        if (dataset->examples[i].tau < min_tau) min_tau = dataset->examples[i].tau;
        if (dataset->examples[i].tau > max_tau) max_tau = dataset->examples[i].tau;
    }
    
    float span = max_tau - min_tau;
    if (span > 0.0f) {
        for (int32_t i = 0; i < dataset->n_examples; i++) {
            dataset->examples[i].tau = (dataset->examples[i].tau - min_tau) / span;
        }
    } else {
        for (int32_t i = 0; i < dataset->n_examples; i++) {
            dataset->examples[i].tau = 0.0f;
        }
    }
}

#ifdef __cplusplus
}
#endif
