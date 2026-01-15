#include "ggml-lattice-storage.h"
#include "ggml.h"

#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <vector>

// Hash function for lattice_address
struct lattice_address_hash {
    size_t operator()(const struct lattice_address& addr) const {
        return (size_t)((addr.level << 16) | (addr.index << 2) | addr.sub_index);
    }
};

// Equality function for lattice_address
struct lattice_address_equal {
    bool operator()(const struct lattice_address& a, const struct lattice_address& b) const {
        return a.level == b.level && a.index == b.index && a.sub_index == b.sub_index;
    }
};

// Internal storage structure
struct lattice_storage_internal {
    std::unordered_map<
        struct lattice_address,
        std::vector<struct lattice_memory_entry>,
        lattice_address_hash,
        lattice_address_equal
    > storage;
    int n_levels;
    int n_per_level;
};

struct lattice_memory_storage* ggml_lattice_storage_init(int n_levels, int n_per_level) {
    auto* internal = new lattice_storage_internal();
    internal->n_levels = n_levels;
    internal->n_per_level = n_per_level;
    
    auto* storage = new struct lattice_memory_storage();
    storage->storage = internal;
    storage->n_levels = n_levels;
    storage->n_per_level = n_per_level;
    
    return storage;
}

void ggml_lattice_storage_free(struct lattice_memory_storage* storage) {
    if (!storage) return;
    
    auto* internal = (lattice_storage_internal*)storage->storage;
    if (internal) {
        // Free text strings
        for (auto& pair : internal->storage) {
            for (auto& entry : pair.second) {
                if (entry.text) {
                    delete[] entry.text;
                }
                if (entry.metadata) {
                    delete (char*)entry.metadata;  // Assuming metadata is char*
                }
            }
        }
        delete internal;
    }
    delete storage;
}

bool ggml_lattice_storage_write_leaf(
    struct lattice_memory_storage* storage,
    const struct lattice_address* addr,
    const struct lattice_memory_entry* entry
) {
    if (!storage || !addr || !entry) {
        return false;
    }
    
    auto* internal = (lattice_storage_internal*)storage->storage;
    
    // Copy entry
    struct lattice_memory_entry entry_copy;
    memcpy(entry_copy.embedding, entry->embedding, 8 * sizeof(float));
    
    if (entry->text) {
        size_t text_len = strlen(entry->text);
        entry_copy.text = new char[text_len + 1];
        strcpy(entry_copy.text, entry->text);
    } else {
        entry_copy.text = NULL;
    }
    
    entry_copy.metadata = NULL;  // Simplified for now
    
    // Store
    internal->storage[*addr].push_back(entry_copy);
    
    return true;
}

int ggml_lattice_storage_read_leaf(
    struct lattice_memory_storage* storage,
    const struct lattice_address* addr,
    struct lattice_memory_entry* entries_out,
    int max_entries
) {
    if (!storage || !addr || !entries_out || max_entries <= 0) {
        return 0;
    }
    
    auto* internal = (lattice_storage_internal*)storage->storage;
    
    auto it = internal->storage.find(*addr);
    if (it == internal->storage.end()) {
        return 0;
    }
    
    int count = 0;
    for (const auto& entry : it->second) {
        if (count >= max_entries) break;
        
        // Copy entry
        memcpy(entries_out[count].embedding, entry.embedding, 8 * sizeof(float));
        
        if (entry.text) {
            size_t text_len = strlen(entry.text);
            entries_out[count].text = new char[text_len + 1];
            strcpy(entries_out[count].text, entry.text);
        } else {
            entries_out[count].text = NULL;
        }
        
        entries_out[count].metadata = NULL;
        count++;
    }
    
    return count;
}

bool ggml_lattice_storage_has_leaf(
    struct lattice_memory_storage* storage,
    const struct lattice_address* addr
) {
    if (!storage || !addr) {
        return false;
    }
    
    auto* internal = (lattice_storage_internal*)storage->storage;
    return internal->storage.find(*addr) != internal->storage.end();
}

bool ggml_lattice_storage_save(
    struct lattice_memory_storage* storage,
    const char* path
) {
    // Simplified: use binary format
    // In production, would use proper serialization
    FILE* f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    
    auto* internal = (lattice_storage_internal*)storage->storage;
    
    // Write header
    int32_t n_levels = storage->n_levels;
    int32_t n_per_level = storage->n_per_level;
    int32_t n_addresses = (int32_t)internal->storage.size();
    
    fwrite(&n_levels, sizeof(int32_t), 1, f);
    fwrite(&n_per_level, sizeof(int32_t), 1, f);
    fwrite(&n_addresses, sizeof(int32_t), 1, f);
    
    // Write entries
    for (const auto& pair : internal->storage) {
        // Write address
        fwrite(&pair.first, sizeof(struct lattice_address), 1, f);
        
        // Write entry count
        int32_t n_entries = (int32_t)pair.second.size();
        fwrite(&n_entries, sizeof(int32_t), 1, f);
        
        // Write entries
        for (const auto& entry : pair.second) {
            fwrite(entry.embedding, sizeof(float), 8, f);
            
            // Write text length and text
            if (entry.text) {
                int32_t text_len = (int32_t)strlen(entry.text);
                fwrite(&text_len, sizeof(int32_t), 1, f);
                fwrite(entry.text, sizeof(char), text_len, f);
            } else {
                int32_t text_len = 0;
                fwrite(&text_len, sizeof(int32_t), 1, f);
            }
        }
    }
    
    fclose(f);
    return true;
}

bool ggml_lattice_storage_load(
    struct lattice_memory_storage* storage,
    const char* path
) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    
    auto* internal = (lattice_storage_internal*)storage->storage;
    
    // Read header
    int32_t n_levels, n_per_level, n_addresses;
    fread(&n_levels, sizeof(int32_t), 1, f);
    fread(&n_per_level, sizeof(int32_t), 1, f);
    fread(&n_addresses, sizeof(int32_t), 1, f);
    
    // Read entries
    for (int i = 0; i < n_addresses; i++) {
        struct lattice_address addr;
        fread(&addr, sizeof(struct lattice_address), 1, f);
        
        int32_t n_entries;
        fread(&n_entries, sizeof(int32_t), 1, f);
        
        std::vector<struct lattice_memory_entry> entries;
        for (int j = 0; j < n_entries; j++) {
            struct lattice_memory_entry entry;
            fread(entry.embedding, sizeof(float), 8, f);
            
            int32_t text_len;
            fread(&text_len, sizeof(int32_t), 1, f);
            
            if (text_len > 0) {
                entry.text = new char[text_len + 1];
                fread(entry.text, sizeof(char), text_len, f);
                entry.text[text_len] = '\0';
            } else {
                entry.text = NULL;
            }
            
            entry.metadata = NULL;
            entries.push_back(entry);
        }
        
        internal->storage[addr] = entries;
    }
    
    fclose(f);
    return true;
}

void ggml_lattice_storage_get_stats(
    struct lattice_memory_storage* storage,
    int* total_addresses_out,
    int* total_entries_out
) {
    if (!storage || !total_addresses_out || !total_entries_out) {
        return;
    }
    
    auto* internal = (lattice_storage_internal*)storage->storage;
    
    *total_addresses_out = (int)internal->storage.size();
    
    int total_entries = 0;
    for (const auto& pair : internal->storage) {
        total_entries += (int)pair.second.size();
    }
    *total_entries_out = total_entries;
}
