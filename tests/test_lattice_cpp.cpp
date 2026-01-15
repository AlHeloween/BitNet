#include "ggml-lattice-addressing.h"
#include "ggml-lattice-storage.h"
#include <cassert>
#include <cstdio>

int main() {
    // Test lattice address operations
    struct lattice_address addr = {3, 256, 0};
    
    // Test parent
    struct lattice_address parent = ggml_lattice_get_parent(&addr);
    assert(parent.level == 2);
    assert(parent.index == 128);  // 256 / 2
    
    // Test children
    struct lattice_address children[2];
    int num_children = 0;
    ggml_lattice_get_children(&addr, 7, children, &num_children);
    assert(num_children == 2);
    assert(children[0].level == 4);
    assert(children[0].index == 512 || children[0].index == 513);
    
    printf("Lattice addressing: PASS\n");
    
    // Test storage
    struct lattice_memory_storage* storage = ggml_lattice_storage_init(8, 512);
    assert(storage != NULL);
    
    struct lattice_memory_entry entry;
    for (int i = 0; i < 8; i++) {
        entry.embedding[i] = (float)i;
    }
    entry.text = NULL;
    entry.metadata = NULL;
    
    bool written = ggml_lattice_storage_write_leaf(storage, &addr, &entry);
    assert(written);
    
    bool has_leaf = ggml_lattice_storage_has_leaf(storage, &addr);
    assert(has_leaf);
    
    struct lattice_memory_entry entries_out[10];
    int num_read = ggml_lattice_storage_read_leaf(storage, &addr, entries_out, 10);
    assert(num_read == 1);
    
    ggml_lattice_storage_free(storage);
    
    printf("Lattice storage: PASS\n");
    
    return 0;
}
