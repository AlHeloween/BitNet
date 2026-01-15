# Aurora Build Integration for BitNet

## Overview

Aurora compute reduction has been integrated into llama.cpp's forward pass. To build BitNet with Aurora support, you need to ensure Aurora sources are compiled and linked.

## Integration Points

### Modified Files

1. **`3rdparty/llama.cpp/src/llama.cpp`**:
   - Added `#include "../../include/llama-aurora-integration.h"` (line ~32)
   - Added `aurora_params` field to `llama_context` struct (line ~3442)
   - Modified attention computation to use Aurora hooks (line ~9806)
   - Added post-forward hook call (line ~17685)
   - Added Aurora params initialization (line ~19728)
   - Added Aurora params cleanup in destructor (line ~3353)

2. **`3rdparty/llama.cpp/include/llama.h`**:
   - (No changes needed - forward declaration is sufficient)

### Aurora Source Files

Located in `external/bitnet/src/`:
- `aurora_memory_bank.cpp`
- `ggml-aurora-memory.cpp`
- `llama-aurora-attention.cpp`
- `llama-aurora-integration.cpp`

## Build Configuration

### Option 1: Add Aurora Sources to llama Library

Modify `external/bitnet/3rdparty/llama.cpp/src/CMakeLists.txt`:

```cmake
add_library(llama
            ../include/llama.h
            llama.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            # Aurora sources (add these)
            ../../../src/aurora_memory_bank.cpp
            ../../../src/ggml-aurora-memory.cpp
            ../../../src/llama-aurora-attention.cpp
            ../../../src/llama-aurora-integration.cpp
            )

# Add Aurora include directories
target_include_directories(llama PUBLIC 
    . 
    ../include
    ../../../include  # Aurora headers
)
```

### Option 2: Create Separate Aurora Library

In `external/bitnet/src/CMakeLists.txt`, create Aurora library:

```cmake
# Create Aurora library
add_library(aurora STATIC
    aurora_memory_bank.cpp
    ggml-aurora-memory.cpp
    llama-aurora-attention.cpp
    llama-aurora-integration.cpp
)

target_include_directories(aurora PUBLIC
    ../include
    3rdparty/llama.cpp/ggml/include
)

target_link_libraries(aurora PUBLIC m)  # Math library

# Link Aurora to llama
target_link_libraries(llama PRIVATE aurora)
```

Then modify `external/bitnet/3rdparty/llama.cpp/src/CMakeLists.txt` to link Aurora:

```cmake
# After target_link_libraries(llama PUBLIC ggml)
target_link_libraries(llama PRIVATE aurora)
```

## Build Steps

1. **Configure CMake**:
   ```bash
   cd external/bitnet
   mkdir -p build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. **Build**:
   ```bash
   cmake --build . --config Release
   ```

3. **Verify**:
   ```bash
   # Check that llama-cli was built
   ls build/bin/llama-cli.exe  # Windows
   ls build/bin/llama-cli      # Linux/Mac
   ```

## Testing

After building, test the integration:

```bash
# Test memory bank operations
python scripts/test_aurora_integration.py

# Populate memory from Wikipedia
python scripts/populate_aurora_memory_from_wiki.py --dataset-dir .cache/wiki/en_50k

# Launch Qt GUI
python scripts/aurora_compute_reduction_qt.py
```

## Troubleshooting

### Compilation Errors

- **Missing headers**: Ensure `../../../include` is in include path
- **Undefined references**: Ensure Aurora sources are compiled and linked
- **Link errors**: Ensure Aurora library is linked to llama

### Runtime Issues

- **Aurora hooks not called**: Verify `ctx->aurora_params` is set and `enable_aurora_memory` is true
- **Memory banks not found**: Ensure memory banks are loaded before use
- **Performance not improved**: Verify compute reduction is actually active (check metrics)

## Notes

- Aurora integration modifies llama.cpp submodule - consider creating a patch file
- For production, consider maintaining Aurora as a separate library
- Memory banks are managed externally (Python or C++ code that creates llama_context)
