# Building Aurora Compute Reduction Components

This guide explains how to build the Aurora compute reduction components for BitNet.

## Prerequisites

- BitNet repository cloned to `external/bitnet/`
- CMake 3.14 or later
- C++11 compatible compiler (Clang or GCC)
- llama.cpp submodule available at `external/bitnet/3rdparty/llama.cpp/`

## Build Options

### Option 1: Standalone Library

Build Aurora as a standalone static library:

```bash
cd external/bitnet
mkdir -p build_aurora
cd build_aurora

cmake .. -DCMAKE_BUILD_TYPE=Release
make aurora
```

This creates `libaurora.a` in the build directory.

### Option 2: Integrated with BitNet

To integrate Aurora into BitNet's build system:

1. Add to `external/bitnet/src/CMakeLists.txt`:
```cmake
# Include Aurora sources
include(CMakeLists_aurora.cmake)
```

2. Link Aurora library in your target:
```cmake
target_link_libraries(your_target aurora)
```

### Option 3: Manual Compilation

For quick testing, compile manually:

```bash
cd external/bitnet

# Compile Aurora components
g++ -std=c++11 -Iinclude -I3rdparty/llama.cpp/ggml/include \
    -c src/aurora_memory_bank.cpp -o build/aurora_memory_bank.o
g++ -std=c++11 -Iinclude -I3rdparty/llama.cpp/ggml/include \
    -c src/ggml-aurora-memory.cpp -o build/ggml-aurora-memory.o
g++ -std=c++11 -Iinclude -I3rdparty/llama.cpp/ggml/include \
    -c src/llama-aurora-attention.cpp -o build/llama-aurora-attention.o
g++ -std=c++11 -Iinclude -I3rdparty/llama.cpp/ggml/include \
    -c src/llama-aurora-integration.cpp -o build/llama-aurora-integration.o

# Create static library
ar rcs build/libaurora.a build/*.o
```

## Testing

After building, run the test suite:

```bash
cd external/bitnet/tests

# Compile tests
g++ -std=c++11 -I../include -I../3rdparty/llama.cpp/ggml/include \
    test_aurora_memory_cpp.cpp ../src/aurora_memory_bank.cpp \
    -o test_aurora_memory_cpp

# Run tests
./test_aurora_memory_cpp
```

## Integration with llama.cpp

To use Aurora in llama.cpp, you need to:

1. Link the Aurora library
2. Include Aurora headers in llama.cpp
3. Modify `llama_decode_internal()` to call Aurora hooks

See `src/aurora_integration_example.cpp` for integration examples.

## Troubleshooting

### Compilation Errors

- **Missing GGML headers**: Ensure `3rdparty/llama.cpp/ggml/include` is in include path
- **C++11 not supported**: Use `-std=c++11` or later
- **Math functions not found**: Link with `-lm` on Linux/Unix

### Link Errors

- **Undefined references**: Ensure all Aurora source files are compiled and linked
- **GGML symbols not found**: Link with llama.cpp/ggml library

### Runtime Errors

- **Memory bank initialization fails**: Check parameters (n_clusters, dim, depth, seed)
- **Memory reads return empty**: Verify memory banks are populated before querying

## See Also

- `docs/aurora_compute_reduction.md` - Architecture documentation
- `docs/aurora_compute_reduction_usage.md` - Usage guide
- `src/aurora_integration_example.cpp` - Integration examples
