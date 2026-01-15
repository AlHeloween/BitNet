# CMake configuration for Aurora compute reduction components
# Include this in the main CMakeLists.txt or src/CMakeLists.txt

# Aurora source files
set(AURORA_SOURCES
    src/aurora_memory_bank.cpp
    src/ggml-aurora-memory.cpp
    src/llama-aurora-attention.cpp
    src/llama-aurora-integration.cpp
)

# Aurora headers
set(AURORA_HEADERS
    include/aurora_memory_bank.h
    include/ggml-aurora-memory.h
    include/llama-aurora-attention.h
    include/llama-aurora-integration.h
)

# Aurora include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)

# Create Aurora static library
add_library(aurora STATIC ${AURORA_SOURCES})

# Link with math library (for sqrt, etc.)
target_link_libraries(aurora m)

# Set C++ standard
set_target_properties(aurora PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)

# Install headers
install(FILES ${AURORA_HEADERS} DESTINATION include/aurora)

# Install library
install(TARGETS aurora DESTINATION lib)
