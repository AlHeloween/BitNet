set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h)
set(GGML_SOURCES_BITNET ggml-bitnet-mad.cpp)
set(GGML_SOURCES_BITNET ggml-bitnet-lut.cpp)

# Aurora memory bank
set(AURORA_HEADERS 
    ../include/aurora_memory_bank.h 
    ../include/ggml-aurora-memory.h 
    ../include/llama-aurora-attention.h 
    ../include/llama-aurora-integration.h
    ../include/llama-aurora-navigation.h
    ../include/ggml-dual-complex.h
    ../include/ggml-dual-quaternion.h
    ../include/ggml-dq-linear.h
    ../include/ggml-dq-attention.h
    ../include/probe-encoder.h
    ../include/ggml-ffe-quantization.h
    ../include/ggml-lattice-addressing.h
    ../include/ggml-lattice-storage.h
    ../include/ggml-probe-to-lattice.h
    ../include/ggml-transcender.h
    ../include/ggml-screw-theory.h
    ../include/ggml-temporal-evolution.h
    ../include/genesis_metrics.h
    ../include/ggml-gpu-metrics.h
    ../include/aurora_config.h
)
set(AURORA_SOURCES 
    aurora_memory_bank.cpp 
    ggml-aurora-memory.cpp 
    llama-aurora-attention.cpp 
    llama-aurora-integration.cpp
    llama-aurora-navigation.cpp
    ggml-dual-complex.cpp
    ggml-dual-quaternion.cpp
    ggml-dq-linear.cpp
    ggml-dq-attention.cpp
    probe-encoder.cpp
    ggml-ffe-quantization.cpp
    ggml-lattice-addressing.cpp
    ggml-lattice-storage.cpp
    ggml-probe-to-lattice.cpp
    ggml-transcender.cpp
    ggml-screw-theory.cpp
    ggml-temporal-evolution.cpp
    genesis_metrics.cpp
    ggml-gpu-metrics.cpp
    aurora_config.cpp
)

include_directories(3rdparty/llama.cpp/ggml/include)
include_directories(../include)
include_directories(../../lib)  # For math_dualPhaser.h, math_dualQuat.h, etc.

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()

# CUDA support for GPU acceleration
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    
    # CUDA kernels for Aurora operations
    set(CUDA_SOURCES
        gpu/ggml-dual-complex-cuda.cu
        gpu/ggml-dual-quaternion-cuda.cu
        gpu/ggml-dq-linear-cuda.cu
        gpu/ggml-dq-attention-cuda.cu
        gpu/ggml-ffe-quantization-cuda.cu
        gpu/ggml-transcender-cuda.cu
        gpu/ggml-screw-theory-cuda.cu
        gpu/ggml-navigation-cuda.cu
    )
    
    # Add CUDA sources to Aurora sources
    list(APPEND AURORA_SOURCES ${CUDA_SOURCES})
    
    # Set CUDA properties
    set_property(SOURCE ${CUDA_SOURCES} PROPERTY LANGUAGE CUDA)
    set_property(SOURCE ${CUDA_SOURCES} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    # Add CUDA include directories
    include_directories(${CUDA_INCLUDE_DIRS})
    
    message(STATUS "CUDA found: Aurora GPU acceleration enabled")
else()
    message(STATUS "CUDA not found: Aurora will use CPU-only operations")
endif()

# ============================================================
# Standalone Build Targets (Phase 2: Production Readiness)
# ============================================================

# Option: Build standalone Aurora library (aurora-core)
option(BUILD_AURORA_STANDALONE "Build standalone Aurora library" OFF)

if(BUILD_AURORA_STANDALONE)
    # Create aurora-core static library
    add_library(aurora-core STATIC ${AURORA_SOURCES})
    
    target_include_directories(aurora-core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
    )
    
    if(CUDA_FOUND)
        target_link_libraries(aurora-core PUBLIC ${CUDA_LIBRARIES})
        target_include_directories(aurora-core PUBLIC ${CUDA_INCLUDE_DIRS})
    endif()
    
    target_link_libraries(aurora-core PUBLIC m)  # Math library
    
    set_target_properties(aurora-core PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Install aurora-core
    install(TARGETS aurora-core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(FILES ${AURORA_HEADERS} DESTINATION include/aurora)
    
    message(STATUS "Aurora-core library target created")
endif()

# Option: Build standalone CLI (aurora-cli)
option(BUILD_AURORA_CLI "Build standalone Aurora CLI" OFF)

if(BUILD_AURORA_CLI)
    if(NOT BUILD_AURORA_STANDALONE)
        message(FATAL_ERROR "BUILD_AURORA_CLI requires BUILD_AURORA_STANDALONE=ON")
    endif()
    
    # CLI source (minimal example - can be expanded)
    set(AURORA_CLI_SOURCES
        aurora_cli_main.cpp
    )
    
    # Check if CLI main exists, otherwise create a placeholder
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/aurora_cli_main.cpp")
        message(STATUS "Creating placeholder aurora_cli_main.cpp")
        file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/aurora_cli_main.cpp"
            "// Aurora CLI - Standalone command-line interface\n"
            "// TODO: Implement full CLI functionality\n"
            "#include <stdio.h>\n"
            "#include \"llama-aurora-integration.h\"\n"
            "\n"
            "int main(int argc, char** argv) {\n"
            "    printf(\"Aurora CLI (standalone build)\\n\");\n"
            "    printf(\"This is a placeholder - full CLI implementation pending\\n\");\n"
            "    return 0;\n"
            "}\n"
        )
    endif()
    
    add_executable(aurora-cli ${AURORA_CLI_SOURCES})
    
    target_link_libraries(aurora-cli PRIVATE aurora-core)
    
    if(CUDA_FOUND)
        target_link_libraries(aurora-cli PRIVATE ${CUDA_LIBRARIES})
    endif()
    
    set_target_properties(aurora-cli PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
    )
    
    # Install aurora-cli
    install(TARGETS aurora-cli DESTINATION bin)
    
    message(STATUS "Aurora-cli executable target created")
endif()

# Option: Build standalone GUI (aurora-gui) - Phase 3 placeholder
option(BUILD_AURORA_GUI "Build standalone Aurora Qt GUI" OFF)

if(BUILD_AURORA_GUI)
    if(NOT BUILD_AURORA_STANDALONE)
        message(FATAL_ERROR "BUILD_AURORA_GUI requires BUILD_AURORA_STANDALONE=ON")
    endif()
    
    # Find Qt6 (required for GUI)
    find_package(Qt6 QUIET COMPONENTS Core Widgets)
    
    if(NOT Qt6_FOUND)
        message(WARNING "Qt6 not found - aurora-gui will not be built")
        message(WARNING "Install Qt6 and set Qt6_DIR to enable GUI build")
    else()
        # GUI source (placeholder - Phase 3)
        set(AURORA_GUI_SOURCES
            aurora_gui_main.cpp
        )
        
        # Check if GUI main exists, otherwise create a placeholder
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/aurora_gui_main.cpp")
            message(STATUS "Creating placeholder aurora_gui_main.cpp")
            file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/aurora_gui_main.cpp"
                "// Aurora GUI - Standalone Qt6 application\n"
                "// TODO: Implement full Qt6 GUI (Phase 3)\n"
                "#include <QApplication>\n"
                "#include <QMainWindow>\n"
                "#include <QLabel>\n"
                "#include <stdio.h>\n"
                "\n"
                "int main(int argc, char** argv) {\n"
                "    QApplication app(argc, argv);\n"
                "    \n"
                "    QMainWindow window;\n"
                "    window.setWindowTitle(\"Aurora GUI (Placeholder)\");\n"
                "    window.resize(800, 600);\n"
                "    \n"
                "    QLabel* label = new QLabel(\"Aurora GUI - Full implementation pending (Phase 3)\", &window);\n"
                "    label->setAlignment(Qt::AlignCenter);\n"
                "    window.setCentralWidget(label);\n"
                "    \n"
                "    window.show();\n"
                "    \n"
                "    return app.exec();\n"
                "}\n"
            )
        endif()
        
        add_executable(aurora-gui ${AURORA_GUI_SOURCES})
        
        target_link_libraries(aurora-gui PRIVATE
            aurora-core
            Qt6::Core
            Qt6::Widgets
        )
        
        set_target_properties(aurora-gui PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
        
        # Install aurora-gui
        install(TARGETS aurora-gui DESTINATION bin)
        
        message(STATUS "Aurora-gui executable target created (Qt6)")
    endif()
endif()

# Option: Build test suite (Phase 4)
option(BUILD_AURORA_TESTS "Build Aurora test suite" OFF)

if(BUILD_AURORA_TESTS)
    if(NOT BUILD_AURORA_STANDALONE)
        message(FATAL_ERROR "BUILD_AURORA_TESTS requires BUILD_AURORA_STANDALONE=ON")
    endif()
    
    # Create tests directory if it doesn't exist
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    
    # Real text test suite
    set(AURORA_TEST_SOURCES
        tests/test_real_text.cpp
    )
    
    # Check if test source exists, otherwise create placeholder
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_real_text.cpp")
        message(STATUS "test_real_text.cpp not found, skipping test target")
    else()
        add_executable(aurora-test-real-text ${AURORA_TEST_SOURCES})
        
        target_link_libraries(aurora-test-real-text PRIVATE aurora-core)
        
        if(CUDA_FOUND)
            target_link_libraries(aurora-test-real-text PRIVATE ${CUDA_LIBRARIES})
        endif()
        
        # Link with llama if available (for actual model testing)
        # target_link_libraries(aurora-test-real-text PRIVATE llama)
        
        set_target_properties(aurora-test-real-text PROPERTIES
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
        )
        
        # Install test executable
        install(TARGETS aurora-test-real-text DESTINATION bin)
        
        message(STATUS "Aurora test suite target created (test-real-text)")
    endif()
endif()
