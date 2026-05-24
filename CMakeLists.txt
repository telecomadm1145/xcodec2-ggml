cmake_minimum_required(VERSION 3.14)
project(xcodec2_tts CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Suppress MSVC STL version check for older Clang
add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)

include(FetchContent)

# ── llama.cpp (includes ggml) ───────────────────────────────────
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        master
)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama_cpp)

# ── Shared xcodec2 source list ──────────────────────────────────
set(XCODEC2_SOURCES
    xcodec2.c
    xcodec2_load.c
    xcodec2_graph.c
)

# ── xcodec2-tts (inference via external service + xcodec2 decoder) ──
add_executable(xcodec2-tts
    main.cpp
    ${XCODEC2_SOURCES}
)
target_link_libraries(xcodec2-tts PRIVATE ggml)
target_include_directories(xcodec2-tts PRIVATE
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

# ── xcodec2-server (HTTP decode server) ─────────────────────────
add_executable(xcodec2-server
    server.cpp
    ${XCODEC2_SOURCES}
)
target_link_libraries(xcodec2-server PRIVATE ggml)
target_include_directories(xcodec2-server PRIVATE
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

# ── Standalone xcodec2 decoder (codes.txt -> wav) ──────────────
add_executable(xcodec2-decode
    ${XCODEC2_SOURCES}
)
target_link_libraries(xcodec2-decode PRIVATE ggml)
target_include_directories(xcodec2-decode PRIVATE
    ${llama_cpp_SOURCE_DIR}/ggml/include
)
target_compile_definitions(xcodec2-decode PRIVATE XCODEC2_STANDALONE)

# ── xcodec2 quantize tool ──────────────────────────────────────
add_executable(xcodec2-quantize
    quantize.cpp
)
target_link_libraries(xcodec2-quantize PRIVATE ggml)
target_include_directories(xcodec2-quantize PRIVATE
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

# ── Platform-specific libs ──────────────────────────────────────
if(WIN32)
    target_link_libraries(xcodec2-server PRIVATE ws2_32)
else()
    target_link_libraries(xcodec2-tts PRIVATE m)
    target_link_libraries(xcodec2-server PRIVATE m)
    target_link_libraries(xcodec2-decode PRIVATE m)
endif()

if(GGML_VULKAN)
    target_link_libraries(xcodec2-tts PRIVATE ggml-vulkan)
    target_link_libraries(xcodec2-server PRIVATE ggml-vulkan)
    target_link_libraries(xcodec2-decode PRIVATE ggml-vulkan)
endif()

