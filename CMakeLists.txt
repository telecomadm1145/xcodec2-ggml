cmake_minimum_required(VERSION 3.14)
project(xcodec2_tts CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

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

# ── xcodec2 TTS (llama + xcodec2 decoder) ──────────────────────
add_executable(xcodec2_tts
    main.cpp
    xcodec2.c
    xcodec2_load.c
    xcodec2_graph.c
)
target_link_libraries(xcodec2_tts PRIVATE llama ggml)
target_include_directories(xcodec2_tts PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

# ── Standalone xcodec2 decoder (codes.txt -> wav) ──────────────
add_executable(xcodec2_decode
    xcodec2.c
    xcodec2_load.c
    xcodec2_graph.c
)
target_link_libraries(xcodec2_decode PRIVATE ggml)
target_include_directories(xcodec2_decode PRIVATE
    ${llama_cpp_SOURCE_DIR}/ggml/include
)
target_compile_definitions(xcodec2_decode PRIVATE XCODEC2_STANDALONE)

# ── xcodec2 quantize tool ──────────────────────────────────────
add_executable(xcodec2_quantize
    quantize.cpp
)
target_link_libraries(xcodec2_quantize PRIVATE ggml)
target_include_directories(xcodec2_quantize PRIVATE
    ${llama_cpp_SOURCE_DIR}/ggml/include
)

if(NOT WIN32)
    target_link_libraries(xcodec2_tts PRIVATE m)
    target_link_libraries(xcodec2_decode PRIVATE m)
endif()
