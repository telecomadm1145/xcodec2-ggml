# xcodec2-ggml

A C/C++ implementation of the [XCodec2](https://github.com/zhenye234/xcodec2) neural audio codec decoder using [ggml](https://github.com/ggml-org/llama.cpp/tree/master/ggml), with optional [llama.cpp](https://github.com/ggml-org/llama.cpp) integration for a full Text-to-Speech (TTS) pipeline.

## Features

- **xcodec2 decoder** — Decodes speech token IDs to audio waveforms using a GGUF model
- **Full TTS pipeline** — Chains an LLM (via llama.cpp) with xcodec2 for end-to-end text-to-speech
- **Quantization tool** — Quantize xcodec2 models to Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, or F16
- **GPU acceleration** — Vulkan and CPU (AVX2/FMA) backends via ggml
- **Cross-platform** — Builds on Windows and Linux

## Building

### Prerequisites

- CMake 3.14+
- A C/C++ compiler (Clang 19+, MSVC 19+, or GCC 11+)
- Ninja (recommended) or Make
- (Optional) [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) for GPU acceleration

### Windows (Visual Studio 2022 + Clang 19)

```powershell
# Open VS Developer Command Prompt, then:
set VULKAN_SDK=C:\VulkanSDK\1.4.350.0

cmake -G Ninja ^
  -DCMAKE_C_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe" ^
  -DCMAKE_CXX_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin/clang-cl.exe" ^
  -DGGML_VULKAN=ON ^
  -DCMAKE_BUILD_TYPE=Release .

ninja -j8
```

### Linux

```bash
cmake -G Ninja \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release .

ninja -j$(nproc)
```

### Without Vulkan (CPU only)

```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .
ninja -j$(nproc)
```

## Usage

### 1. Export model to GGUF

Convert the PyTorch XCodec2 model to GGUF format:

```bash
python export_gguf.py --model-dir /path/to/xcodec2 --output xcodec2.gguf
```

### 2. (Optional) Quantize

```bash
./xcodec2_quantize xcodec2.gguf xcodec2-q8_0.gguf q8_0
```

### 3a. Standalone decoder (codes → WAV)

```bash
./xcodec2_decode -m xcodec2.gguf -i codes.txt -o output.wav
```

Where `codes.txt` contains one integer speech token ID per line.

### 3b. Full TTS pipeline (text → WAV)

```bash
./xcodec2_tts -m llasa-3b.gguf -c xcodec2.gguf -t "Hello, world!" -o output.wav
```

Options:
- `-m` — Path to LLM GGUF model (e.g., Llasa-3B)
- `-c` — Path to xcodec2 GGUF model
- `-t` — Input text
- `-o` — Output WAV file (default: `output.wav`)
- `-ngl` — Number of GPU layers (default: 99)
- `-n` — Max tokens (default: 2048)
- `--temp` — Temperature (default: 0.8)
- `--top-p` — Top-p sampling (default: 1.0)
- `--rep-penalty` — Repetition penalty (default: 1.1)

## Project Structure

| File | Description |
|------|-------------|
| `CMakeLists.txt` | Build configuration (fetches llama.cpp via FetchContent) |
| `xcodec2.h` | Public API and model structures |
| `xcodec2.c` | Decoder inference + ISTFT + standalone CLI |
| `xcodec2_graph.c` | ggml compute graph construction |
| `xcodec2_load.c` | GGUF model loading |
| `main.cpp` | Full TTS pipeline (LLM + xcodec2) |
| `quantize.cpp` | Model quantization tool |
| `export_gguf.py` | PyTorch → GGUF converter |
| `export_ggml.py` | Alternative ggml export script |

## License

MIT
