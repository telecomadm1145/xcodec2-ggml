# xcodec2-ggml

A highly optimized C/C++ implementation of the [XCodec2](https://github.com/zhenye234/X-Codec-2.0) neural audio codec decoder using [ggml](https://github.com/ggml-org/llama.cpp/tree/master/ggml), with optional [llama.cpp](https://github.com/ggml-org/llama.cpp) integration for a full Text-to-Speech (TTS) pipeline.

This repository provides a suite of CLI tools and an HTTP API server for text-to-speech token generation and audio decoding, with optional Vulkan GPU acceleration.

### Download Pre-exported Models

You can download pre-exported GGUF decoder models (fp16 & f32 version) directly from Hugging Face:

- **XCodec2 Decoder GGUF Model**: [telecomadm1145/Anime-XCodec2-44.1kHz-v2-decoder-gguf](https://huggingface.co/telecomadm1145/Anime-XCodec2-44.1kHz-v2-decoder-gguf)

### Recommended LLM Models (Speech Token Generation)

The following LLM models are recommended for use with the TTS pipeline:

- **[NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B)** — 3B parameter model, higher quality output
- **[OmniAICreator/Galgame-Llasa-1B-v3](https://huggingface.co/OmniAICreator/Galgame-Llasa-1B-v3)** — 1B parameter model, faster inference

> **Note:** These models are not in GGUF format. You can check quantized version from its model tree. Alternatively, you can convert them using tools such as  [GGUF My Repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo)  or `llama.cpp`'s `convert_hf_to_gguf.py`. See the [Quantization Notes](#quantization-notes) section below before choosing a quant level.

---

## Tool Suite

After building, the project produces the following binaries:

- `xcodec2-tts` — End-to-end Text-to-Speech CLI tool. Integrates with `llama-server` (HTTP) or `llama-cli` (subprocess) to generate speech tokens and decode them to WAV.
- `xcodec2-server` — Lightweight HTTP server. Exposes endpoints for decoding speech tokens (`/decode`) and performing end-to-end TTS (`/tts`) to return binary WAV audio.
- `xcodec2-decode` — Standalone decoder that converts raw speech token files directly into WAV audio.
- `xcodec2-quantize` — Quantizer to compress xcodec2 GGUF models into Q8_0, Q4_0, etc.

---

## Building the Code

### Prerequisites

- CMake 3.14+
- A C/C++ compiler (Visual Studio 2022 / MSVC, Clang, or GCC)
- Ninja build system (highly recommended)
- (Optional) [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) for GPU decoding support

### 1. Windows (MSVC + Clang + Vulkan)

Open a standard command shell (or PowerShell) and invoke the compilation within the Visual Studio developer environment (using `vcvarsall.bat` so standard headers are resolved correctly for nested builds):

```powershell
# Setup VS environment, configure CMake, and build
cmd.exe /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 && cmake -DGGML_OPENMP=OFF -DGGML_VULKAN=ON . && ninja'
```

*Note: We disable OpenMP (`-DGGML_OPENMP=OFF`) to bypass symbol conflicts with Strawberry Perl's compiler runtime on Windows, allowing clean linking with Clang.*

### 2. Linux

```bash
cmake -DGGML_VULKAN=ON -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release .
ninja
```

---

## Quantization Notes

When converting or quantizing LLM models for use with this pipeline, keep the following in mind:

- **Q4_0 notably degrades output quality** for speech token generation — the model may produce garbled or incoherent audio. It is recommended to use **Q5_0, Q5_1, Q6_K, or Q8_0** for better results.
- For GGUF conversion, [GGUF My Repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) is a convenient web-based tool that handles the export process. Alternatively, use `llama.cpp`'s `convert_hf_to_gguf.py` locally.
- The `xcodec2-quantize` tool is for quantizing the **XCodec2 decoder** only, not the LLM.

---

## Vulkan GPU Decoding Acceleration

By default, **Vulkan decoding is disabled at runtime** to avoid competing for VRAM with the larger LLM (e.g. `llama-server`).
To enable Vulkan GPU acceleration for decoding, pass the `--gpu` or `--vulkan` flags to `xcodec2-tts`, `xcodec2-server`, or `xcodec2-decode`. If GPU acceleration is not available or initialization fails, the model will gracefully fall back to CPU decoding.

---

## Usage Instructions

### Recommended Workflow

**Use `xcodec2-server` + a persistent `llama-server` instance** rather than the `xcodec2-tts` CLI subprocess mode. This avoids reloading the LLM model on every TTS request, which is slow and resource-intensive.

```
llama-server  →  xcodec2-server  →  client (curl / your app)
  (LLM)             (decoder)
```

> **Important:** When starting `llama-server`, always include the `--special` flag to ensure special tokens (such as `<|s_xxxx|>` speech tokens) are handled correctly by the tokenizer. Without this flag, token generation may silently fail or produce incorrect output.

```bash
# Example llama-server launch
llama-server -m model.gguf --special -c 4096 --port 8080
```

---

### 1. `xcodec2-tts` (End-to-End TTS CLI)

Supports three distinct modes of operation:

#### Mode A: Server Mode (Inference via running `llama-server`)

Generates speech tokens by querying a running `llama-server` HTTP instance, then decodes them locally.

```bash
./xcodec2-tts.exe --url http://localhost:8080 -c xcodec2.gguf -t "Hello, welcome to xcodec2!" -o output.wav
```

#### Mode B: CLI Subprocess Mode (Inference via launching `llama-cli` binary)

Launches a local `llama-cli` subprocess to run the LLM inference.

```bash
./xcodec2-tts.exe --cli /path/to/llama-cli.exe -m /path/to/llm.gguf -c xcodec2.gguf -t "Text to generate" -o output.wav
```

> **Note:** This mode reloads the LLM from disk on every invocation. For repeated TTS use, Mode A with a persistent `llama-server` is strongly preferred.

#### Mode C: Direct Token Decoding Mode

Directly converts a string of space/comma-separated tokens or `<|s_xxxx|>` formatted tags to WAV audio.

```bash
./xcodec2-tts.exe --tokens "<|s_10023|><|s_29014|><|s_512|>" -c xcodec2.gguf -o output.wav
# Or:
./xcodec2-tts.exe --tokens "10023, 29014, 512" -c xcodec2.gguf -o output.wav
```

#### CLI Options

- `-c <path>` — Path to xcodec2 GGUF model (default: `xcodec2.gguf`)
- `-o <path>` — Path to output WAV file (default: `output.wav`)
- `-t <text>` — Input inline text for TTS
- `-f <path>` — Input text file for TTS
- `-n <max>` — Max tokens to generate (default: `2048`)
- `--temp <val>` — LLM sampling temperature (default: `0.8`)
- `--top-p <val>` — LLM top-p sampling (default: `0.95`)
- `--top-k <val>` — LLM top-k sampling (default: `50`)
- `--repeat-penalty <val>` — Repetition penalty to prevent generation loops (default: `1.1`)
- `--gpu` or `--vulkan` — Enable Vulkan GPU acceleration for decoding
- `--save-tokens <path>` — Save generated token IDs to a text file
- `--debug-tokens` — Print generated speech token IDs during execution
- `--no-decode` — Skip decoding to WAV, only output the speech tokens

---

### 2. `xcodec2-server` (HTTP Audio Service)

Starts a lightweight HTTP server to handle token decoding and TTS requests.

```bash
# Start server on default port 8082, with GPU decoding enabled
./xcodec2-server.exe -c xcodec2.gguf --port 8082 --gpu
```

#### Endpoints

##### `GET /health`

Returns server status.

```json
{ "status": "ok" }
```

##### `POST /decode`

Decodes an array of token IDs or formatted token strings directly to WAV audio.

- **Request Body (JSON)**:
  ```json
  { "tokens": [10023, 29014, 512] }
  // Or:
  { "tokens": "<|s_10023|><|s_29014|><|s_512|>" }
  // Or:
  { "tokens": "10023, 29014, 512" }
  ```
- **Response**: Binary stream of `audio/wav`
- **Example Curl**:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d "{\"tokens\": [1023, 5678, 912]}" http://localhost:8082/decode -o output.wav
  ```

##### `POST /tts`

Performs end-to-end TTS by querying a backend `llama-server` instance to generate speech tokens, then decodes them to WAV.

- **Request Body (JSON)**:
  ```json
  {
    "text": "Hello, this is generated entirely via the xcodec2 server API.",
    "llama_url": "http://localhost:8080",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repeat_penalty": 1.1,
    "max_tokens": 2048
  }
  ```
- **Response**: Binary stream of `audio/wav`
- **Example Curl**:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"Hello from the API!\", \"llama_url\": \"http://localhost:8080\"}" http://localhost:8082/tts -o tts_output.wav
  ```

---

### 3. `xcodec2-decode` (Standalone Decoder CLI)

Converts a file containing whitespace or newline-separated speech token integers to WAV.

```bash
./xcodec2-decode.exe -m xcodec2.gguf -i codes.txt -o output.wav [--gpu]
```

---

### 4. `xcodec2-quantize` (Model Compression)

Quantizes the xcodec2 GGUF model into compressed formats to reduce disk usage and speed up CPU memory access.

```bash
./xcodec2-quantize.exe xcodec2.gguf xcodec2-q8_0.gguf q8_0
```

*Supported quant types: `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `f16`.*

> **Note:** This tool is for the XCodec2 decoder model only. For the LLM, use `llama.cpp`'s own quantization tools. See [Quantization Notes](#quantization-notes) for guidance on choosing a quant level.

---

## Project Structure

| File | Description |
|------|-------------|
| [CMakeLists.txt](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/CMakeLists.txt) | Build configuration (fetches llama.cpp dynamically) |
| [xcodec2.h](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/xcodec2.h) | Public API and structure definitions |
| [xcodec2.c](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/xcodec2.c) | Decoder graph execution + ISTFT + standalone decoder CLI |
| [xcodec2_load.c](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/xcodec2_load.c) | Model GGUF loader & Vulkan/CPU backend allocator |
| [xcodec2_graph.c](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/xcodec2_graph.c) | ggml graph builder |
| [main.cpp](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/main.cpp) | `xcodec2-tts` source code |
| [server.cpp](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/server.cpp) | `xcodec2-server` HTTP API server |
| [quantize.cpp](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/quantize.cpp) | Quantization tool code |
| [export_gguf.py](file:///c:/Users/Administrator/Downloads/xcodec2-0.1.7.tar/xcodec2-0.1.7/ggml/export_gguf.py) | PyTorch to GGUF model converter |

---

## License

MIT