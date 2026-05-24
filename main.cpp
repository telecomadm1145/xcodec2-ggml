/*
 * main.cpp - Refactored TTS pipeline: LLM (via external llama-server or llama-cli) -> xcodec2 decoder -> WAV
 */

#include "ggml.h"

extern "C" {
#include "xcodec2.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <locale>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define popen _popen
#define pclose _pclose
#endif

/* ── Helper: Convert system codepage string to UTF-8 (Windows) ─ */
#ifdef _WIN32
static std::string acp_to_utf8(const std::string &s) {
  if (s.empty()) return s;
  // Step 1: ANSI (ACP) -> UTF-16
  int wlen = MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int)s.size(), NULL, 0);
  if (wlen <= 0) return s;
  std::vector<wchar_t> wbuf(wlen);
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int)s.size(), wbuf.data(), wlen);
  // Step 2: UTF-16 -> UTF-8
  int ulen = WideCharToMultiByte(CP_UTF8, 0, wbuf.data(), wlen, NULL, 0, NULL, NULL);
  if (ulen <= 0) return s;
  std::string utf8(ulen, '\0');
  WideCharToMultiByte(CP_UTF8, 0, wbuf.data(), wlen, &utf8[0], ulen, NULL, NULL);
  return utf8;
}
#else
static std::string acp_to_utf8(const std::string &s) { return s; }
#endif

/* ── Helper: Read file to string ─────────────────────────────── */
static bool read_file_to_string(const char *path, std::string &out) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize(size);
  if (size > 0 && fread(&out[0], 1, size, f) != (size_t)size) {
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

/* ── WAV writer ──────────────────────────────────────────────── */
static void write_wav(const char *path, const float *data, int n, int sr) {
  FILE *f = fopen(path, "wb");
  if (!f) return;
  std::vector<int16_t> pcm(n);
  for (int i = 0; i < n; i++) {
    float s = data[i] * 32767.0f;
    s = s > 32767.f ? 32767.f : (s < -32768.f ? -32768.f : s);
    pcm[i] = (int16_t)s;
  }
  int ds = n * 2, fs = 36 + ds;
  int16_t fmt = 1, ch = 1, ba = 2, bps = 16;
  int br = sr * 2;
  fwrite("RIFF", 1, 4, f);
  fwrite(&fs, 4, 1, f);
  fwrite("WAVEfmt ", 1, 8, f);
  int cs = 16;
  fwrite(&cs, 4, 1, f);
  fwrite(&fmt, 2, 1, f);
  fwrite(&ch, 2, 1, f);
  fwrite(&sr, 4, 1, f);
  fwrite(&br, 4, 1, f);
  fwrite(&ba, 2, 1, f);
  fwrite(&bps, 2, 1, f);
  fwrite("data", 1, 4, f);
  fwrite(&ds, 4, 1, f);
  fwrite(pcm.data(), 2, n, f);
  fclose(f);
  printf("wrote %s (%d samples, %d Hz)\n", path, n, sr);
}

/* ── Build the chat prompt using model's native template ─────── */
static std::string build_prompt(const std::string &text) {
  // Llasa / Llama-3 Instruct 官方模板
  std::string prompt =
      "<|start_header_id|>system<|end_header_id|>\n\n"
      "Cutting Knowledge Date: December 2023\n"
      "Today Date: 26 Jul 2024\n\n"
      "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
      "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>" + text +
      "<|TEXT_UNDERSTANDING_END|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
      "<|SPEECH_GENERATION_START|>";
  return prompt;
}

/* ── JSON and String Escaping Utilities ──────────────────────── */
static std::string escape_json(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if ((unsigned char)c < 32) {
      char buf[16];
      sprintf(buf, "\\u%04x", c);
      out += buf;
    } else {
      out += c;
    }
  }
  return out;
}

static std::string parse_json_content(const std::string &json) {
  size_t pos = json.find("\"content\"");
  if (pos == std::string::npos) return "";
  pos = json.find(":", pos + 9);
  if (pos == std::string::npos) return "";
  pos = json.find("\"", pos + 1);
  if (pos == std::string::npos) return "";
  
  std::string content;
  size_t i = pos + 1;
  while (i < json.size()) {
    if (json[i] == '"') {
      break;
    }
    if (json[i] == '\\' && i + 1 < json.size()) {
      char next = json[i+1];
      if (next == '"') { content += '"'; i += 2; }
      else if (next == '\\') { content += '\\'; i += 2; }
      else if (next == 'n') { content += '\n'; i += 2; }
      else if (next == 'r') { content += '\r'; i += 2; }
      else if (next == 't') { content += '\t'; i += 2; }
      else { content += json[i]; i++; }
    } else {
      content += json[i];
      i++;
    }
  }
  return content;
}

/* ── Direct Speech Tokens Parser ────────────────────────────── */
static std::vector<int32_t> parse_tokens_string(const std::string &s) {
  std::vector<int32_t> ids;
  if (s.find("<|s_") != std::string::npos) {
    size_t pos = 0;
    while (true) {
      pos = s.find("<|s_", pos);
      if (pos == std::string::npos) break;
      size_t end_pos = s.find("|>", pos + 4);
      if (end_pos == std::string::npos) break;
      
      std::string num_str = s.substr(pos + 4, end_pos - (pos + 4));
      try {
        ids.push_back(std::stoi(num_str));
      } catch (...) {}
      pos = end_pos + 2;
    }
  } else {
    // Parse space, comma, or semi-colon separated integers
    size_t start = 0;
    while (start < s.size()) {
      while (start < s.size() && (isspace(s[start]) || s[start] == ',' || s[start] == ';')) {
        start++;
      }
      if (start >= s.size()) break;
      size_t end = start;
      while (end < s.size() && (isdigit(s[end]) || s[end] == '-')) {
        end++;
      }
      if (end > start) {
        try {
          ids.push_back(std::stoi(s.substr(start, end - start)));
        } catch (...) {}
      }
      start = end;
    }
  }
  return ids;
}

/* ── Subprocess Execution Helper ────────────────────────────── */
#ifdef _WIN32
static std::string exec_command(const std::string &cmd) {
  std::string result;
  HANDLE hChildStd_OUT_Rd = NULL;
  HANDLE hChildStd_OUT_Wr = NULL;

  SECURITY_ATTRIBUTES saAttr;
  saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
  saAttr.bInheritHandle = TRUE;
  saAttr.lpSecurityDescriptor = NULL;

  if (!CreatePipe(&hChildStd_OUT_Rd, &hChildStd_OUT_Wr, &saAttr, 0)) {
    fprintf(stderr, "Error: CreatePipe failed\n");
    return "";
  }

  if (!SetHandleInformation(hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0)) {
    CloseHandle(hChildStd_OUT_Rd);
    CloseHandle(hChildStd_OUT_Wr);
    return "";
  }

  STARTUPINFOA siStartInfo;
  PROCESS_INFORMATION piProcInfo;
  ZeroMemory(&siStartInfo, sizeof(STARTUPINFOA));
  siStartInfo.cb = sizeof(STARTUPINFOA);
  siStartInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  siStartInfo.hStdOutput = hChildStd_OUT_Wr;
  siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

  ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

  std::vector<char> cmd_copy(cmd.begin(), cmd.end());
  cmd_copy.push_back('\0');

  BOOL bSuccess = CreateProcessA(
    NULL, 
    cmd_copy.data(), 
    NULL,          
    NULL,          
    TRUE,          
    0,             
    NULL,          
    NULL,          
    &siStartInfo,  
    &piProcInfo    
  );

  // Close the write end of the pipe so we can read EOF
  CloseHandle(hChildStd_OUT_Wr);

  if (!bSuccess) {
    fprintf(stderr, "Error: CreateProcess failed for command: %s (Error code: %lu)\n", cmd.c_str(), GetLastError());
    CloseHandle(hChildStd_OUT_Rd);
    return "";
  }

  char buffer[4096];
  DWORD dwRead;
  while (ReadFile(hChildStd_OUT_Rd, buffer, sizeof(buffer), &dwRead, NULL) && dwRead > 0) {
    result.append(buffer, dwRead);
  }

  WaitForSingleObject(piProcInfo.hProcess, INFINITE);
  CloseHandle(piProcInfo.hProcess);
  CloseHandle(piProcInfo.hThread);
  CloseHandle(hChildStd_OUT_Rd);

  return result;
}
#else
static std::string exec_command(const std::string &cmd) {
  std::string result;
  char buffer[512];
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    fprintf(stderr, "Error: Failed to execute command: %s\n", cmd.c_str());
    return "";
  }
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);
  return result;
}
#endif


/* ── Print Help Documentation ───────────────────────────────── */
static void print_help(const char *prog) {
  fprintf(stderr, "=========================================================================\n");
  fprintf(stderr, " xcodec2 TTS CLI - End-to-end speech generation & token decoding\n");
  fprintf(stderr, "=========================================================================\n\n");
  fprintf(stderr, "Usage Modes:\n\n");
  fprintf(stderr, "1. Server Mode (Inference via llama-server HTTP API):\n");
  fprintf(stderr, "   %s [--url <server_url>] [-c xcodec2.gguf] (-t \"text\" | -f file.txt) [options]\n", prog);
  fprintf(stderr, "   Default URL: http://localhost:8080\n\n");
  fprintf(stderr, "2. CLI Mode (Inference by launching a local llama-cli subprocess):\n");
  fprintf(stderr, "   %s --cli <cli_path> -m <llm.gguf> [-c xcodec2.gguf] (-t \"text\" | -f file.txt) [options]\n\n", prog);
  fprintf(stderr, "3. Direct Token Decoding (Convert pasted speech tokens directly to WAV):\n");
  fprintf(stderr, "   %s --tokens \"<|s_1234|><|s_5678|>\" [-c xcodec2.gguf] [-o output.wav]\n", prog);
  fprintf(stderr, "   Or:\n");
  fprintf(stderr, "   %s --tokens \"1234, 5678, 9012\" [-c xcodec2.gguf] [-o output.wav]\n\n", prog);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -c <path>                 Path to xcodec2 GGUF model (default: xcodec2.gguf)\n");
  fprintf(stderr, "  -o <path>                 Path to output WAV file (default: output.wav)\n");
  fprintf(stderr, "  -t <text>                 Input text for TTS generation (inline string)\n");
  fprintf(stderr, "  -f <path>                 Input text file for TTS generation\n");
  fprintf(stderr, "  -n <max_tokens>           Max tokens to generate (default: 2048)\n");
  fprintf(stderr, "  --temp <temp>             LLM sampling temperature (default: 0.8)\n");
  fprintf(stderr, "  --top-p <top_p>           LLM top-p sampling (default: 0.95)\n");
  fprintf(stderr, "  --top-k <top_k>           LLM top-k sampling (default: 50)\n");
  fprintf(stderr, "  --rep-penalty,            \n");
  fprintf(stderr, "  --repeat-penalty <val>    Repetition penalty to prevent loops (default: 1.1)\n");
  fprintf(stderr, "  --save-tokens <path>      Save generated/parsed speech tokens to a text file\n");
  fprintf(stderr, "  --debug-tokens            Print the speech tokens during execution\n");
  fprintf(stderr, "  --no-decode               Run inference and extract tokens, but do not decode to WAV\n");
  fprintf(stderr, "  --gpu, --vulkan           Use Vulkan GPU backend for xcodec2 decoding (default: CPU)\n");
  fprintf(stderr, "  -h, --help                Show this help message\n");
  fprintf(stderr, "=========================================================================\n");
}

/* ── Main ────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
  std::setlocale(LC_NUMERIC, "C");

  const char *llm_path = nullptr;
  const char *codec_path = "xcodec2.gguf";
  const char *output_path = "output.wav";
  const char *text_inline = nullptr;
  const char *text_file = nullptr;
  const char *save_tokens_path = nullptr;
  
  // Refactored service endpoints/paths
  const char *server_url = "http://localhost:8080";
  const char *cli_path = nullptr;
  const char *tokens_input = nullptr;

  int max_tokens = 2048;
  float temperature = 0.8f; 
  float top_p = 0.95f;
  int top_k = 50;           
  float rep_penalty = 1.1f; // Changed default to 1.1 to match user requests

  bool debug_tokens = false;
  bool do_decode = true;
  bool show_help = false;
  bool use_gpu = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) show_help = true;
    else if (!strcmp(argv[i], "-m") && i + 1 < argc) llm_path = argv[++i];
    else if (!strcmp(argv[i], "-c") && i + 1 < argc) codec_path = argv[++i];
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) output_path = argv[++i];
    else if (!strcmp(argv[i], "-t") && i + 1 < argc) text_inline = argv[++i];
    else if (!strcmp(argv[i], "-f") && i + 1 < argc) text_file = argv[++i];
    else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_tokens = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temperature = atof(argv[++i]);
    else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);
    else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
    else if ((!strcmp(argv[i], "--rep-penalty") || !strcmp(argv[i], "--repeat-penalty")) && i + 1 < argc) rep_penalty = atof(argv[++i]);
    else if (!strcmp(argv[i], "--debug-tokens")) debug_tokens = true;
    else if (!strcmp(argv[i], "--no-decode")) do_decode = false;
    else if (!strcmp(argv[i], "--save-tokens") && i + 1 < argc) save_tokens_path = argv[++i];
    else if (!strcmp(argv[i], "--url") && i + 1 < argc) server_url = argv[++i];
    else if (!strcmp(argv[i], "--cli") && i + 1 < argc) cli_path = argv[++i];
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_input = argv[++i];
    else if (!strcmp(argv[i], "--gpu") || !strcmp(argv[i], "--vulkan")) use_gpu = true;
  }

  // Check help or invalid args
  bool has_tokens_mode = (tokens_input != nullptr);
  bool has_text_mode = (text_inline != nullptr || text_file != nullptr);

  if (show_help || (!has_tokens_mode && !has_text_mode)) {
    print_help(argv[0]);
    return show_help ? 0 : 1;
  }

  if (cli_path && !llm_path && !has_tokens_mode) {
    fprintf(stderr, "Error: -m <llm.gguf> is required when running in CLI mode (--cli).\n");
    return 1;
  }

  if (do_decode && !codec_path) {
    fprintf(stderr, "Error: -c xcodec2.gguf is required unless --no-decode is specified.\n");
    return 1;
  }

  std::vector<int32_t> speech_ids;

  if (has_tokens_mode) {
    /* ── Mode 3: Direct Tokens Decoding Mode ────────────────── */
    printf("Parsing direct speech tokens...\n");
    speech_ids = parse_tokens_string(tokens_input);
  } else {
    // Load input text for Mode 1 & 2
    std::string input_text;
    if (text_file) {
      if (!read_file_to_string(text_file, input_text)) {
        fprintf(stderr, "Error: Failed to read text file %s\n", text_file);
        return 1;
      }
    } else {
      input_text = text_inline;
    }

    // Convert from system codepage to UTF-8 (needed on Windows for CJK text)
    input_text = acp_to_utf8(input_text);

    std::string prompt = build_prompt(input_text);
    std::string generated_text;

    if (cli_path) {
      /* ── Mode 2: CLI Subprocess Mode ───────────────────────── */
      printf("Running inference in CLI mode via %s...\n", cli_path);
      
      // Write prompt to a temporary file to avoid shell escape issues on command line
      std::string temp_prompt_path = "llama_prompt_temp.txt";
      FILE *fp = fopen(temp_prompt_path.c_str(), "wb");
      if (!fp) {
        fprintf(stderr, "Error: Failed to create temporary prompt file.\n");
        return 1;
      }
      fwrite(prompt.c_str(), 1, prompt.size(), fp);
      fclose(fp);

      // Build llama-cli command line
      std::string cmd = "\"";
      cmd += cli_path;
      cmd += "\" -m \"";
      cmd += llm_path;
      cmd += "\" -f ";
      cmd += temp_prompt_path;
      cmd += " -n ";
      cmd += std::to_string(max_tokens);
      cmd += " --temp ";
      cmd += std::to_string(temperature);
      cmd += " --top-p ";
      cmd += std::to_string(top_p);
      cmd += " --top-k ";
      cmd += std::to_string(top_k);
      cmd += " -no-cnv --special --no-display-prompt --log-disable --repeat-penalty ";
      cmd += std::to_string(rep_penalty);
      cmd += " -r \"<|SPEECH_GENERATION_END|>\" -r \"<|eot_id|>\"";
      
      generated_text = exec_command(cmd);
      
      // Clean up temporary prompt file
      remove(temp_prompt_path.c_str());
    } else {
      /* ── Mode 1: Server API Mode ───────────────────────────── */
      // Ensure URL has /completion path
      std::string clean_url = server_url;
      if (clean_url.find("/completion") == std::string::npos) {
        if (clean_url.back() == '/') {
          clean_url += "completion";
        } else {
          clean_url += "/completion";
        }
      }
      printf("Sending inference request to llama-server at %s...\n", clean_url.c_str());

      // Prepare JSON payload with repeat penalty
      std::string escaped_prompt = escape_json(prompt);
      std::string json_payload = "{";
      json_payload += "\"prompt\": \"" + escaped_prompt + "\",";
      json_payload += "\"temperature\": " + std::to_string(temperature) + ",";
      json_payload += "\"top_p\": " + std::to_string(top_p) + ",";
      json_payload += "\"top_k\": " + std::to_string(top_k) + ",";
      json_payload += "\"repeat_penalty\": " + std::to_string(rep_penalty) + ",";
      json_payload += "\"n_predict\": " + std::to_string(max_tokens) + ",";
      json_payload += "\"special\": true,";
      json_payload += "\"stop\": [\"<|SPEECH_GENERATION_END|>\", \"<|eot_id|>\", \"<|end_of_text|>\"]";
      json_payload += "}";

      // Write JSON payload to temporary file to avoid complex shell quoting issues
      std::string temp_json_path = "llama_req_temp.json";
      FILE *fj = fopen(temp_json_path.c_str(), "wb");
      if (!fj) {
        fprintf(stderr, "Error: Failed to create temporary request file.\n");
        return 1;
      }
      fwrite(json_payload.c_str(), 1, json_payload.size(), fj);
      fclose(fj);

      // Call curl to make the POST request
      std::string cmd = "curl -s -X POST -H \"Content-Type: application/json\" -d @" + temp_json_path + " " + clean_url;
      std::string response = exec_command(cmd);

      // Clean up temporary request file
      remove(temp_json_path.c_str());

      if (response.empty()) {
        fprintf(stderr, "Error: Failed to get response from server. Make sure llama-server is running at %s\n", server_url);
        return 1;
      }

      generated_text = parse_json_content(response);
      //  printf("%s",response.c_str());
    }

    /* ── Extract Speech Token IDs ───────────────────────────── */
    size_t pos = 0;
    while (true) {
      pos = generated_text.find("<|s_", pos);
      if (pos == std::string::npos) break;
      size_t end_pos = generated_text.find("|>", pos + 4);
      if (end_pos == std::string::npos) break;
      
      std::string num_str = generated_text.substr(pos + 4, end_pos - (pos + 4));
      try {
        int32_t sid = std::stoi(num_str);
        speech_ids.push_back(sid);
      } catch (...) {}
      pos = end_pos + 2;
    }
  }

  printf("Extracted/Parsed %zu speech tokens\n", speech_ids.size());

  if (debug_tokens && !speech_ids.empty()) {
    printf("Tokens: ");
    for (size_t i = 0; i < speech_ids.size(); i++) {
      printf("%d ", speech_ids[i]);
    }
    printf("\n");
  }

  if (save_tokens_path) {
    FILE *f = fopen(save_tokens_path, "w");
    if (f) {
      for (size_t i = 0; i < speech_ids.size(); i++) {
        fprintf(f, "%d\n", speech_ids[i]);
      }
      fclose(f);
      printf("Saved speech tokens to %s\n", save_tokens_path);
    }
  }

  /* ── 4. Decode speech tokens using local xcodec2 decoder ───── */
  if (do_decode) {
    if (speech_ids.empty()) {
      fprintf(stderr, "Error: No speech tokens generated or parsed, decoding skipped.\n");
      return 1;
    }

    printf("Decoding speech tokens via local xcodec2 GGUF model: %s...\n", codec_path);
    
    // Initialize ggml backend
    ggml_backend_load_all();

    struct xcodec2_model codec = {};
    if (xcodec2_load(&codec, codec_path, use_gpu) != 0) {
      fprintf(stderr, "Error: Failed to load xcodec2 model: %s\n", codec_path);
      return 1;
    }

    float *audio = nullptr;
    int audio_len = 0;
    if (xcodec2_decode(&codec, speech_ids.data(), (int)speech_ids.size(), &audio, &audio_len) == 0) {
      write_wav(output_path, audio, audio_len, codec.hparams.sample_rate);
      free(audio);
    } else {
      fprintf(stderr, "Error: Failed to decode speech tokens.\n");
      xcodec2_free(&codec);
      return 1;
    }
    xcodec2_free(&codec);
  }

  return 0;
}