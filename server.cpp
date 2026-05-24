/*
 * server.cpp - xcodec2 HTTP decode server
 *
 * Loads the xcodec2 GGUF model once, serves decode requests over HTTP.
 *
 * Endpoints:
 *   GET  /health          -> {"status":"ok"}
 *   POST /decode          -> WAV audio (binary)
 *     Body (JSON): {"tokens": [1,2,3]}
 *                  {"tokens": "<|s_1|><|s_2|><|s_3|>"}
 *                  {"tokens": "1, 2, 3"}
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
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define CLOSESOCKET closesocket
#define SOCKET_INVALID INVALID_SOCKET
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int socket_t;
#define CLOSESOCKET close
#define SOCKET_INVALID (-1)
#endif

/* ── Global model (loaded once) ─────────────────────────────── */
static struct xcodec2_model g_model = {};
static std::mutex g_model_mutex;

/* ── Token Parser (same logic as xcodec2-tts) ───────────────── */
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
      try { ids.push_back(std::stoi(num_str)); } catch (...) {}
      pos = end_pos + 2;
    }
  } else {
    size_t start = 0;
    while (start < s.size()) {
      while (start < s.size() && (isspace(s[start]) || s[start] == ',' || s[start] == ';' || s[start] == '[' || s[start] == ']'))
        start++;
      if (start >= s.size()) break;
      size_t end = start;
      while (end < s.size() && (isdigit(s[end]) || s[end] == '-'))
        end++;
      if (end > start) {
        try { ids.push_back(std::stoi(s.substr(start, end - start))); } catch (...) {}
      }
      start = end;
    }
  }
  return ids;
}

/* ── Minimal JSON string value extractor ────────────────────── */
// Extracts the value for "key" from a JSON object string.
// Handles string values (returns unescaped), array values (returns raw [...]), and number values.
static std::string json_get_value(const std::string &json, const std::string &key) {
  std::string search = "\"" + key + "\"";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return "";
  pos++;

  // skip whitespace
  while (pos < json.size() && isspace(json[pos])) pos++;
  if (pos >= json.size()) return "";

  if (json[pos] == '"') {
    // String value - extract with unescape
    std::string result;
    size_t i = pos + 1;
    while (i < json.size()) {
      if (json[i] == '"') break;
      if (json[i] == '\\' && i + 1 < json.size()) {
        char next = json[i + 1];
        if (next == '"') { result += '"'; i += 2; }
        else if (next == '\\') { result += '\\'; i += 2; }
        else if (next == 'n') { result += '\n'; i += 2; }
        else if (next == 'r') { result += '\r'; i += 2; }
        else if (next == 't') { result += '\t'; i += 2; }
        else { result += json[i]; i++; }
      } else {
        result += json[i];
        i++;
      }
    }
    return result;
  } else if (json[pos] == '[') {
    // Array value - return raw content including brackets
    int depth = 0;
    size_t start = pos;
    for (size_t i = pos; i < json.size(); i++) {
      if (json[i] == '[') depth++;
      else if (json[i] == ']') { depth--; if (depth == 0) return json.substr(start, i - start + 1); }
    }
    return "";
  } else {
    // Number or boolean
    size_t end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && !isspace(json[end])) end++;
    return json.substr(pos, end - pos);
  }
}

/* ── Build the chat prompt using model's native template ─────── */
static std::string build_prompt(const std::string &text) {
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

/* ── WAV builder (returns in-memory WAV) ────────────────────── */
static std::vector<uint8_t> build_wav(const float *data, int n, int sr) {
  std::vector<int16_t> pcm(n);
  for (int i = 0; i < n; i++) {
    float s = data[i] * 32767.0f;
    s = s > 32767.f ? 32767.f : (s < -32768.f ? -32768.f : s);
    pcm[i] = (int16_t)s;
  }

  int ds = n * 2, fs = 36 + ds;
  int16_t fmt = 1, ch = 1, ba = 2, bps = 16;
  int br = sr * 2;
  int cs = 16;

  std::vector<uint8_t> wav;
  wav.reserve(44 + ds);

  auto write_bytes = [&](const void *p, size_t sz) {
    const uint8_t *b = (const uint8_t *)p;
    wav.insert(wav.end(), b, b + sz);
  };

  write_bytes("RIFF", 4);
  write_bytes(&fs, 4);
  write_bytes("WAVEfmt ", 8);
  write_bytes(&cs, 4);
  write_bytes(&fmt, 2);
  write_bytes(&ch, 2);
  write_bytes(&sr, 4);
  write_bytes(&br, 4);
  write_bytes(&ba, 2);
  write_bytes(&bps, 2);
  write_bytes("data", 4);
  write_bytes(&ds, 4);
  write_bytes(pcm.data(), ds);

  return wav;
}

/* ── HTTP helpers ───────────────────────────────────────────── */
static void send_response(socket_t sock, int status_code, const char *status_text,
                          const char *content_type, const void *body, size_t body_len) {
  char header[1024];
  int hlen = snprintf(header, sizeof(header),
    "HTTP/1.1 %d %s\r\n"
    "Content-Type: %s\r\n"
    "Content-Length: %zu\r\n"
    "Connection: close\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type\r\n"
    "\r\n",
    status_code, status_text, content_type, body_len);

  send(sock, header, hlen, 0);
  if (body && body_len > 0) {
    send(sock, (const char *)body, (int)body_len, 0);
  }
}

static void send_json_response(socket_t sock, int code, const char *status, const std::string &json) {
  send_response(sock, code, status, "application/json", json.c_str(), json.size());
}

static void send_error(socket_t sock, int code, const char *status, const char *message) {
  std::string json = "{\"error\": \"";
  json += message;
  json += "\"}";
  send_json_response(sock, code, status, json);
}

/* ── Read full HTTP request ─────────────────────────────────── */
static std::string recv_request(socket_t sock) {
  std::string data;
  char buf[8192];

  // Read headers first
  while (true) {
    int n = recv(sock, buf, sizeof(buf), 0);
    if (n <= 0) break;
    data.append(buf, n);
    if (data.find("\r\n\r\n") != std::string::npos) break;
  }

  // Check Content-Length for body
  size_t cl_pos = data.find("Content-Length:");
  if (cl_pos == std::string::npos) cl_pos = data.find("content-length:");
  if (cl_pos != std::string::npos) {
    size_t val_start = cl_pos + 15;
    while (val_start < data.size() && data[val_start] == ' ') val_start++;
    size_t val_end = data.find("\r\n", val_start);
    int content_length = atoi(data.substr(val_start, val_end - val_start).c_str());

    size_t header_end = data.find("\r\n\r\n") + 4;
    int body_received = (int)data.size() - (int)header_end;

    while (body_received < content_length) {
      int n = recv(sock, buf, sizeof(buf), 0);
      if (n <= 0) break;
      data.append(buf, n);
      body_received += n;
    }
  }

  return data;
}

/* ── Handle a single client connection ──────────────────────── */
static void handle_client(socket_t sock) {
  std::string request = recv_request(sock);
  if (request.empty()) {
    CLOSESOCKET(sock);
    return;
  }

  // Parse method and path
  std::string method, path;
  {
    size_t sp1 = request.find(' ');
    if (sp1 != std::string::npos) {
      method = request.substr(0, sp1);
      size_t sp2 = request.find(' ', sp1 + 1);
      if (sp2 != std::string::npos) {
        path = request.substr(sp1 + 1, sp2 - sp1 - 1);
      }
    }
  }

  // OPTIONS (CORS preflight)
  if (method == "OPTIONS") {
    send_response(sock, 204, "No Content", "text/plain", nullptr, 0);
    CLOSESOCKET(sock);
    return;
  }

  // GET /health
  if (method == "GET" && (path == "/health" || path == "/")) {
    std::string json = "{\"status\":\"ok\"}";
    send_json_response(sock, 200, "OK", json);
    CLOSESOCKET(sock);
    return;
  }

  // POST /decode
  if (method == "POST" && path == "/decode") {
    // Extract body
    size_t body_start = request.find("\r\n\r\n");
    if (body_start == std::string::npos) {
      send_error(sock, 400, "Bad Request", "Missing request body");
      CLOSESOCKET(sock);
      return;
    }
    std::string body = request.substr(body_start + 4);

    // Parse tokens from JSON body
    std::string tokens_val = json_get_value(body, "tokens");
    if (tokens_val.empty()) {
      send_error(sock, 400, "Bad Request", "Missing 'tokens' field in JSON body");
      CLOSESOCKET(sock);
      return;
    }

    std::vector<int32_t> speech_ids = parse_tokens_string(tokens_val);
    if (speech_ids.empty()) {
      send_error(sock, 400, "Bad Request", "No valid tokens parsed");
      CLOSESOCKET(sock);
      return;
    }

    printf("[decode] received %zu tokens, decoding...\n", speech_ids.size());

    // Decode with mutex (model is not thread-safe)
    float *audio = nullptr;
    int audio_len = 0;
    int rc;
    {
      std::lock_guard<std::mutex> lock(g_model_mutex);
      rc = xcodec2_decode(&g_model, speech_ids.data(), (int)speech_ids.size(), &audio, &audio_len);
    }

    if (rc != 0 || audio == nullptr) {
      send_error(sock, 500, "Internal Server Error", "Decode failed");
      CLOSESOCKET(sock);
      return;
    }

    // Build WAV and send
    std::vector<uint8_t> wav = build_wav(audio, audio_len, g_model.hparams.sample_rate);
    free(audio);

    printf("[decode] done: %d samples, %zu bytes WAV\n", audio_len, wav.size());
    send_response(sock, 200, "OK", "audio/wav", wav.data(), wav.size());
    CLOSESOCKET(sock);
    return;
  }

  // POST /tts
  if (method == "POST" && (path == "/tts" || path == "/generate")) {
    // Extract body
    size_t body_start = request.find("\r\n\r\n");
    if (body_start == std::string::npos) {
      send_error(sock, 400, "Bad Request", "Missing request body");
      CLOSESOCKET(sock);
      return;
    }
    std::string body = request.substr(body_start + 4);

    // Parse parameters from JSON body
    std::string text = json_get_value(body, "text");
    if (text.empty()) {
      send_error(sock, 400, "Bad Request", "Missing 'text' field in JSON body");
      CLOSESOCKET(sock);
      return;
    }

    std::string llama_url = json_get_value(body, "llama_url");
    if (llama_url.empty()) {
      llama_url = "http://localhost:8080";
    }

    std::string temp_val = json_get_value(body, "temperature");
    float temperature = temp_val.empty() ? 0.8f : atof(temp_val.c_str());

    std::string topp_val = json_get_value(body, "top_p");
    float top_p = topp_val.empty() ? 0.95f : atof(topp_val.c_str());

    std::string topk_val = json_get_value(body, "top_k");
    int top_k = topk_val.empty() ? 50 : atoi(topk_val.c_str());

    std::string rep_val = json_get_value(body, "repeat_penalty");
    if (rep_val.empty()) rep_val = json_get_value(body, "rep_penalty");
    float rep_penalty = rep_val.empty() ? 1.1f : atof(rep_val.c_str());

    std::string max_tokens_val = json_get_value(body, "max_tokens");
    int max_tokens = max_tokens_val.empty() ? 2048 : atoi(max_tokens_val.c_str());

    printf("[tts] received text: \"%s\", generating tokens via %s...\n", text.c_str(), llama_url.c_str());

    // Build the llama prompt
    std::string prompt = build_prompt(text);

    // Prepare JSON payload with repeat penalty
    std::string clean_url = llama_url;
    if (clean_url.find("/completion") == std::string::npos) {
      if (clean_url.back() == '/') {
        clean_url += "completion";
      } else {
        clean_url += "/completion";
      }
    }

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

    // Use a thread-safe / unique temporary filename to support concurrent requests.
    static std::mutex temp_file_mutex;
    std::string temp_json_path;
    {
      std::lock_guard<std::mutex> lock(temp_file_mutex);
      static int temp_counter = 0;
      temp_json_path = "llama_req_server_temp_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + "_" + std::to_string(temp_counter++) + ".json";
    }

    FILE *fj = fopen(temp_json_path.c_str(), "wb");
    if (!fj) {
      send_error(sock, 500, "Internal Server Error", "Failed to create temporary request file");
      CLOSESOCKET(sock);
      return;
    }
    fwrite(json_payload.c_str(), 1, json_payload.size(), fj);
    fclose(fj);

    // Call curl to make the POST request
    std::string cmd = "curl -s -X POST -H \"Content-Type: application/json\" -d @" + temp_json_path + " " + clean_url;
    std::string response = exec_command(cmd);

    // Clean up temporary request file
    remove(temp_json_path.c_str());

    if (response.empty()) {
      send_error(sock, 500, "Internal Server Error", "Failed to connect to llama-server");
      CLOSESOCKET(sock);
      return;
    }

    std::string generated_text = parse_json_content(response);
    std::vector<int32_t> speech_ids;

    /* Extract Speech Token IDs */
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

    if (speech_ids.empty()) {
      send_error(sock, 500, "Internal Server Error", "No speech tokens generated by llama-server");
      CLOSESOCKET(sock);
      return;
    }

    printf("[tts] generated %zu tokens, decoding...\n", speech_ids.size());

    // Decode with mutex (model is not thread-safe)
    float *audio = nullptr;
    int audio_len = 0;
    int rc;
    {
      std::lock_guard<std::mutex> lock(g_model_mutex);
      rc = xcodec2_decode(&g_model, speech_ids.data(), (int)speech_ids.size(), &audio, &audio_len);
    }

    if (rc != 0 || audio == nullptr) {
      send_error(sock, 500, "Internal Server Error", "Decode failed");
      CLOSESOCKET(sock);
      return;
    }

    // Build WAV and send
    std::vector<uint8_t> wav = build_wav(audio, audio_len, g_model.hparams.sample_rate);
    free(audio);

    printf("[tts] done: %d samples, %zu bytes WAV\n", audio_len, wav.size());
    send_response(sock, 200, "OK", "audio/wav", wav.data(), wav.size());
    CLOSESOCKET(sock);
    return;
  }

  // 404
  send_error(sock, 404, "Not Found", "Unknown endpoint");
  CLOSESOCKET(sock);
}

/* ── Print Help ─────────────────────────────────────────────── */
static void print_help(const char *prog) {
  fprintf(stderr, "=========================================================================\n");
  fprintf(stderr, " xcodec2 Decode Server - HTTP API for speech token decoding\n");
  fprintf(stderr, "=========================================================================\n\n");
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [-c xcodec2.gguf] [--host 0.0.0.0] [--port 8082]\n\n", prog);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -c <path>          Path to xcodec2 GGUF model (default: xcodec2.gguf)\n");
  fprintf(stderr, "  --host <addr>      Listen address (default: 0.0.0.0)\n");
  fprintf(stderr, "  --port <port>      Listen port (default: 8082)\n");
  fprintf(stderr, "  --gpu, --vulkan    Use Vulkan GPU backend for xcodec2 decoding (default: CPU)\n");
  fprintf(stderr, "  -h, --help         Show this help message\n\n");
  fprintf(stderr, "Endpoints:\n");
  fprintf(stderr, "  GET  /health       Health check, returns {\"status\":\"ok\"}\n");
  fprintf(stderr, "  POST /decode       Decode speech tokens to WAV audio\n");
  fprintf(stderr, "    Body (JSON):\n");
  fprintf(stderr, "      {\"tokens\": [1234, 5678, 9012]}              Integer array\n");
  fprintf(stderr, "      {\"tokens\": \"<|s_1234|><|s_5678|><|s_9012|>\"}  Token string\n");
  fprintf(stderr, "      {\"tokens\": \"1234, 5678, 9012\"}              Comma-separated\n");
  fprintf(stderr, "    Response: audio/wav binary\n");
  fprintf(stderr, "  POST /tts          Perform end-to-end TTS (calls llama-server to get tokens and decodes to WAV)\n");
  fprintf(stderr, "    Body (JSON):\n");
  fprintf(stderr, "      {\"text\": \"hello world\", \"llama_url\": \"http://localhost:8080\"}\n");
  fprintf(stderr, "    Response: audio/wav binary\n");
  fprintf(stderr, "=========================================================================\n");
}

/* ── Main ────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
  const char *codec_path = "xcodec2.gguf";
  const char *host = "0.0.0.0";
  int port = 8082;
  bool use_gpu = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      print_help(argv[0]);
      return 0;
    }
    else if (!strcmp(argv[i], "-c") && i + 1 < argc) codec_path = argv[++i];
    else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
    else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gpu") || !strcmp(argv[i], "--vulkan")) use_gpu = true;
  }

  /* ── Init ggml backends ───────────────────────────────────── */
  printf("Loading xcodec2 model: %s\n", codec_path);
  ggml_backend_load_all();

  if (xcodec2_load(&g_model, codec_path, use_gpu) != 0) {
    fprintf(stderr, "Error: Failed to load xcodec2 model: %s\n", codec_path);
    return 1;
  }
  printf("Model loaded successfully. sample_rate=%d\n", g_model.hparams.sample_rate);

  /* ── Init sockets ─────────────────────────────────────────── */
#ifdef _WIN32
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    fprintf(stderr, "Error: WSAStartup failed\n");
    return 1;
  }
#endif

  socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd == SOCKET_INVALID) {
    fprintf(stderr, "Error: Failed to create socket\n");
    return 1;
  }

  // Allow port reuse
  int opt = 1;
#ifdef _WIN32
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));
#else
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);

  if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
    // Fallback to INADDR_ANY
    addr.sin_addr.s_addr = INADDR_ANY;
  }

  if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    fprintf(stderr, "Error: Failed to bind to %s:%d\n", host, port);
    CLOSESOCKET(server_fd);
    return 1;
  }

  if (listen(server_fd, 16) < 0) {
    fprintf(stderr, "Error: Failed to listen\n");
    CLOSESOCKET(server_fd);
    return 1;
  }

  printf("\n=========================================================================\n");
  printf(" xcodec2-server listening on http://%s:%d\n", host, port);
  printf("   POST /decode   - decode speech tokens to WAV\n");
  printf("   POST /tts      - end-to-end text-to-speech to WAV\n");
  printf("   GET  /health   - health check\n");
  printf("=========================================================================\n\n");

  /* ── Accept loop ──────────────────────────────────────────── */
  while (true) {
    struct sockaddr_in client_addr;
    int client_len = sizeof(client_addr);
    socket_t client = accept(server_fd, (struct sockaddr *)&client_addr, 
#ifdef _WIN32
      &client_len
#else
      (socklen_t *)&client_len
#endif
    );

    if (client == SOCKET_INVALID) {
      fprintf(stderr, "Warning: accept() failed, continuing...\n");
      continue;
    }

    // Handle in a detached thread for basic concurrency
    std::thread(handle_client, client).detach();
  }

  CLOSESOCKET(server_fd);
  xcodec2_free(&g_model);

#ifdef _WIN32
  WSACleanup();
#endif

  return 0;
}
