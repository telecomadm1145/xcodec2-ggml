/*
 * main.cpp - Full TTS pipeline: LLM (llama.cpp) -> xcodec2 decoder -> WAV
 *
 * Flow:
 *   1. Build chat prompt with text
 *   2. LLM generates speech tokens <|s_XXXXX|> autoregressively
 *   3. Extract integer IDs from speech tokens
 *   4. xcodec2 decodes IDs to audio waveform
 *   5. Write WAV file
 */

#include "llama.h"
#include "ggml.h"

extern "C" {
#include "xcodec2.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <locale>
#include <regex>

/* ── WAV writer ──────────────────────────────────────────────── */
static void write_wav(const char * path, const float * data, int n, int sr) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    std::vector<int16_t> pcm(n);
    for (int i = 0; i < n; i++) {
        float s = data[i] * 32767.0f;
        s = s > 32767.f ? 32767.f : (s < -32768.f ? -32768.f : s);
        pcm[i] = (int16_t)s;
    }
    int ds = n * 2, fs = 36 + ds;
    int16_t fmt = 1, ch = 1, ba = 2, bps = 16;
    int br = sr * 2;
    fwrite("RIFF", 1, 4, f); fwrite(&fs, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    int cs = 16; fwrite(&cs, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);  fwrite(&br, 4, 1, f);
    fwrite(&ba, 2, 1, f);  fwrite(&bps, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&ds, 4, 1, f);
    fwrite(pcm.data(), 2, n, f);
    fclose(f);
    printf("wrote %s (%d samples, %d Hz)\n", path, n, sr);
}

/* ── Extract speech token ID from string like "<|s_12345|>" ── */
static bool parse_speech_token(const std::string & s, int32_t & out) {
    // Match <|s_DIGITS|>
    if (s.size() >= 7 && s.substr(0, 4) == "<|s_" && s.substr(s.size() - 2) == "|>") {
        std::string num = s.substr(4, s.size() - 6);
        try { out = std::stoi(num); return true; }
        catch (...) { return false; }
    }
    return false;
}

/* ── Build the chat prompt ───────────────────────────────────── */
static std::string build_prompt(const std::string & text) {
    // Format matching the Llasa chat template
    std::string formatted = "<|TEXT_UNDERSTANDING_START|>" + text + "<|TEXT_UNDERSTANDING_END|>";
    // Using the standard chat format for Llasa models
    std::string prompt;
    prompt += "<|begin_of_text|>";
    prompt += "<|start_header_id|>user<|end_header_id|>\n\n";
    prompt += "Convert the text to speech:" + formatted;
    prompt += "<|eot_id|>";
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    prompt += "<|SPEECH_GENERATION_START|>";
    return prompt;
}

/* ── Main ────────────────────────────────────────────────────── */
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * llm_path = nullptr;
    const char * codec_path = nullptr;
    const char * output_path = "output.wav";
    const char * text = nullptr;
    int n_gpu = 99;
    int max_tokens = 2048;
    float temperature = 0.8f;
    float top_p = 1.0f;
    float rep_penalty = 1.1f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m")  && i+1<argc) llm_path   = argv[++i];
        else if (!strcmp(argv[i], "-c")  && i+1<argc) codec_path = argv[++i];
        else if (!strcmp(argv[i], "-o")  && i+1<argc) output_path= argv[++i];
        else if (!strcmp(argv[i], "-t")  && i+1<argc) text       = argv[++i];
        else if (!strcmp(argv[i], "-ngl")&& i+1<argc) n_gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")  && i+1<argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1<argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i+1<argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--rep-penalty") && i+1<argc) rep_penalty = atof(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s -m llm.gguf -c xcodec2.gguf -t \"text\" [-o out.wav]\n", argv[0]);
            return 1;
        }
    }
    if (!llm_path || !codec_path || !text) {
        fprintf(stderr, "Usage: %s -m llm.gguf -c xcodec2.gguf -t \"text\" [-o out.wav]\n", argv[0]);
        return 1;
    }

    /* ── Load backends ─────────────────────────────────────────── */
    ggml_backend_load_all();

    /* ── 1. Load LLM ───────────────────────────────────────────── */
    printf("Loading LLM from %s ...\n", llm_path);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;
    llama_model * llm = llama_model_load_from_file(llm_path, mparams);
    if (!llm) { fprintf(stderr, "failed to load LLM\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(llm);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = max_tokens + 512;
    cparams.n_batch = 512;
    llama_context * lctx = llama_init_from_model(llm, cparams);
    if (!lctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    /* ── 2. Load xcodec2 decoder ───────────────────────────────── */
    printf("Loading xcodec2 from %s ...\n", codec_path);
    struct xcodec2_model codec = {};
    if (xcodec2_load(&codec, codec_path) != 0) return 1;

    /* ── 3. Tokenize prompt ────────────────────────────────────── */
    std::string prompt = build_prompt(text);
    printf("Prompt: %s\n", prompt.c_str());

    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), n_prompt, true, true);
    printf("Prompt tokens: %d\n", n_prompt);

    /* Find the SPEECH_GENERATION_END token ID */
    llama_token speech_end_id = -1;
    {
        const char * end_str = "<|SPEECH_GENERATION_END|>";
        int n = -llama_tokenize(vocab, end_str, strlen(end_str), nullptr, 0, false, false);
        if (n == 1) {
            llama_token buf;
            llama_tokenize(vocab, end_str, strlen(end_str), &buf, 1, false, false);
            speech_end_id = buf;
        }
        printf("SPEECH_GENERATION_END token id: %d\n", speech_end_id);
    }

    /* ── 4. Initialize sampler ─────────────────────────────────── */
    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);

    if (temperature > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, rep_penalty, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    /* ── 5. Generate speech tokens ─────────────────────────────── */
    printf("Generating speech tokens...\n");
    std::vector<int32_t> speech_ids;

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    int n_decode = 0;
    for (int n_pos = 0; n_pos + batch.n_tokens < max_tokens; ) {
        if (llama_decode(lctx, batch)) {
            fprintf(stderr, "decode failed at pos %d\n", n_pos);
            break;
        }
        n_pos += batch.n_tokens;

        llama_token tok = llama_sampler_sample(smpl, lctx, -1);

        /* Check for end conditions */
        if (tok == speech_end_id || llama_vocab_is_eog(vocab, tok)) {
            printf("\n[end token at %d decoded tokens]\n", n_decode);
            break;
        }

        /* Convert token to string and check if it's a speech token */
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            int32_t sid;
            if (parse_speech_token(piece, sid)) {
                speech_ids.push_back(sid);
                if (speech_ids.size() % 50 == 0)
                    printf("  ... %zu speech tokens\n", speech_ids.size());
            }
        }

        batch = llama_batch_get_one(&tok, 1);
        n_decode++;
    }

    printf("Generated %zu speech token IDs\n", speech_ids.size());

    /* ── 6. Decode speech tokens to audio ──────────────────────── */
    if (speech_ids.empty()) {
        fprintf(stderr, "no speech tokens generated\n");
    } else {
        printf("Decoding to audio...\n");
        float * audio = nullptr;
        int audio_len = 0;
        if (xcodec2_decode(&codec, speech_ids.data(), (int)speech_ids.size(),
                           &audio, &audio_len) == 0) {
            write_wav(output_path, audio, audio_len, codec.hparams.sample_rate);
            free(audio);
        } else {
            fprintf(stderr, "xcodec2 decode failed\n");
        }
    }

    /* ── Cleanup ───────────────────────────────────────────────── */
    llama_sampler_free(smpl);
    llama_free(lctx);
    llama_model_free(llm);
    xcodec2_free(&codec);

    return 0;
}
