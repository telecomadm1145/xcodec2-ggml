/*
 * xcodec2 decoder (token -> audio) using ggml + GGUF
 *
 * Decode pipeline:
 *   vq_code indices -> FSQ codebook lookup -> fc_post_a(Linear 2048->1024)
 *   -> VocosBackbone(embed Conv1d -> 2xResnetBlock -> 12xTransformer -> 2xResnetBlock -> LayerNorm)
 *   -> ISTFTHead(Linear -> mag/phase -> iSTFT) -> waveform
 */
#ifndef XCODEC2_H
#define XCODEC2_H

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Hyperparameters ─────────────────────────────────────────── */
struct xcodec2_hparams {
    int hidden_dim;       // 1024
    int n_layers;         // 12 transformer layers
    int n_heads;          // 16
    int head_dim;         // 64
    int rope_dim;         // 64
    int hop_length;       // 320
    int sample_rate;      // 16000
    int n_fft;            // 1280
    int vq_dim;           // 2048
    int codebook_size;    // 65536
    int codebook_dim;     // 8
};

/* ── ResnetBlock weights ─────────────────────────────────────── */
struct xcodec2_resnet_block {
    struct ggml_tensor * norm1_w;   // [C]
    struct ggml_tensor * norm1_b;   // [C]
    struct ggml_tensor * conv1_w;   // [C, C, 3]
    struct ggml_tensor * conv1_b;   // [C]
    struct ggml_tensor * norm2_w;   // [C]
    struct ggml_tensor * norm2_b;   // [C]
    struct ggml_tensor * conv2_w;   // [C, C, 3]
    struct ggml_tensor * conv2_b;   // [C]
};

/* ── TransformerBlock weights ────────────────────────────────── */
struct xcodec2_transformer_block {
    struct ggml_tensor * att_norm_w;   // [D] RMSNorm
    struct ggml_tensor * ffn_norm_w;   // [D] RMSNorm
    struct ggml_tensor * c_attn_w;     // [3D, D] QKV projection
    struct ggml_tensor * c_proj_w;     // [D, D]  output projection
    struct ggml_tensor * mlp_fc1_w;    // [4D, D]
    struct ggml_tensor * mlp_fc2_w;    // [D, 4D]
};

/* ── Full model ──────────────────────────────────────────────── */
struct xcodec2_model {
    struct xcodec2_hparams hparams;

    /* FSQ codebook [codebook_size, vq_dim] - precomputed */
    struct ggml_tensor * fsq_codebook;

    /* fc_post_a: Linear(2048, 1024) */
    struct ggml_tensor * fc_post_a_w;
    struct ggml_tensor * fc_post_a_b;

    /* Backbone embed: Conv1d(1024, 1024, k=7, p=3) */
    struct ggml_tensor * embed_w;
    struct ggml_tensor * embed_b;

    /* prior_net: 2x ResnetBlock */
    struct xcodec2_resnet_block prior_net[2];

    /* transformers: 12x TransformerBlock */
    struct xcodec2_transformer_block transformer[12];

    /* final LayerNorm */
    struct ggml_tensor * final_ln_w;
    struct ggml_tensor * final_ln_b;

    /* post_net: 2x ResnetBlock */
    struct xcodec2_resnet_block post_net[2];

    /* Upsampler (optional, for 44.1kHz variants) */
    int n_upsamplers; // 0 if no upsampler
    struct {
        struct ggml_tensor * conv_w; // [C_in, C_out, K]
        struct ggml_tensor * conv_b; // [C_out]
        struct xcodec2_resnet_block resnet;
    } upsampler[4];
    struct ggml_tensor * upsampler_out_proj_w;
    struct ggml_tensor * upsampler_out_proj_b;

    /* ISTFTHead: Linear(1024, n_fft+2) */
    struct ggml_tensor * head_out_w;
    struct ggml_tensor * head_out_b;

    /* ISTFT hann window [n_fft] */
    struct ggml_tensor * istft_window;

    /* ggml contexts */
    struct ggml_context   * ctx_data;    // holds weight data
    struct gguf_context   * ctx_gguf;    // GGUF file context

    /* backend resources */
    ggml_backend_t          backend;     // backend used for computation
    ggml_backend_buffer_t   buffer_w;    // backend buffer for weights
};

/* ── API ─────────────────────────────────────────────────────── */

/* Load model from GGUF file. Returns 0 on success. */
int xcodec2_load(struct xcodec2_model * model, const char * path, bool use_gpu);

/* Free model resources. */
void xcodec2_free(struct xcodec2_model * model);

/*
 * Decode token indices to audio waveform.
 *   codes:      input token indices, length = n_codes
 *   audio_out:  output buffer (caller allocates), length returned via out_len
 *   Returns 0 on success.
 */
int xcodec2_decode(struct xcodec2_model * model,
                   const int32_t * codes, int n_codes,
                   float ** audio_out, int * out_len);

#ifdef __cplusplus
}
#endif
#endif /* XCODEC2_H */
