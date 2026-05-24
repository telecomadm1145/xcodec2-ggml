/* xcodec2.c - main decode function + ISTFT + CLI entry point */
#include "xcodec2.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Declared in xcodec2_graph.c */
extern struct ggml_cgraph *xcodec2_build_graph(struct xcodec2_model *model,
                                               struct ggml_context *ctx,
                                               struct ggml_tensor *indices,
                                               int n_codes);

/* ── ISTFT (same-padding) ────────────────────────────────────── */
static void istft_same(const float *spec_real, const float *spec_imag,
                       int n_fft, int hop, int n_frames, const float *window,
                       float *output, int *out_len) {
  int win_len = n_fft;
  int half = n_fft / 2 + 1;
  int d = win_len - hop;
  int pad_left = d / 2;
  int pad_right = d - pad_left;
  int output_size = (n_frames - 1) * hop + win_len;
  int final_len = output_size - pad_left - (pad_right > 0 ? pad_right : 0);

  float *buf = (float *)calloc(output_size, sizeof(float));
  float *win_env = (float *)calloc(output_size, sizeof(float));
  float *frame = (float *)malloc(n_fft * sizeof(float));

  for (int t = 0; t < n_frames; t++) {
    /* iFFT of complex spectrum for this frame */
    /* spec has half complex bins; do iRFFT manually */
    const float *sr = spec_real + t * half;
    const float *si = spec_imag + t * half;

    for (int n = 0; n < n_fft; n++) {
      double val = 0.0;
      for (int k = 0; k < half; k++) {
        double angle = 2.0 * M_PI * k * n / n_fft;
        val += sr[k] * cos(angle) - si[k] * sin(angle);
        /* Mirror for k > 0 and k < half-1 */
        if (k > 0 && k < half - 1) {
          val += sr[k] * cos(angle) - si[k] * sin(angle);
        }
      }
      frame[n] = (float)(val / n_fft);
    }

    /* Apply window and overlap-add */
    int start = t * hop;
    for (int n = 0; n < win_len; n++) {
      buf[start + n] += frame[n] * window[n];
      win_env[start + n] += window[n] * window[n];
    }
  }

  /* Normalize by window envelope and trim padding */
  *out_len = final_len;
  for (int i = 0; i < final_len; i++) {
    float env = win_env[i + pad_left];
    output[i] = (env > 1e-11f) ? buf[i + pad_left] / env : 0.0f;
  }

  free(buf);
  free(win_env);
  free(frame);
}

static void print_ggml_tensor_stats(struct ggml_cgraph *gf, const char *name) {
  struct ggml_tensor *t = ggml_graph_get_tensor(gf, name);
  if (!t) {
    printf("[GGML] %-25s not found\n", name);
    return;
  }
  
  if (t->type != GGML_TYPE_F32) {
    printf("[GGML] %-25s shape: [%4d, %4d, %4d, %4d] (non-F32 type %d)\n",
           name, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3], t->type);
    return;
  }
  
  int64_t n = ggml_nelements(t);
  float *data = (float *)t->data;
  if (!data) {
    printf("[GGML] %-25s has no data\n", name);
    return;
  }
  
  double sum = 0;
  double sq_sum = 0;
  float min_val = data[0];
  float max_val = data[0];
  
  for (int64_t i = 0; i < n; i++) {
    float val = data[i];
    sum += val;
    sq_sum += (double)val * val;
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }
  
  double mean = sum / n;
  double variance = (sq_sum / n) - (mean * mean);
  double std = sqrt(variance > 0 ? variance : 0);
  
  printf("[GGML] %-25s shape: [%d, %d, %d, %d]\n",
         name, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
  printf("       mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
         mean, std, min_val, max_val);
  
  int print_n = n > 5 ? 5 : (int)n;
  printf("       first %d: [", print_n);
  for (int i = 0; i < print_n; i++) {
    printf("%.6f%s", data[i], (i == print_n - 1) ? "" : ", ");
  }
  printf("]\n");
}

/* ── Main decode function ────────────────────────────────────── */
int xcodec2_decode(struct xcodec2_model *model, const int32_t *codes,
                   int n_codes, float **audio_out, int *out_len) {
  struct xcodec2_hparams *hp = &model->hparams;

  /* Estimate memory needed for compute graph */
  size_t mem_size = 2048ULL * 1024 * 1024; /* 2 GB base for graph */
  mem_size += (size_t)n_codes * hp->vq_dim * sizeof(float) * 20;
  mem_size += (size_t)n_codes * hp->hidden_dim * sizeof(float) * 200;

  struct ggml_init_params params = {
      .mem_size = mem_size,
      .mem_buffer = NULL,
      .no_alloc = true,
  };
  struct ggml_context *ctx = ggml_init(params);
  if (!ctx) {
    fprintf(stderr, "failed to init compute context\n");
    return -1;
  }

  /* Create input tensor with token indices (metadata only, no data pointer yet) */
  struct ggml_tensor *indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_codes);
  ggml_set_name(indices, "input_codes");
  ggml_set_input(indices);

  /* Build graph */
  struct ggml_cgraph *gf = xcodec2_build_graph(model, ctx, indices, n_codes);
  // ggml_graph_dump_dot(gf, 0, "output.dot");

  /* Allocate memory for compute graph on the selected backend */
  ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));
  if (!ggml_gallocr_reserve(gallocr, gf)) {
    fprintf(stderr, "failed to reserve compute graph memory\n");
    ggml_free(ctx);
    ggml_gallocr_free(gallocr);
    return -1;
  }
  if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
    fprintf(stderr, "failed to allocate compute graph memory\n");
    ggml_free(ctx);
    ggml_gallocr_free(gallocr);
    return -1;
  }

  /* Set inputs */
  ggml_backend_tensor_set(indices, codes, 0, n_codes * sizeof(int32_t));

  struct ggml_tensor *pos = ggml_graph_get_tensor(gf, "position_ids");
  if (pos) {
    int32_t *pos_ids = (int32_t *)malloc(n_codes * sizeof(int32_t));
    for (int i = 0; i < n_codes; i++) {
      pos_ids[i] = i;
    }
    ggml_backend_tensor_set(pos, pos_ids, 0, n_codes * sizeof(int32_t));
    free(pos_ids);
  }

  /* Compute on backend */
  ggml_backend_graph_compute(model->backend, gf);
#if 0
  printf("\n=== GGML LAYER-BY-LAYER STATISTICS ===\n");
  print_ggml_tensor_stats(gf, "emb");
  print_ggml_tensor_stats(gf, "fc_post_a");
  print_ggml_tensor_stats(gf, "transposed_4_conv");
  print_ggml_tensor_stats(gf, "embed_conv");
  print_ggml_tensor_stats(gf, "prior_resnet_0");
  print_ggml_tensor_stats(gf, "prior_resnet_1");
  print_ggml_tensor_stats(gf, "transposed_4_attn");
  print_ggml_tensor_stats(gf, "tf0_att_norm");
  print_ggml_tensor_stats(gf, "tf0_qkv");
  print_ggml_tensor_stats(gf, "tf0_q_rope");
  print_ggml_tensor_stats(gf, "tf0_k_rope");
  print_ggml_tensor_stats(gf, "tf0_attn_out");
  print_ggml_tensor_stats(gf, "tf0_attn_proj_out");
  print_ggml_tensor_stats(gf, "tf0_attn_residual");
  print_ggml_tensor_stats(gf, "tf0_ffn_norm");
  print_ggml_tensor_stats(gf, "tf0_ffn_proj");
  print_ggml_tensor_stats(gf, "tf0_ffn_out");
  for (int i = 0; i < hp->n_layers; i++) {
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "transformer_%d", i);
    print_ggml_tensor_stats(gf, name_buf);
  }
  print_ggml_tensor_stats(gf, "transposed_4_postnet");
  print_ggml_tensor_stats(gf, "post_resnet_0");
  print_ggml_tensor_stats(gf, "post_resnet_1");
  print_ggml_tensor_stats(gf, "before_final_norm");
  print_ggml_tensor_stats(gf, "final_ln");
  if (model->n_upsamplers > 0) {
    print_ggml_tensor_stats(gf, "upsampler_in");
    for (int i = 0; i < model->n_upsamplers; i++) {
      char name_buf[64];
      snprintf(name_buf, sizeof(name_buf), "upsampler_conv_%d", i);
      print_ggml_tensor_stats(gf, name_buf);
      snprintf(name_buf, sizeof(name_buf), "upsampler_resnet_%d", i);
      print_ggml_tensor_stats(gf, name_buf);
    }
    print_ggml_tensor_stats(gf, "upsampler_out_proj");
    print_ggml_tensor_stats(gf, "upsampler_out_swish");
  }
  print_ggml_tensor_stats(gf, "stft_pred");
  printf("======================================\n\n");
#endif
  /* Get STFT prediction output: [n_fft+2, T] in ggml */
  struct ggml_tensor *pred = ggml_graph_get_tensor(gf, "stft_pred");
  if (!pred) {
    fprintf(stderr, "stft_pred not found\n");
    ggml_free(ctx);
    return -1;
  }

  int n_fft = hp->n_fft;
  int half = n_fft / 2 + 1; /* 641 */
  int T = pred->ne[1];      /* time frames (ggml layout) */
  /* pred shape: ne[0]=out_dim, ne[1]=T */

  /* Extract mag and phase from prediction
     In PyTorch: x_pred = linear_out.transpose(1,2) -> [B, n_fft+2, T]
     Then mag, p = chunk(2, dim=1) -> each [B, half, T]
     mag = exp(mag), clamp to 1e2
     real = mag * cos(p), imag = mag * sin(p) */
  int out_dim = n_fft + 2;
  float *pred_data = (float *)malloc(ggml_nbytes(pred));
  ggml_backend_tensor_get(pred, pred_data, 0, ggml_nbytes(pred));

  /* pred is [out_dim, T] with ne[0]=T (innermost in ggml memory) */
  /* Actually ggml stores row-major with ne[0] as contiguous dim */
  int n_frames = T; /* this should equal n_codes after conv processing */

  float *spec_real = (float *)malloc(half * n_frames * sizeof(float));
  float *spec_imag = (float *)malloc(half * n_frames * sizeof(float));

  /* Interpret pred as [T, out_dim] in memory (ggml ne[0]=out_dim, ne[1]=T)
     or [out_dim, T] depending on the graph output layout.
     After mul_mat output + transpose operations, let's handle both cases. */
  for (int t = 0; t < n_frames; t++) {
    for (int k = 0; k < half; k++) {
      /* mag part: first half of out_dim */
      float log_mag = pred_data[t * out_dim + k];
      float phase = pred_data[t * out_dim + half + k];
      float mag = expf(log_mag);
      if (mag > 1e2f)
        mag = 1e2f;
      spec_real[t * half + k] = mag * cosf(phase);
      spec_imag[t * half + k] = mag * sinf(phase);
    }
  }

  /* Get ISTFT window */
  float *window = (float *)model->istft_window->data;

  /* Perform ISTFT */
  int max_out = (n_frames - 1) * hp->hop_length + n_fft + 1024;
  float *audio = (float *)malloc(max_out * sizeof(float));
  int audio_len = 0;

  istft_same(spec_real, spec_imag, n_fft, hp->hop_length, n_frames, window,
             audio, &audio_len);
#if 0
  printf("[GGML] reconstructed_audio shape: [%d]\n", audio_len);
#endif
  if (audio_len > 0) {
    double audio_sum = 0, audio_sq_sum = 0;
    float audio_min = audio[0], audio_max = audio[0];
    for (int i = 0; i < audio_len; i++) {
        float val = audio[i];
        audio_sum += val;
        audio_sq_sum += (double)val * val;
        if (val < audio_min) audio_min = val;
        if (val > audio_max) audio_max = val;
    }
    double audio_mean = audio_sum / audio_len;
    double audio_var = (audio_sq_sum / audio_len) - (audio_mean * audio_mean);
    double audio_std = sqrt(audio_var > 0 ? audio_var : 0);
    printf("xcodec2: audio mean=%.6f, std=%.6f, min=%.6f, max=%.6f\n",
           audio_mean, audio_std, audio_min, audio_max);
    int print_len = audio_len > 5 ? 5 : audio_len;
    //printf("       first %d: [", print_len);
    //for (int i = 0; i < print_len; i++) {
    //    printf("%.6f%s", audio[i], (i == print_len - 1) ? "" : ", ");
    //}
    //printf("]\n");
  }

  *audio_out = audio;
  *out_len = audio_len;

  free(pred_data);
  ggml_gallocr_free(gallocr);
  free(spec_real);
  free(spec_imag);
  ggml_free(ctx);
  return 0;
}

#ifdef XCODEC2_STANDALONE
/* ── WAV writer ──────────────────────────────────────────────── */
static void write_wav(const char *path, const float *data, int n_samples,
                      int sr) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    return;
  }
  int16_t *pcm = (int16_t *)malloc(n_samples * sizeof(int16_t));
  for (int i = 0; i < n_samples; i++) {
    float s = data[i] * 32767.0f;
    if (s > 32767.0f)
      s = 32767.0f;
    if (s < -32768.0f)
      s = -32768.0f;
    pcm[i] = (int16_t)s;
  }
  int data_size = n_samples * 2;
  int file_size = 36 + data_size;
  /* RIFF header */
  fwrite("RIFF", 1, 4, f);
  fwrite(&file_size, 4, 1, f);
  fwrite("WAVE", 1, 4, f);
  /* fmt chunk */
  fwrite("fmt ", 1, 4, f);
  int fmt_size = 16;
  fwrite(&fmt_size, 4, 1, f);
  int16_t audio_fmt = 1;
  fwrite(&audio_fmt, 2, 1, f);
  int16_t channels = 1;
  fwrite(&channels, 2, 1, f);
  fwrite(&sr, 4, 1, f);
  int byte_rate = sr * 2;
  fwrite(&byte_rate, 4, 1, f);
  int16_t block_align = 2;
  fwrite(&block_align, 2, 1, f);
  int16_t bps = 16;
  fwrite(&bps, 2, 1, f);
  /* data chunk */
  fwrite("data", 1, 4, f);
  fwrite(&data_size, 4, 1, f);
  fwrite(pcm, 2, n_samples, f);
  fclose(f);
  free(pcm);
  printf("xcodec2: wrote %s (%d samples, %d Hz)\n", path, n_samples, sr);
}

/* ── CLI ─────────────────────────────────────────────────────── */
static void print_usage(const char *prog) {
  fprintf(stderr, "Usage: %s -m <model.gguf> -i <codes.txt> -o <output.wav> [--gpu | --vulkan]\n",
          prog);
  fprintf(stderr, "\ncodes.txt: one integer per line (token indices)\n");
}

int main(int argc, char **argv) {
  const char *model_path = NULL;
  const char *input_path = NULL;
  const char *output_path = "output.wav";
  bool use_gpu = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
      model_path = argv[++i];
    else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
      input_path = argv[++i];
    else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
      output_path = argv[++i];
    else if (strcmp(argv[i], "--gpu") == 0 || strcmp(argv[i], "--vulkan") == 0)
      use_gpu = true;
    else {
      print_usage(argv[0]);
      return 1;
    }
  }
  if (!model_path || !input_path) {
    print_usage(argv[0]);
    return 1;
  }

  /* Load model */
  struct xcodec2_model model = {0};
  if (xcodec2_load(&model, model_path, use_gpu) != 0)
    return 1;

  /* Read codes from text file */
  FILE *f = fopen(input_path, "r");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", input_path);
    return 1;
  }
  int cap = 4096, n = 0;
  int32_t *codes = (int32_t *)malloc(cap * sizeof(int32_t));
  int val;
  while (fscanf(f, "%d", &val) == 1) {
    if (n >= cap) {
      cap *= 2;
      codes = (int32_t *)realloc(codes, cap * sizeof(int32_t));
    }
    codes[n++] = (int32_t)val;
  }
  fclose(f);
  printf("read %d codes\n", n);

  /* Decode */
  float *audio = NULL;
  int audio_len = 0;
  if (xcodec2_decode(&model, codes, n, &audio, &audio_len) != 0) {
    fprintf(stderr, "decode failed\n");
    free(codes);
    xcodec2_free(&model);
    return 1;
  }

  /* Write WAV */
  write_wav(output_path, audio, audio_len, model.hparams.sample_rate);

  free(audio);
  free(codes);
  xcodec2_free(&model);
  return 0;
}
#endif /* XCODEC2_STANDALONE */
