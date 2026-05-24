/* xcodec2_graph.c - build ggml compute graph for decoder inference */
#include "xcodec2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Debug Helper ────────────────────────────────────────────── */
#if 0
#define DEBUG_PRINT_SHAPE(name, t)                                             \
  printf("[DEBUG-SHAPE] %-20s: [%4d, %4d, %4d, %4d]\n", (name),                \
         (int)(t)->ne[0], (int)(t)->ne[1], (int)(t)->ne[2], (int)(t)->ne[3])
#else
#define DEBUG_PRINT_SHAPE(a,b)
#endif

/* ── helpers ─────────────────────────────────────────────────── */

/* swish nonlinearity used by ResnetBlock */

/* swish nonlinearity used by ResnetBlock */
static struct ggml_tensor *swish(struct ggml_context *ctx,
                                 struct ggml_tensor *x) {
  return ggml_silu(ctx, x);
}
/* Conv1d: weight [Cout, Cin, K], input [Cin, T] -> [Cout, T'] */
static struct ggml_tensor *conv1d(struct ggml_context *ctx,
                                  struct ggml_tensor *x, struct ggml_tensor *w,
                                  struct ggml_tensor *b, int stride, int pad,
                                  int dilation) {
  /* 【修复】：GGML 的 im2col 强制要求卷积核 (w) 必须是 F16 类型。
     如果加载的模型是 F32，我们需要在这里建一个强转节点。 */
  struct ggml_tensor *w_cast = w;
  if (w->type != GGML_TYPE_F16) {
    w_cast = ggml_cast(ctx, w, GGML_TYPE_F16);
  }

  struct ggml_tensor *y = ggml_conv_1d(ctx, w_cast, x, stride, pad, dilation);

  if (b) {
    /* b [Cout] -> broadcast add over T dimension */
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
  }
  return y;
}

/* ConvTranspose1d: weight [Cout, Cin, K], input [Cin, T] -> [Cout, T'] */
static struct ggml_tensor *conv_transpose_1d(struct ggml_context *ctx,
                                             struct ggml_tensor *x,
                                             struct ggml_tensor *w,
                                             struct ggml_tensor *b, int stride,
                                             int pad) {
  /* w is [K, Cout, Cin, 1] in GGML, effectively [K, Cout, Cin] */
  struct ggml_tensor *w_cast = w;
  if (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32) {
    w_cast =
        ggml_cast(ctx, w, GGML_TYPE_F32); /* Assuming F32 for conv_transpose */
  }

  struct ggml_tensor *y = ggml_conv_transpose_1d(ctx, w_cast, x, stride, 0, 1);

  if (pad > 0) {
    int Cout = y->ne[1];
    int T_out = y->ne[0];
    y = ggml_view_2d(ctx, y, T_out - 2 * pad, Cout, y->nb[1], pad * y->nb[0]);
    y = ggml_cont(ctx, y);
  }

  if (b) {
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
  }
  return ggml_cont(ctx, y);
}
static struct ggml_tensor *group_norm(struct ggml_context *ctx,
                                      struct ggml_tensor *x,
                                      struct ggml_tensor *w,
                                      struct ggml_tensor *b) {
  int T = x->ne[0];
  int C = x->ne[1];

  // 1. Reshape to 3D: [T, 1, C] so that the channel dimension C is at ne[2] as expected by ggml_group_norm
  struct ggml_tensor *x_3d = ggml_reshape_3d(ctx, x, T, 1, C);

  // 2. Apply group norm on the 3D tensor
  struct ggml_tensor *y_3d = ggml_group_norm(ctx, x_3d, 32, 1e-6f);

  // 3. Reshape back to 2D: [T, C]
  struct ggml_tensor *y = ggml_reshape_2d(ctx, y_3d, T, C);

  // 4. Scale and shift
  y = ggml_mul(ctx, y, ggml_reshape_2d(ctx, w, 1, C));
  y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C));

  return y;
}

/* ── ResnetBlock ─────────────────────────────────────────────── */
static struct ggml_tensor *resnet_block(struct ggml_context *ctx,
                                        struct ggml_tensor *x,
                                        struct xcodec2_resnet_block *blk) {
  struct ggml_tensor *h = x;
  h = group_norm(ctx, h, blk->norm1_w, blk->norm1_b);
  h = swish(ctx, h);
  h = conv1d(ctx, h, blk->conv1_w, blk->conv1_b, 1, 1, 1);
  h = group_norm(ctx, h, blk->norm2_w, blk->norm2_b);
  h = swish(ctx, h);
  h = conv1d(ctx, h, blk->conv2_w, blk->conv2_b, 1, 1, 1);
  return ggml_add(ctx, x, h);
}

/* ── RMSNorm ─────────────────────────────────────────────────── */
static struct ggml_tensor *rms_norm(struct ggml_context *ctx,
                                    struct ggml_tensor *x,
                                    struct ggml_tensor *w) {
  x = ggml_rms_norm(ctx, x, 1e-6f);
  return ggml_mul(ctx, x, w);
}

/* ── TransformerBlock ────────────────────────────────────────── */
/* ── TransformerBlock ────────────────────────────────────────── */
static struct ggml_tensor *
transformer_block(struct ggml_context *ctx, struct ggml_tensor *x,
                  struct ggml_tensor *pos, struct xcodec2_transformer_block *tb,
                  int n_heads, int head_dim, int rope_dim, int layer_idx) {
  int D = x->ne[0];
  int T = x->ne[1];

  if (layer_idx == 0) {
    // printf("--- Transformer Block 0 Start ---\n");
    DEBUG_PRINT_SHAPE("x_in_transformer", x);
  }

  /* Pre-norm attention */
  struct ggml_tensor *xn = rms_norm(ctx, x, tb->att_norm_w);
  if (layer_idx == 0) {
    ggml_set_name(xn, "tf0_att_norm");
  }

  /* QKV projection: [T, D] x [D, 3D]^T -> [T, 3D] */
  struct ggml_tensor *qkv = ggml_mul_mat(ctx, tb->c_attn_w, xn);
  if (layer_idx == 0) {
    DEBUG_PRINT_SHAPE("qkv", qkv);
    ggml_set_name(qkv, "tf0_qkv");
  }

  /* Split into Q, K, V: each [T, D] */
  struct ggml_tensor *q = ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], 0);
  struct ggml_tensor *k =
      ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], D * sizeof(float));
  struct ggml_tensor *v =
      ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], 2 * D * sizeof(float));

  /* Make the views contiguous before reshaping */
  q = ggml_cont(ctx, q);
  k = ggml_cont(ctx, k);
  v = ggml_cont(ctx, v);

  /* Reshape to [head_dim, n_heads, T] for multi-head attention */
  q = ggml_reshape_3d(ctx, q, head_dim, n_heads, T);
  k = ggml_reshape_3d(ctx, k, head_dim, n_heads, T);
  v = ggml_reshape_3d(ctx, v, head_dim, n_heads, T);
  if (layer_idx == 0) {
    DEBUG_PRINT_SHAPE("q_reshaped", q);
  }

  /* Apply RoPE to Q, K */
  q = ggml_rope(ctx, q, pos, rope_dim, 0);
  k = ggml_rope(ctx, k, pos, rope_dim, 0);
  if (layer_idx == 0) {
    ggml_set_name(q, "tf0_q_rope");
    ggml_set_name(k, "tf0_k_rope");
  }

  // k = ggml_cast(ctx, k, GGML_TYPE_F16);
  // v = ggml_cast(ctx, v, GGML_TYPE_F16);

  /* 【最终修复】: 将 Q, K, V 排列为一致 of [head_dim, T, n_heads] */
  q = ggml_permute(ctx, q, 0, 2, 1, 3);
  k = ggml_permute(ctx, k, 0, 2, 1, 3);
  v = ggml_permute(ctx, v, 0, 2, 1,
                   3); /* <-- 必须和 Q, K 严格保持相同的排列轴！ */
  if (layer_idx == 0) {
    DEBUG_PRINT_SHAPE("q_permuted", q);
  }

  /* 对于 Flash Attention，强烈建议保持 V 在内存中连续 */
  v = ggml_cont(ctx, v);
  if (layer_idx == 0) {
    DEBUG_PRINT_SHAPE("v_permuted", v);
  }

  /* Scaled dot-product attention (non-causal) */
  struct ggml_tensor *attn = ggml_flash_attn_ext(
      ctx, q, k, v, NULL, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
  if (layer_idx == 0) {
    DEBUG_PRINT_SHAPE("attn_out", attn);
    ggml_set_name(attn, "tf0_attn_out");
  }

  /* Reshape back to [T, D] */
  attn = ggml_reshape_2d(
      ctx, ggml_cont(ctx, attn), D, T);

  /* Output projection */
  attn = ggml_mul_mat(ctx, tb->c_proj_w, attn);
  if (layer_idx == 0) {
    ggml_set_name(attn, "tf0_attn_proj_out");
  }

  /* Residual */
  x = ggml_add(ctx, x, attn);
  if (layer_idx == 0) {
    ggml_set_name(x, "tf0_attn_residual");
  }

  /* Pre-norm FFN */
  struct ggml_tensor *xn2 = rms_norm(ctx, x, tb->ffn_norm_w);
  struct ggml_tensor *ff = ggml_mul_mat(ctx, tb->mlp_fc1_w, xn2);
  if (layer_idx == 0) {
    ggml_set_name(xn2, "tf0_ffn_norm");
    ggml_set_name(ff, "tf0_ffn_proj");
  }
  ff = ggml_silu(ctx, ff);
  ff = ggml_mul_mat(ctx, tb->mlp_fc2_w, ff);
  if (layer_idx == 0) {
    ggml_set_name(ff, "tf0_ffn_out");
  }

  return ggml_add(ctx, x, ff);
}
/* ── Build full decoder graph ────────────────────────────────── */
struct ggml_cgraph *xcodec2_build_graph(struct xcodec2_model *model,
                                        struct ggml_context *ctx,
                                        struct ggml_tensor *indices,
                                        int n_codes) {
  struct xcodec2_hparams *hp = &model->hparams;

  // printf("\n==== BUILDING GGML GRAPH ====\n");
  DEBUG_PRINT_SHAPE("input_indices", indices);

  /* 【修复5】: 创建 position 索引张量，标记为图的输入 */
  int T_seq = indices->ne[0];
  struct ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_seq);
  ggml_set_name(pos, "position_ids");
  // 在 ggml_graph_compute 之前
  int32_t *pos_data = (int32_t *)pos->data;
  if (pos_data) {
    for (int i = 0; i < T_seq; i++) {
      pos_data[i] = i;
    }
  }
  ggml_set_input(pos);

  /* 1. FSQ codebook lookup: indices[n_codes] -> embeddings[n_codes, vq_dim] */
  struct ggml_tensor *emb = ggml_get_rows(ctx, model->fsq_codebook, indices);
  ggml_set_name(emb, "emb");
  DEBUG_PRINT_SHAPE("emb (lookup)", emb);

  /* 2. fc_post_a: Linear(2048, 1024) */
  struct ggml_tensor *x = ggml_mul_mat(ctx, model->fc_post_a_w, emb);
  x = ggml_add(ctx, x, model->fc_post_a_b);
  ggml_set_name(x, "fc_post_a");
  DEBUG_PRINT_SHAPE("x (after fc_post_a)", x);

  /* 【修复1】: Transpose before Conv1d */
  x = ggml_cont(ctx, ggml_transpose(ctx, x));
  ggml_set_name(x, "transposed_4_conv");
  DEBUG_PRINT_SHAPE("x (transposed_4_conv)", x);

  /* 3. VocosBackbone */
  /* 3a. embed Conv1d */
  x = conv1d(ctx, x, model->embed_w, model->embed_b, 1, 3, 1);
  ggml_set_name(x, "embed_conv");
  DEBUG_PRINT_SHAPE("x (after embed_conv)", x);

  /* 3b. prior_net: 2x ResnetBlock */
  for (int i = 0; i < 2; i++) {
    x = resnet_block(ctx, x, &model->prior_net[i]);
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "prior_resnet_%d", i);
    ggml_set_name(x, name_buf);
  }
  DEBUG_PRINT_SHAPE("x (after prior_net)", x);

  /* 3c. Transpose for transformer: [T, C] -> [C, T] */
  x = ggml_cont(ctx, ggml_transpose(ctx, x));
  ggml_set_name(x, "transposed_4_attn");
  DEBUG_PRINT_SHAPE("x (transposed_4_attn)", x);

  /* 3d. transformers: 12x TransformerBlock */
  for (int i = 0; i < hp->n_layers; i++) {
    /* 将 pos 张量传入 transformer_block */
    x = transformer_block(ctx, x, pos, &model->transformer[i], hp->n_heads,
                          hp->head_dim, hp->rope_dim, i);
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "transformer_%d", i);
    ggml_set_name(x, name_buf);
  }
  DEBUG_PRINT_SHAPE("x (after transformers)", x);

  /* 3e. Transpose back for conv: [C, T] -> [T, C] */
  x = ggml_cont(ctx, ggml_transpose(ctx, x));
  ggml_set_name(x, "transposed_4_postnet");
  DEBUG_PRINT_SHAPE("x (transposed_4_postnet)", x);

  /* 3f. post_net: 2x ResnetBlock */
  for (int i = 0; i < 2; i++) {
    x = resnet_block(ctx, x, &model->post_net[i]);
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "post_resnet_%d", i);
    ggml_set_name(x, name_buf);
  }

  /* 3g. Transpose for final LayerNorm: need [C, T] */
  x = ggml_cont(ctx, ggml_transpose(ctx, x));
  ggml_set_name(x, "before_final_norm");
  DEBUG_PRINT_SHAPE("x (before final_norm)", x);

  /* 3h. final LayerNorm */
  x = ggml_norm(ctx, x, 1e-6f);
  x = ggml_mul(ctx, x, model->final_ln_w);
  x = ggml_add(ctx, x, model->final_ln_b);
  ggml_set_name(x, "final_ln");

  /* 3i. Upsampler (optional, for 44.1kHz variants) */
  if (model->n_upsamplers > 0) {
    /* x is currently [C, T] (ne[0]=C, ne[1]=T).
       GGML's conv_transpose_1d and conv1d expect [T, C] (ne[0]=T, ne[1]=C).
       So we must transpose x.
    */
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // Now x is [T, C]
    ggml_set_name(x, "upsampler_in");

    for (int i = 0; i < model->n_upsamplers; i++) {
      int k = model->upsampler[i].conv_w->ne[0]; // kernel size
      int u = 3; // stride, hardcoded for now or derived? Let's use 3 as per
                 // Anime-XCodec2 config
      int pad = (k - u) / 2;

      x = conv_transpose_1d(ctx, x, model->upsampler[i].conv_w,
                            model->upsampler[i].conv_b, u, pad);
      char name_buf[64];
      snprintf(name_buf, sizeof(name_buf), "upsampler_conv_%d", i);
      ggml_set_name(x, name_buf);

      x = resnet_block(ctx, x, &model->upsampler[i].resnet);
      snprintf(name_buf, sizeof(name_buf), "upsampler_resnet_%d", i);
      ggml_set_name(x, name_buf);
    }

    /* After the upsampler loop, x is [T', C'] (ne[0]=T', ne[1]=C').
       We need to apply upsampler_out_proj (Linear layer).
       upsampler_out_proj_w has shape [C', C_final] in GGML (ne[0]=C',
       ne[1]=C_final). ggml_mul_mat(A, B) requires A->ne[0] == B->ne[0]. So we
       MUST transpose x to [C', T'] (ne[0]=C', ne[1]=T').
    */
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // Now x is [C', T']
    x = ggml_mul_mat(ctx, model->upsampler_out_proj_w,
                     x); // Returns [C_final, T']

    /* upsampler_out_proj_b is [C_final], we broadcast add over T' */
    x = ggml_add(
        ctx, x, ggml_reshape_2d(ctx, model->upsampler_out_proj_b, x->ne[0], 1));
    ggml_set_name(x, "upsampler_out_proj");

    x = swish(ctx, x); // nonlinearity
    ggml_set_name(x, "upsampler_out_swish");
    // The output of swish is [C_final, T']. It feeds directly into ISTFTHead.
  }

  /* 4. ISTFTHead */
  struct ggml_tensor *pred = ggml_mul_mat(ctx, model->head_out_w, x);
  pred = ggml_add(ctx, pred, model->head_out_b);
  DEBUG_PRINT_SHAPE("pred (final stft out)", pred);
  // printf("=============================\n\n");

  /* Mark output */
  ggml_set_output(pred);
  ggml_set_name(pred, "stft_pred");

  struct ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, pred);
  return gf;
}