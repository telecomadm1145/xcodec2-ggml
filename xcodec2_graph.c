/* xcodec2_graph.c - build ggml compute graph for decoder inference */
#include "xcodec2.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── helpers ─────────────────────────────────────────────────── */

/* SiLU / swish: x * sigmoid(x) */
static struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_mul(ctx, x, ggml_sigmoid(ctx, x));
}

/* swish nonlinearity used by ResnetBlock */
static struct ggml_tensor * swish(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_silu(ctx, x);
}

/* Conv1d: weight [Cout, Cin, K], input [Cin, T] -> [Cout, T'] */
static struct ggml_tensor * conv1d(struct ggml_context * ctx,
                                   struct ggml_tensor * x,
                                   struct ggml_tensor * w,
                                   struct ggml_tensor * b,
                                   int stride, int pad, int dilation) {
    struct ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    if (b) {
        /* b [Cout] -> broadcast add over T dimension */
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

/* GroupNorm(32 groups) + affine */
static struct ggml_tensor * group_norm(struct ggml_context * ctx,
                                       struct ggml_tensor * x,
                                       struct ggml_tensor * w,
                                       struct ggml_tensor * b) {
    /* ggml_group_norm expects [T, C, 1] but we have [T, C] in ggml layout.
       ggml dim0 = T (innermost), dim1 = C for a 2D tensor representing [C, T] in PyTorch.
       Actually in ggml, tensor ne[0] corresponds to the last dim in row-major.
       For Conv1d output: ne[0] = T, ne[1] = C. */
    x = ggml_group_norm(ctx, x, 32, 1e-5f);
    /* affine: x * w + b, w and b are [C] */
    x = ggml_mul(ctx, x, ggml_reshape_2d(ctx, w, 1, w->ne[0]));
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return x;
}

/* ── ResnetBlock ─────────────────────────────────────────────── */
static struct ggml_tensor * resnet_block(struct ggml_context * ctx,
                                         struct ggml_tensor * x,
                                         struct xcodec2_resnet_block * blk) {
    struct ggml_tensor * h = x;
    h = group_norm(ctx, h, blk->norm1_w, blk->norm1_b);
    h = swish(ctx, h);
    h = conv1d(ctx, h, blk->conv1_w, blk->conv1_b, 1, 1, 1);
    h = group_norm(ctx, h, blk->norm2_w, blk->norm2_b);
    h = swish(ctx, h);
    h = conv1d(ctx, h, blk->conv2_w, blk->conv2_b, 1, 1, 1);
    return ggml_add(ctx, x, h);
}

/* ── RMSNorm ─────────────────────────────────────────────────── */
static struct ggml_tensor * rms_norm(struct ggml_context * ctx,
                                     struct ggml_tensor * x,
                                     struct ggml_tensor * w) {
    x = ggml_rms_norm(ctx, x, 1e-6f);
    return ggml_mul(ctx, x, w);
}

/* ── RoPE (applied to q, k in attention) ─────────────────────── */
static struct ggml_tensor * apply_rope(struct ggml_context * ctx,
                                       struct ggml_tensor * x,
                                       int rope_dim, int n_past) {
    /* x: [head_dim, n_heads, T, 1]  (ggml layout)
       ggml_rope applies rotary embeddings on the first rope_dim elements */
    return ggml_rope(ctx, x, NULL, rope_dim, 0);
}

/* ── TransformerBlock ────────────────────────────────────────── */
static struct ggml_tensor * transformer_block(struct ggml_context * ctx,
                                              struct ggml_tensor * x,
                                              struct xcodec2_transformer_block * tb,
                                              int n_heads, int head_dim,
                                              int rope_dim) {
    int T = x->ne[0];
    int D = x->ne[1]; /* hidden_dim */

    /* Pre-norm attention */
    struct ggml_tensor * xn = rms_norm(ctx, x, tb->att_norm_w);

    /* QKV projection: [T, D] x [D, 3D]^T -> [T, 3D] */
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, tb->c_attn_w, xn);

    /* Split into Q, K, V: each [T, D] */
    struct ggml_tensor * q = ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], 0);
    struct ggml_tensor * k = ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], D * sizeof(float));
    struct ggml_tensor * v = ggml_view_2d(ctx, qkv, D, T, qkv->nb[1], 2 * D * sizeof(float));

    /* Reshape to [head_dim, n_heads, T] for multi-head attention */
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, T);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, T);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, T);

    /* Apply RoPE to Q, K */
    q = ggml_rope(ctx, q, NULL, rope_dim, 0);
    k = ggml_rope(ctx, k, NULL, rope_dim, 0);

    /* Permute for attention: [head_dim, T, n_heads] */
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    /* Scaled dot-product attention (non-causal) */
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);

    /* Reshape back to [T, D] */
    attn = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)), D, T);

    /* Output projection */
    attn = ggml_mul_mat(ctx, tb->c_proj_w, attn);

    /* Residual */
    x = ggml_add(ctx, x, attn);

    /* Pre-norm FFN */
    struct ggml_tensor * xn2 = rms_norm(ctx, x, tb->ffn_norm_w);
    struct ggml_tensor * ff = ggml_mul_mat(ctx, tb->mlp_fc1_w, xn2);
    ff = ggml_silu(ctx, ff);
    ff = ggml_mul_mat(ctx, tb->mlp_fc2_w, ff);

    return ggml_add(ctx, x, ff);
}

/* ── Build full decoder graph ────────────────────────────────── */
struct ggml_cgraph * xcodec2_build_graph(struct xcodec2_model * model,
                                          struct ggml_context * ctx,
                                          struct ggml_tensor * indices,
                                          int n_codes) {
    struct xcodec2_hparams * hp = &model->hparams;

    /* 1. FSQ codebook lookup: indices[n_codes] -> embeddings[n_codes, vq_dim]
       We use ggml_get_rows to index into the codebook */
    struct ggml_tensor * emb = ggml_get_rows(ctx, model->fsq_codebook, indices);
    /* emb: [vq_dim, n_codes] in ggml layout */

    /* 2. fc_post_a: Linear(2048, 1024), applied per-frame
       emb is [vq_dim, n_codes], weight is [hidden_dim, vq_dim]
       result: [hidden_dim, n_codes] */
    struct ggml_tensor * x = ggml_mul_mat(ctx, model->fc_post_a_w, emb);
    x = ggml_add(ctx, x, model->fc_post_a_b);

    /* 3. VocosBackbone
       Input to backbone is x: [hidden_dim, n_codes] = [1024, T] in ggml (= [T, 1024] logical)
       The backbone expects [B, T, D] -> transpose to [T, D] for ggml matmul convention.
       But conv1d expects [C, T] in ggml (ne[0]=T, ne[1]=C). x is already [hidden_dim, n_codes] -> good for conv. */

    /* 3a. embed Conv1d(1024, 1024, k=7, p=3) */
    /* Need to transpose for conv: ggml conv1d expects input [T, C] with weight [K, Cin, Cout]
       Actually in ggml: ggml_conv_1d(kernel, src, ...) where
       kernel: [K, Cin, Cout] and src: [T, Cin] -> output [T', Cout]
       Let's keep x as [vq_dim=T_content, n_codes_batch] and handle shapes carefully.

       In ggml, for 1D convolution:
       - a (kernel): ne[0]=K, ne[1]=Cin, ne[2]=Cout
       - b (input):  ne[0]=T, ne[1]=Cin
       Our embed_w from PyTorch Conv1d(1024,1024,7) is stored as [Cout, Cin, K] = [1024, 1024, 7]
       In GGUF it will be [7, 1024, 1024] in ggml layout (reversed). This matches ggml_conv_1d expectation.
    */
    x = conv1d(ctx, x, model->embed_w, model->embed_b, 1, 3, 1);

    /* 3b. prior_net: 2x ResnetBlock */
    for (int i = 0; i < 2; i++)
        x = resnet_block(ctx, x, &model->prior_net[i]);

    /* 3c. Transpose for transformer: [T, C] in ggml -> transformers work on [T, D] */
    /* x is [T, C] in ggml = ne[0]=T, ne[1]=C. For matmul in transformer we need [D, T]?
       Actually ggml_mul_mat(A, B) = A^T @ B. With A=[D_out, D_in] and B=[D_in, T],
       output is [D_out, T]. Our x after conv is ne[0]=T, ne[1]=1024.
       We need to transpose so x is [1024, T] for the linear layers. */
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    /* 3d. transformers: 12x TransformerBlock */
    for (int i = 0; i < hp->n_layers; i++)
        x = transformer_block(ctx, x, &model->transformer[i],
                              hp->n_heads, hp->head_dim, hp->rope_dim);

    /* 3e. Transpose back for conv: [D, T] -> [T, D] then post_net */
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    /* 3f. post_net: 2x ResnetBlock */
    for (int i = 0; i < 2; i++)
        x = resnet_block(ctx, x, &model->post_net[i]);

    /* 3g. Transpose for final LayerNorm: need [T, D] */
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    /* 3h. final LayerNorm */
    x = ggml_norm(ctx, x, 1e-6f);
    x = ggml_mul(ctx, x, model->final_ln_w);
    x = ggml_add(ctx, x, model->final_ln_b);

    /* 4. ISTFTHead: Linear(1024, n_fft+2=1282) */
    struct ggml_tensor * pred = ggml_mul_mat(ctx, model->head_out_w, x);
    pred = ggml_add(ctx, pred, model->head_out_b);
    /* pred: [1282, T] in ggml = [T, 1282] logical */

    /* Mark output - the ISTFT is done in CPU post-processing */
    ggml_set_output(pred);
    ggml_set_name(pred, "stft_pred");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, pred);
    return gf;
}
