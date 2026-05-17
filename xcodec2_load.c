/* xcodec2_load.c - GGUF model loading */
#include "xcodec2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); }
    return t;
}

static void load_resnet(struct xcodec2_resnet_block * blk, struct ggml_context * ctx,
                        const char * prefix, int idx) {
    char buf[128];
#define L(field, sfx) \
    snprintf(buf, sizeof(buf), "%s.%d." sfx, prefix, idx); \
    blk->field = get_tensor(ctx, buf);
    L(norm1_w, "norm1.weight") L(norm1_b, "norm1.bias")
    L(conv1_w, "conv1.weight") L(conv1_b, "conv1.bias")
    L(norm2_w, "norm2.weight") L(norm2_b, "norm2.bias")
    L(conv2_w, "conv2.weight") L(conv2_b, "conv2.bias")
#undef L
}

int xcodec2_load(struct xcodec2_model * model, const char * path) {
    struct gguf_init_params params = { .no_alloc = false, .ctx = &model->ctx_data };
    model->ctx_gguf = gguf_init_from_file(path, params);
    if (!model->ctx_gguf) { fprintf(stderr, "failed to load %s\n", path); return -1; }

    struct ggml_context * ctx = model->ctx_data;

    /* hparams from GGUF metadata */
    struct xcodec2_hparams * hp = &model->hparams;
    int ki;
#define LOAD_U32(field, key) \
    ki = gguf_find_key(model->ctx_gguf, key); \
    hp->field = (ki >= 0) ? gguf_get_val_u32(model->ctx_gguf, ki) : 0;
    LOAD_U32(hidden_dim,    "xcodec2.hidden_dim")
    LOAD_U32(n_layers,      "xcodec2.n_layers")
    LOAD_U32(n_heads,       "xcodec2.n_heads")
    LOAD_U32(head_dim,      "xcodec2.head_dim")
    LOAD_U32(rope_dim,      "xcodec2.rope_dim")
    LOAD_U32(hop_length,    "xcodec2.hop_length")
    LOAD_U32(sample_rate,   "xcodec2.sample_rate")
    LOAD_U32(n_fft,         "xcodec2.n_fft")
    LOAD_U32(vq_dim,        "xcodec2.vq_dim")
    LOAD_U32(codebook_size, "xcodec2.codebook_size")
    LOAD_U32(codebook_dim,  "xcodec2.codebook_dim")
#undef LOAD_U32

    printf("xcodec2: hidden=%d layers=%d heads=%d hop=%d sr=%d nfft=%d vqdim=%d cb=%d\n",
           hp->hidden_dim, hp->n_layers, hp->n_heads, hp->hop_length,
           hp->sample_rate, hp->n_fft, hp->vq_dim, hp->codebook_size);

    /* load tensor pointers */
    model->fsq_codebook = get_tensor(ctx, "fsq.codebook");
    model->fc_post_a_w  = get_tensor(ctx, "fc_post_a.weight");
    model->fc_post_a_b  = get_tensor(ctx, "fc_post_a.bias");
    model->embed_w      = get_tensor(ctx, "backbone.embed.weight");
    model->embed_b      = get_tensor(ctx, "backbone.embed.bias");

    for (int i = 0; i < 2; i++) load_resnet(&model->prior_net[i], ctx, "backbone.prior_net", i);

    char buf[128];
    for (int i = 0; i < hp->n_layers; i++) {
        struct xcodec2_transformer_block * tb = &model->transformer[i];
#define T(field, sfx) \
        snprintf(buf, sizeof(buf), "backbone.transformer.%d." sfx, i); \
        tb->field = get_tensor(ctx, buf);
        T(att_norm_w,  "att_norm.weight")
        T(ffn_norm_w,  "ffn_norm.weight")
        T(c_attn_w,    "att.c_attn.weight")
        T(c_proj_w,    "att.c_proj.weight")
        T(mlp_fc1_w,   "mlp.fc1.weight")
        T(mlp_fc2_w,   "mlp.fc2.weight")
#undef T
    }

    model->final_ln_w = get_tensor(ctx, "backbone.final_layer_norm.weight");
    model->final_ln_b = get_tensor(ctx, "backbone.final_layer_norm.bias");

    for (int i = 0; i < 2; i++) load_resnet(&model->post_net[i], ctx, "backbone.post_net", i);

    model->head_out_w   = get_tensor(ctx, "head.out.weight");
    model->head_out_b   = get_tensor(ctx, "head.out.bias");
    model->istft_window = get_tensor(ctx, "head.istft.window");

    printf("xcodec2: loaded %d tensors\n", gguf_get_n_tensors(model->ctx_gguf));
    return 0;
}

void xcodec2_free(struct xcodec2_model * model) {
    if (model->ctx_data) { ggml_free(model->ctx_data); model->ctx_data = NULL; }
    if (model->ctx_gguf) { gguf_free(model->ctx_gguf); model->ctx_gguf = NULL; }
}
