/*
 * quantize.cpp - Quantize xcodec2 GGUF model weights
 *
 * Usage: xcodec2_quantize input.gguf output.gguf type
 *   type: q8_0, q4_0, q4_1, q5_0, q5_1, f16
 */

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>

static const std::map<std::string, enum ggml_type> QUANT_TYPES = {
    {"f16",  GGML_TYPE_F16},
    {"q8_0", GGML_TYPE_Q8_0},
    {"q4_0", GGML_TYPE_Q4_0},
    {"q4_1", GGML_TYPE_Q4_1},
    {"q5_0", GGML_TYPE_Q5_0},
    {"q5_1", GGML_TYPE_Q5_1},
};

/* Tensors that should stay F32 (small or sensitive) */
static bool should_skip_quantize(const char * name) {
    std::string n(name);
    /* Keep biases, norms, window, and codebook as F32 */
    if (n.find(".bias")    != std::string::npos) return true;
    if (n.find("norm")     != std::string::npos) return true;
    if (n.find("window")   != std::string::npos) return true;
    // if (n.find("codebook") != std::string::npos) return true;
    /* Keep 1D tensors (biases etc.) */
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", argv[0]);
        fprintf(stderr, "  type: f16, q8_0, q4_0, q4_1, q5_0, q5_1\n");
        return 1;
    }

    const char * input_path  = argv[1];
    const char * output_path = argv[2];
    const char * type_str    = argv[3];

    auto it = QUANT_TYPES.find(type_str);
    if (it == QUANT_TYPES.end()) {
        fprintf(stderr, "unknown quantization type: %s\n", type_str);
        return 1;
    }
    enum ggml_type target_type = it->second;
    printf("Quantizing to %s\n", type_str);

    std::vector<void *> allocated_buffers;
    std::vector<struct ggml_context *> allocated_contexts;

    /* Load source GGUF */
    struct ggml_context * ctx_in = nullptr;
    struct gguf_init_params params = { .no_alloc = false, .ctx = &ctx_in };
    struct gguf_context * gguf_in = gguf_init_from_file(input_path, params);
    if (!gguf_in) { fprintf(stderr, "failed to load %s\n", input_path); return 1; }

    int n_tensors = gguf_get_n_tensors(gguf_in);
    int n_kv      = gguf_get_n_kv(gguf_in);
    printf("Input: %d tensors, %d KV pairs\n", n_tensors, n_kv);

    /* Create output GGUF */
    struct gguf_context * gguf_out = gguf_init_empty();

    /* Copy all KV metadata */
    for (int i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(gguf_in, i);
        enum gguf_type ktype = gguf_get_kv_type(gguf_in, i);
        switch (ktype) {
            case GGUF_TYPE_UINT32:
                gguf_set_val_u32(gguf_out, key, gguf_get_val_u32(gguf_in, i));
                break;
            case GGUF_TYPE_FLOAT32:
                gguf_set_val_f32(gguf_out, key, gguf_get_val_f32(gguf_in, i));
                break;
            case GGUF_TYPE_STRING:
                gguf_set_val_str(gguf_out, key, gguf_get_val_str(gguf_in, i));
                break;
            default:
                fprintf(stderr, "warning: skipping KV %s (type %d)\n", key, ktype);
                break;
        }
    }

    /* Process each tensor */
    int n_quantized = 0, n_kept = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_in, i);
        struct ggml_tensor * src = ggml_get_tensor(ctx_in, name);
        if (!src) { fprintf(stderr, "tensor not found: %s\n", name); continue; }

        int64_t block_size = ggml_blck_size(target_type);
        bool skip = should_skip_quantize(name) || ggml_n_dims(src) < 2 || (block_size > 0 && src->ne[0] % block_size != 0);

        if (skip || src->type != GGML_TYPE_F32) {
            /* Keep as-is */
            gguf_add_tensor(gguf_out, src);
            n_kept++;
            printf("  [keep] %s (%s)\n", name, ggml_type_name(src->type));
        } else {
            /* Quantize */
            int64_t nelements = ggml_nelements(src);
            float * src_data = (float *)src->data;

            /* Create a temporary tensor with target type */
            size_t new_size = nelements * ggml_type_size(target_type) / ggml_blck_size(target_type);
            void * new_data = malloc(new_size);
            allocated_buffers.push_back(new_data);

            /* Use ggml quantize functions */
            int64_t n_per_row = src->ne[0];
            int64_t nrows = nelements / n_per_row;
            ggml_quantize_chunk(target_type, src_data, new_data, 0, nrows, n_per_row, nullptr);

            /* Create output tensor with same shape but new type */
            struct ggml_init_params tparams = {
                .mem_size = ggml_tensor_overhead(),
                .mem_buffer = nullptr,
                .no_alloc = true,
            };
            struct ggml_context * tctx = ggml_init(tparams);
            allocated_contexts.push_back(tctx);

            struct ggml_tensor * dst = ggml_new_tensor(tctx, target_type, ggml_n_dims(src), src->ne);
            ggml_set_name(dst, name);
            dst->data = new_data;

            gguf_add_tensor(gguf_out, dst);
            n_quantized++;

            printf("  [quant] %s: %s -> %s (%.1f MB -> %.1f MB)\n",
                   name, ggml_type_name(src->type), ggml_type_name(target_type),
                   (float)(nelements * sizeof(float)) / 1048576.f,
                   (float)new_size / 1048576.f);
        }
    }

    /* Write output */
    printf("\nWriting %s ...\n", output_path);
    gguf_write_to_file(gguf_out, output_path, false);

    printf("Done! %d quantized, %d kept as-is\n", n_quantized, n_kept);

    /* Free temporary contexts and buffers */
    for (void * ptr : allocated_buffers) {
        free(ptr);
    }
    for (struct ggml_context * ctx : allocated_contexts) {
        ggml_free(ctx);
    }

    gguf_free(gguf_out);
    ggml_free(ctx_in);
    gguf_free(gguf_in);

    return 0;
}
