#!/usr/bin/env python3
"""
Export xcodec2 decoder weights to GGUF format.

Usage:
    pip install gguf safetensors
    # Standard xcodec2 (16kHz):
    python export_gguf.py --model HKUST-Audio/xcodec2 --output xcodec2-decoder.gguf
    # Anime variant (44.1kHz):
    python export_gguf.py --model NandemoGHS/Anime-XCodec2-44.1kHz-v2 --output xcodec2-anime.gguf
"""

import argparse, sys, os
import numpy as np
import torch
from gguf import GGUFWriter


def build_fsq_codebook(levels, proj_out_w, proj_out_b=None):
    """Enumerate all FSQ indices, map to [-1,1], apply project_out."""
    cdim = len(levels)
    total = 1
    for l in levels: total *= l
    codes = np.zeros((total, cdim), dtype=np.float32)
    for idx in range(total):
        r = idx
        for d in range(cdim - 1, -1, -1):
            codes[idx, d] = r % levels[d]; r //= levels[d]
    for d in range(cdim):
        L = levels[d]
        codes[:, d] = 2.0 * codes[:, d] / (L - 1) - 1.0 if L > 1 else 0.0
    projected = codes @ proj_out_w.T
    if proj_out_b is not None: projected += proj_out_b
    return projected


def load_model(model_path):
    """Load xcodec2 model, handling safetensors .beta -> .bias rename."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from xcodec2.configuration_bigcodec import BigCodecConfig
    from xcodec2.modeling_xcodec2 import XCodec2Model

    # Try safetensors first (for models like Anime-XCodec2)
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        ckpt_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        ckpt = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)
        config = BigCodecConfig.from_pretrained(model_path)
        model = XCodec2Model.from_pretrained(None, config=config, state_dict=ckpt)
    except Exception:
        model = XCodec2Model.from_pretrained(model_path)
    model.eval()
    return model


def export(model_path, output_path, use_f16=False):
    model = load_model(model_path)
    sd = model.state_dict()

    # Detect sample rate from config
    sr = getattr(model.config, 'sample_rate', 16000)
    hop = getattr(model.config, 'hop_length', 320)
    ups = getattr(model.config, 'upsample_factors', []) or []
    ups_total = 1
    for u in ups: ups_total *= u
    n_fft = hop * 4

    print(f"Model: sr={sr}, hop={hop}, upsample={ups}, n_fft={n_fft}")

    writer = GGUFWriter(output_path, arch="xcodec2")

    # Metadata
    writer.add_uint32("xcodec2.hidden_dim",    1024)
    writer.add_uint32("xcodec2.n_layers",      12)
    writer.add_uint32("xcodec2.n_heads",       16)
    writer.add_uint32("xcodec2.head_dim",      64)
    writer.add_uint32("xcodec2.rope_dim",      64)
    writer.add_uint32("xcodec2.hop_length",    hop)
    writer.add_uint32("xcodec2.sample_rate",   sr)
    writer.add_uint32("xcodec2.n_fft",         n_fft)
    writer.add_uint32("xcodec2.vq_dim",        2048)
    writer.add_uint32("xcodec2.codebook_dim",  8)
    writer.add_string("xcodec2.model_id",      model_path)

    # Detect upsample factors
    if ups:
        writer.add_uint32("xcodec2.n_upsample", len(ups))
        for i, u in enumerate(ups):
            writer.add_uint32(f"xcodec2.upsample_factor.{i}", u)

    n_t = [0]
    def add(name, data):
        d = data.astype(np.float16 if (use_f16 and data.ndim >= 2) else np.float32)
        writer.add_tensor(name, d)
        n_t[0] += 1

    # 1. FSQ Codebook
    q = model.generator.quantizer
    fsq = q.fsqs[0]
    levels = [int(l) for l in fsq.levels.tolist()]
    pw = q.project_out.weight.detach().cpu().numpy()
    pb = q.project_out.bias.detach().cpu().numpy() if q.project_out.bias is not None else None
    cb = build_fsq_codebook(levels, pw, pb)
    writer.add_uint32("xcodec2.codebook_size", cb.shape[0])
    add("fsq.codebook", cb)
    print(f"FSQ codebook: {cb.shape}, levels={levels}")

    # 2. fc_post_a
    add("fc_post_a.weight", sd['fc_post_a.weight'].cpu().numpy())
    add("fc_post_a.bias",   sd['fc_post_a.bias'].cpu().numpy())

    # 3. Backbone
    p = "generator.backbone."
    add("backbone.embed.weight", sd[p+'embed.weight'].cpu().numpy())
    add("backbone.embed.bias",   sd[p+'embed.bias'].cpu().numpy())

    for net in ["prior_net", "post_net"]:
        for i in range(2):
            r = f"{p}{net}.{i}."
            for s in ["norm1.weight","norm1.bias","conv1.weight","conv1.bias",
                       "norm2.weight","norm2.bias","conv2.weight","conv2.bias"]:
                add(f"backbone.{net}.{i}.{s}", sd[r+s].cpu().numpy())

    for i in range(12):
        t = f"{p}transformers.{i}."
        add(f"backbone.transformer.{i}.att_norm.weight",  sd[t+'att_norm.weight'].cpu().numpy())
        add(f"backbone.transformer.{i}.ffn_norm.weight",  sd[t+'ffn_norm.weight'].cpu().numpy())
        add(f"backbone.transformer.{i}.att.c_attn.weight", sd[t+'att.c_attn.weight'].cpu().numpy())
        add(f"backbone.transformer.{i}.att.c_proj.weight", sd[t+'att.c_proj.weight'].cpu().numpy())
        add(f"backbone.transformer.{i}.mlp.fc1.weight",   sd[t+'mlp.fc1.weight'].cpu().numpy())
        add(f"backbone.transformer.{i}.mlp.fc2.weight",   sd[t+'mlp.fc2.weight'].cpu().numpy())

    add("backbone.final_layer_norm.weight", sd[p+'final_layer_norm.weight'].cpu().numpy())
    add("backbone.final_layer_norm.bias",   sd[p+'final_layer_norm.bias'].cpu().numpy())

    # 4. Upsampler (if present)
    gen = model.generator
    if gen.upsampler is not None:
        print(f"Upsampler: {len(ups)} stages")
        for i in range(len(ups)):
            up_prefix = f"generator.upsampler.upsample_layers.{i}."
            # ConvTranspose1d (weight-normed)
            w_key = up_prefix + "weight"
            if w_key in sd:
                add(f"upsampler.{i}.conv.weight", sd[w_key].cpu().numpy())
            # Try parametrized weight_norm
            wg_key = up_prefix + "parametrizations.weight.original0"
            wv_key = up_prefix + "parametrizations.weight.original1"
            if wg_key in sd and wv_key in sd:
                g = sd[wg_key].cpu()
                v = sd[wv_key].cpu()
                norm = torch.norm(v.reshape(v.shape[0], -1), dim=1)
                scale = (g.reshape(-1) / norm).reshape([-1]+[1]*(v.dim()-1))
                add(f"upsampler.{i}.conv.weight", (v * scale).numpy())
            if up_prefix + "bias" in sd:
                add(f"upsampler.{i}.conv.bias", sd[up_prefix+"bias"].cpu().numpy())

            # ResnetBlock
            rb_prefix = f"generator.upsampler.resnet_blocks.{i}."
            for s in ["norm1.weight","norm1.bias","conv1.weight","conv1.bias",
                       "norm2.weight","norm2.bias","conv2.weight","conv2.bias"]:
                k = rb_prefix + s
                if k in sd:
                    add(f"upsampler.{i}.resnet.{s}", sd[k].cpu().numpy())

        # out_proj
        op_key = "generator.upsampler.out_proj.weight"
        if op_key in sd:
            add("upsampler.out_proj.weight", sd[op_key].cpu().numpy())
        op_key = "generator.upsampler.out_proj.bias"
        if op_key in sd:
            add("upsampler.out_proj.bias", sd[op_key].cpu().numpy())

    # 5. ISTFTHead
    hp = "generator.head."
    add("head.out.weight", sd[hp+'out.weight'].cpu().numpy())
    add("head.out.bias",   sd[hp+'out.bias'].cpu().numpy())
    add("head.istft.window", sd[hp+'istft.window'].cpu().numpy())

    # Write
    print(f"\nWriting {n_t[0]} tensors to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done! {os.path.getsize(output_path)/1048576:.1f} MB")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='HKUST-Audio/xcodec2')
    ap.add_argument('--output', default='xcodec2-decoder.gguf')
    ap.add_argument('--f16', action='store_true')
    export(ap.parse_args().model, ap.parse_args().output, ap.parse_args().f16)
