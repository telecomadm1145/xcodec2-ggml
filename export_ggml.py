#!/usr/bin/env python3
"""
Export xcodec2 decoder weights to GGML binary format.

Usage:
    python export_ggml.py --model HKUST-Audio/xcodec2 --output xcodec2-decoder.bin

Binary format:
    Header:
        magic:   uint32  (0x78636432 = "xcd2")
        version: uint32  (1)
        n_tensors: uint32

    For each tensor:
        name_len:  uint32
        name:      char[name_len]
        n_dims:    uint32
        dims:      uint32[n_dims]
        dtype:     uint32  (0=f32, 1=f16)
        data:      float[product(dims)] or float16[product(dims)]
"""

import argparse
import struct
import sys
import os
import numpy as np
import torch

def write_tensor(f, name: str, data: np.ndarray, use_f16: bool = False):
    """Write a single tensor in our binary format."""
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)

    shape = data.shape
    f.write(struct.pack('<I', len(shape)))
    for d in shape:
        f.write(struct.pack('<I', d))

    if use_f16:
        f.write(struct.pack('<I', 1))  # dtype = f16
        data_f16 = data.astype(np.float16)
        f.write(data_f16.tobytes())
    else:
        f.write(struct.pack('<I', 0))  # dtype = f32
        data_f32 = data.astype(np.float32)
        f.write(data_f32.tobytes())


def fold_weight_norm(weight_g, weight_v):
    """Fold weight_norm parametrization: w = g * v / ||v||"""
    # weight_v: [out, in, k] or [out, in]
    # weight_g: [out, 1, 1] or [out, 1] or [out]
    norm = torch.norm(weight_v.reshape(weight_v.shape[0], -1), dim=1)
    g = weight_g.reshape(-1)
    scale = g / norm
    # Reshape scale for broadcasting
    shape = [weight_v.shape[0]] + [1] * (weight_v.dim() - 1)
    return (weight_v * scale.reshape(shape)).numpy()


def build_fsq_codebook(levels, dim, project_out_weight, project_out_bias=None):
    """
    Build the full FSQ codebook by enumerating all index combinations.

    For FSQ with levels=[4,4,4,4,4,4,4,4], codebook_dim=8, total entries=4^8=65536.
    Each index maps to a code in [-1, 1] range, then projected through project_out.
    """
    codebook_dim = len(levels)
    # Total codebook size
    total = 1
    for l in levels:
        total *= l

    # Build basis for index decomposition
    # indices_to_codes: index -> code vector in codebook_dim
    codes = np.zeros((total, codebook_dim), dtype=np.float32)
    for idx in range(total):
        remaining = idx
        for d in range(codebook_dim - 1, -1, -1):
            codes[idx, d] = remaining % levels[d]
            remaining //= levels[d]

    # Map to [-1, 1] range: code = 2 * code / (level - 1) - 1
    for d in range(codebook_dim):
        L = levels[d]
        if L > 1:
            codes[:, d] = 2.0 * codes[:, d] / (L - 1) - 1.0
        else:
            codes[:, d] = 0.0

    # Apply project_out: codes @ W^T + bias
    # project_out_weight shape: [out_dim, codebook_dim]
    W = project_out_weight  # numpy array
    projected = codes @ W.T
    if project_out_bias is not None:
        projected += project_out_bias

    return projected  # [total, out_dim]


def export_model(model_path: str, output_path: str, use_f16: bool = False):
    print(f"Loading model from {model_path}...")

    # We need to load the model to get its state dict
    # Add parent directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from xcodec2.modeling_xcodec2 import XCodec2Model
    model = XCodec2Model.from_pretrained(model_path)
    model.eval()

    sd = model.state_dict()

    # Print some info
    print(f"State dict keys: {len(sd)}")
    for k in sorted(sd.keys()):
        print(f"  {k}: {sd[k].shape} {sd[k].dtype}")

    tensors = []  # list of (name, numpy_array)

    # ============================================================
    # 1. FSQ Codebook
    # ============================================================
    # The quantizer is ResidualFSQ with levels=[4,4,4,4,4,4,4,4], num_quantizers=1
    # We need project_in and project_out weights
    # ResidualFSQ has: fsqs (ModuleList of FSQ modules), project_in, project_out

    # Find FSQ-related weights
    print("\n--- Building FSQ codebook ---")

    # The ResidualFSQ in vector_quantize_pytorch structure:
    # generator.quantizer.project_in.weight, .bias
    # generator.quantizer.project_out.weight, .bias
    # generator.quantizer.fsqs.0._codebook (the implicit codebook)

    # First, let's find the project_out weights
    proj_out_weight = None
    proj_out_bias = None
    proj_in_weight = None
    proj_in_bias = None

    for k, v in sd.items():
        if 'quantizer' in k:
            print(f"  Quantizer param: {k} -> {v.shape}")

    # Access model directly to get FSQ info
    quantizer = model.generator.quantizer
    print(f"  Quantizer type: {type(quantizer)}")
    print(f"  Quantizer levels: {quantizer.fsqs[0].levels if hasattr(quantizer, 'fsqs') else 'N/A'}")

    if hasattr(quantizer, 'project_out'):
        proj_out = quantizer.project_out
        if hasattr(proj_out, 'weight'):
            proj_out_weight = proj_out.weight.detach().numpy()
            proj_out_bias = proj_out.bias.detach().numpy() if proj_out.bias is not None else None
            print(f"  project_out weight: {proj_out_weight.shape}")

    if hasattr(quantizer, 'project_in'):
        proj_in = quantizer.project_in
        if hasattr(proj_in, 'weight'):
            proj_in_weight = proj_in.weight.detach().numpy()
            print(f"  project_in weight: {proj_in_weight.shape}")

    # Get levels
    fsq = quantizer.fsqs[0]
    levels_tensor = fsq.levels
    levels = levels_tensor.tolist()
    levels = [int(l) for l in levels]
    print(f"  FSQ levels: {levels}")

    # Build full codebook (already projected)
    codebook = build_fsq_codebook(levels, len(levels), proj_out_weight, proj_out_bias)
    print(f"  Built codebook: {codebook.shape}")  # [65536, 2048]
    tensors.append(("fsq.codebook", codebook))

    # ============================================================
    # 2. fc_post_a (Linear 2048 -> 1024)
    # ============================================================
    fc_post_a_w = sd['fc_post_a.weight'].detach().numpy()
    fc_post_a_b = sd['fc_post_a.bias'].detach().numpy()
    tensors.append(("fc_post_a.weight", fc_post_a_w))
    tensors.append(("fc_post_a.bias", fc_post_a_b))
    print(f"\nfc_post_a weight: {fc_post_a_w.shape}, bias: {fc_post_a_b.shape}")

    # ============================================================
    # 3. VocosBackbone
    # ============================================================
    prefix = "generator.backbone."

    # 3a. embed conv1d (kernel=7, padding=3)
    embed_w = sd[prefix + 'embed.weight'].detach().numpy()
    embed_b = sd[prefix + 'embed.bias'].detach().numpy()
    tensors.append(("backbone.embed.weight", embed_w))
    tensors.append(("backbone.embed.bias", embed_b))
    print(f"\nBackbone embed: weight={embed_w.shape}, bias={embed_b.shape}")

    # 3b. prior_net (2x ResnetBlock)
    for i in range(2):
        rn_prefix = f"{prefix}prior_net.{i}."
        # norm1 (GroupNorm num_groups=32)
        tensors.append((f"backbone.prior_net.{i}.norm1.weight",
                        sd[rn_prefix + 'norm1.weight'].detach().numpy()))
        tensors.append((f"backbone.prior_net.{i}.norm1.bias",
                        sd[rn_prefix + 'norm1.bias'].detach().numpy()))
        # conv1 (Conv1d k=3 p=1)
        tensors.append((f"backbone.prior_net.{i}.conv1.weight",
                        sd[rn_prefix + 'conv1.weight'].detach().numpy()))
        tensors.append((f"backbone.prior_net.{i}.conv1.bias",
                        sd[rn_prefix + 'conv1.bias'].detach().numpy()))
        # norm2
        tensors.append((f"backbone.prior_net.{i}.norm2.weight",
                        sd[rn_prefix + 'norm2.weight'].detach().numpy()))
        tensors.append((f"backbone.prior_net.{i}.norm2.bias",
                        sd[rn_prefix + 'norm2.bias'].detach().numpy()))
        # conv2 (Conv1d k=3 p=1)
        tensors.append((f"backbone.prior_net.{i}.conv2.weight",
                        sd[rn_prefix + 'conv2.weight'].detach().numpy()))
        tensors.append((f"backbone.prior_net.{i}.conv2.bias",
                        sd[rn_prefix + 'conv2.bias'].detach().numpy()))
        print(f"  prior_net.{i}: loaded")

    # 3c. transformers (12x TransformerBlock)
    for i in range(12):
        tb_prefix = f"{prefix}transformers.{i}."

        # att_norm (RMSNorm)
        tensors.append((f"backbone.transformer.{i}.att_norm.weight",
                        sd[tb_prefix + 'att_norm.weight'].detach().numpy()))

        # ffn_norm (RMSNorm)
        tensors.append((f"backbone.transformer.{i}.ffn_norm.weight",
                        sd[tb_prefix + 'ffn_norm.weight'].detach().numpy()))

        # att.c_attn (Linear dim -> 3*dim, no bias)
        tensors.append((f"backbone.transformer.{i}.att.c_attn.weight",
                        sd[tb_prefix + 'att.c_attn.weight'].detach().numpy()))

        # att.c_proj (Linear dim -> dim, no bias)
        tensors.append((f"backbone.transformer.{i}.att.c_proj.weight",
                        sd[tb_prefix + 'att.c_proj.weight'].detach().numpy()))

        # mlp.fc1 (Linear dim -> 4*dim, no bias)
        tensors.append((f"backbone.transformer.{i}.mlp.fc1.weight",
                        sd[tb_prefix + 'mlp.fc1.weight'].detach().numpy()))

        # mlp.fc2 (Linear 4*dim -> dim, no bias)
        tensors.append((f"backbone.transformer.{i}.mlp.fc2.weight",
                        sd[tb_prefix + 'mlp.fc2.weight'].detach().numpy()))

        print(f"  transformer.{i}: loaded")

    # 3d. final_layer_norm
    tensors.append(("backbone.final_layer_norm.weight",
                    sd[prefix + 'final_layer_norm.weight'].detach().numpy()))
    tensors.append(("backbone.final_layer_norm.bias",
                    sd[prefix + 'final_layer_norm.bias'].detach().numpy()))

    # 3e. post_net (2x ResnetBlock)
    for i in range(2):
        rn_prefix = f"{prefix}post_net.{i}."
        tensors.append((f"backbone.post_net.{i}.norm1.weight",
                        sd[rn_prefix + 'norm1.weight'].detach().numpy()))
        tensors.append((f"backbone.post_net.{i}.norm1.bias",
                        sd[rn_prefix + 'norm1.bias'].detach().numpy()))
        tensors.append((f"backbone.post_net.{i}.conv1.weight",
                        sd[rn_prefix + 'conv1.weight'].detach().numpy()))
        tensors.append((f"backbone.post_net.{i}.conv1.bias",
                        sd[rn_prefix + 'conv1.bias'].detach().numpy()))
        tensors.append((f"backbone.post_net.{i}.norm2.weight",
                        sd[rn_prefix + 'norm2.weight'].detach().numpy()))
        tensors.append((f"backbone.post_net.{i}.norm2.bias",
                        sd[rn_prefix + 'norm2.bias'].detach().numpy()))
        tensors.append((f"backbone.post_net.{i}.conv2.weight",
                        sd[rn_prefix + 'conv2.weight'].detach().numpy()))
        tensors.append((f"backbone.post_net.{i}.conv2.bias",
                        sd[rn_prefix + 'conv2.bias'].detach().numpy()))
        print(f"  post_net.{i}: loaded")

    # ============================================================
    # 4. ISTFTHead
    # ============================================================
    head_prefix = "generator.head."
    tensors.append(("head.out.weight",
                    sd[head_prefix + 'out.weight'].detach().numpy()))
    tensors.append(("head.out.bias",
                    sd[head_prefix + 'out.bias'].detach().numpy()))
    print(f"\nISTFT head: out weight={sd[head_prefix + 'out.weight'].shape}")

    # ISTFT window (hann window)
    istft_window = sd[head_prefix + 'istft.window'].detach().numpy()
    tensors.append(("head.istft.window", istft_window))
    print(f"  ISTFT window: {istft_window.shape}")

    # ============================================================
    # Write binary file
    # ============================================================
    print(f"\nWriting {len(tensors)} tensors to {output_path}...")

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x78636432))  # magic "xcd2"
        f.write(struct.pack('<I', 1))            # version
        f.write(struct.pack('<I', len(tensors)))  # n_tensors

        # Hyperparameters
        f.write(struct.pack('<I', 1024))   # hidden_dim
        f.write(struct.pack('<I', 12))     # n_transformer_layers
        f.write(struct.pack('<I', 16))     # n_heads
        f.write(struct.pack('<I', 64))     # rope_dim (pos_meb_dim)
        f.write(struct.pack('<I', 320))    # hop_length
        f.write(struct.pack('<I', 16000))  # sample_rate
        f.write(struct.pack('<I', 1280))   # n_fft (hop_length * 4)
        f.write(struct.pack('<I', 2048))   # vq_dim
        f.write(struct.pack('<I', 65536))  # codebook_size (4^8)
        f.write(struct.pack('<I', 8))      # codebook_dim (len(levels))

        for name, data in tensors:
            write_tensor(f, name, data, use_f16=use_f16)

    file_size = os.path.getsize(output_path)
    print(f"Done! File size: {file_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export xcodec2 decoder to GGML format')
    parser.add_argument('--model', type=str, default='HKUST-Audio/xcodec2',
                        help='HuggingFace model path or local path')
    parser.add_argument('--output', type=str, default='xcodec2-decoder.bin',
                        help='Output binary file path')
    parser.add_argument('--f16', action='store_true',
                        help='Use float16 for large tensors')
    args = parser.parse_args()
    export_model(args.model, args.output, use_f16=args.f16)
