#!/usr/bin/env python3
"""Numpy reference for the MiniMax Music 3 DAV decoder.

Recomputes the decoder straight from minimax_music3_dav.safetensors in F64,
independently of the ggml graph, and compares against what test-mm3 produced.
This is the parity check for the pieces the port had to invent: the runtime
weight-norm folding (the file still stores weight_g / weight_v), the
ConvTranspose1d permutation feeding col2im_1d, and the snake activation
(alpha used as stored, not exp(alpha)).

Usage:
  ./build/test-mm3 <models_dir> --skip-lm --dump-vae /tmp/vae
  python3 tests/mm3-vae-ref.py <models_dir>/vae/minimax_music3_dav.safetensors /tmp/vae

Needs numpy only. No torch, no original checkpoint.
"""
import json
import struct
import sys

import numpy as np

STRIDES = [8, 8, 4, 2]
DILATIONS = [1, 3, 9]


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    out = {}
    dtypes = {"F32": "<f4", "F16": "<f2", "BF16": None}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = blob[start:end]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=dtypes[meta["dtype"]]).astype(np.float32)
        out[name] = arr.reshape(meta["shape"]).astype(np.float64)
    return out


def fold_weight_norm(st, prefix):
    """w = g * v / ||v||, norm over every dim but 0 (torch weight_norm default)."""
    g = st[prefix + ".weight_g"]
    v = st[prefix + ".weight_v"]
    norm = np.sqrt((v ** 2).sum(axis=(1, 2), keepdims=True))
    return v * (g / norm)


def conv1d(x, w, b, pad=0, dilation=1):
    """x [IC, T], w [OC, IC, K] -> [OC, T_out]."""
    OC, IC, K = w.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad)))
    T_out = x.shape[1] - dilation * (K - 1)
    # gather the dilated taps then one GEMM per tap
    y = np.zeros((OC, T_out))
    for k in range(K):
        off = k * dilation
        y += w[:, :, k] @ x[:, off:off + T_out]
    return y + b[:, None]


def conv_transpose1d(x, w, b, stride, pad):
    """x [IC, T], w [IC, OC, K] -> [OC, T*stride]. Torch layout."""
    IC, OC, K = w.shape
    T = x.shape[1]
    full = np.zeros((OC, (T - 1) * stride + K))
    for k in range(K):
        contrib = w[:, :, k].T @ x  # [OC, T]
        idx = np.arange(T) * stride + k
        np.add.at(full, (slice(None), idx), contrib)
    out = full[:, pad:full.shape[1] - pad]
    return out + b[:, None]


def snake(x, alpha):
    """y = x + sin(alpha * x)^2 / (alpha + 1e-9), alpha [1, C, 1] -> [C, 1]."""
    a = alpha.reshape(-1, 1)
    return x + np.sin(a * x) ** 2 / (a + 1e-9)


def res_unit(x, st, prefix, dilation):
    skip = x
    x = snake(x, st[prefix + ".block.0.alpha"])
    x = conv1d(x, fold_weight_norm(st, prefix + ".block.1"), st[prefix + ".block.1.bias"],
               pad=3 * dilation, dilation=dilation)
    x = snake(x, st[prefix + ".block.2.alpha"])
    x = conv1d(x, fold_weight_norm(st, prefix + ".block.3"), st[prefix + ".block.3.bias"])
    return skip + x


def decode_side(st, latent):
    """latent [64, T] -> waveform [T * 512]."""
    x = conv1d(latent, st["dec_in_proj.weight"], st["dec_in_proj.bias"])
    x = conv1d(x, fold_weight_norm(st, "decoder.model.0"), st["decoder.model.0.bias"], pad=3)

    for i, stride in enumerate(STRIDES):
        p = f"decoder.model.{i + 1}"
        x = snake(x, st[p + ".block.0.alpha"])
        x = conv_transpose1d(x, fold_weight_norm(st, p + ".block.1"), st[p + ".block.1.bias"],
                             stride=stride, pad=(2 * stride - stride) // 2)
        for r, dil in enumerate(DILATIONS):
            x = res_unit(x, st, f"{p}.block.{r + 2}", dil)

    x = snake(x, st["decoder.model.5.alpha"])
    x = conv1d(x, fold_weight_norm(st, "decoder.model.6"), st["decoder.model.6.bias"], pad=3)
    return np.tanh(x)[0]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    st_path, prefix = sys.argv[1], sys.argv[2]

    st = load_safetensors(st_path)
    latent = np.fromfile(prefix + "-in.f32", dtype="<f4").astype(np.float64)
    got = np.fromfile(prefix + "-out.f32", dtype="<f4").astype(np.float64)

    T = latent.size // 128
    latent = latent.reshape(128, T)
    print(f"latent [128, {T}] -> {T * 512} samples per side")

    want = np.empty(T * 512 * 2)
    for ch in range(2):
        want[ch::2] = decode_side(st, latent[ch * 64:(ch + 1) * 64])

    assert got.size == want.size, f"size mismatch: got {got.size}, want {want.size}"

    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
    mae = float(np.abs(got - want).mean())
    mx = float(np.abs(got - want).max())
    print(f"cosine     {cos:.8f}")
    print(f"mean |err| {mae:.8f}")
    print(f"max  |err| {mx:.8f}")
    print(f"ref rms    {want.std():.6f}")

    # The port runs F32 activations with F16 conv weights, so agreement with
    # an F64 reference is limited by that, not by the mapping.
    ok = cos > 0.999
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
