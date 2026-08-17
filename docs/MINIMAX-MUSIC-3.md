# MiniMax Music 3

acestep.cpp runs two architectures. ACE-Step is the default; MiniMax Music 3
(MM3) is selected automatically when the resolved diffusion model is one, and
nothing on the ACE-Step path changes when both are installed.

## Files

MM3 ships as three files in the ComfyUI component layout. `--models <dir>` is
scanned flat *and* in the three subdirectories, so a ComfyUI checkout resolves
from a single argument:

```
<models>/text_encoders/minimax_music3_text_encoder_pruned_Q8_0.gguf   8.9 GB
<models>/diffusion_models/MiniMax-Music3-Q8_0.gguf                    2.7 GB
<models>/vae/minimax_music3_dav.safetensors                           217 MB
```

`./models.sh --mm3` downloads that set. `--quant Q4_0` and `--quant BF16` pick
the other packagings.

Each file holds more than its name suggests:

| File | Contents |
|---|---|
| text encoder GGUF | global LM (Qwen3 8B, pruned embeddings), RVQ depth decoder, and the whole HF `tokenizer.json` as an I8 tensor |
| diffusion GGUF | flow matching DiT (36L, dim 2048) and the condition encoder |
| DAV safetensors | flow VAE decoder only (there is no encoder: MM3 is text to music) |

Both GGUFs declare `general.architecture = "minimax_music3"`, so the registry
tells them apart by probing tensor names, the same way ComfyUI's
`model_detection.py` does.

## Running

```
./build/ace-synth --models /path/to/models \
    --caption "warm lo-fi hip hop, mellow rhodes piano, relaxed boom bap drums" \
    --lyrics "[verse]
city lights are fading slow
[chorus]
we let the quiet take us home" \
    --duration 30 --out song.mp3
```

Every flag has a request-JSON equivalent, so `--request card.json` works too.
The run writes `song0.mp3` plus `song0.json`, a replay card carrying the codes
and the two seeds that track consumed; feeding it back reproduces the track
deterministically without resampling the AR stage.

`ace-server` serves MM3 on the same `/synth` endpoint. The response is the
usual multipart body, but the latent parts are empty: an MM3 latent is 128
channels at 86.13 Hz and cannot be fed to the ACE-Step `/vae` endpoint.
Source and reference audio are ignored (no encoder in the checkpoint).

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--duration <s>` | 60 | target length, capped at 360 s (9000 frames) |
| `--steps <N>` | 30 | Euler steps per DiT window |
| `--seed <N>` | random | DiT noise seed (Philox) |
| `--lm-seed <N>` | random | AR sampling seed (mt19937_64) |
| `--lm-cfg <F>` | 1.5 | CFG on the AR logits |
| `--dit-cfg <F>` | 1.7 | CFG on the DiT velocity |
| `--lm-top-k <N>` | 50 | AR top-k |
| `--max-seq <N>` | auto | cap the LM KV cache (default prompt + frames + 1) |
| `--ar-backend <dev>` | default | run the AR stage (LM + depth) on this device |
| `--synth-backend <dev>` | default | run synthesis (DiT + VAE) on this device |
| `--hybrid` | off | shorthand for `--ar-backend ROCm0 --synth-backend Vulkan0` |
| `--keep-loaded` | off | never evict between stages |

`--out <path>` names the output for a flag-built request and also overrides
the basename of `--request` runs; its extension picks mp3 vs wav.

## Hybrid execution

On a machine with more than one usable backend the two stages have different
sweet spots. Measured on an RX 7900 XT (gfx1100, ROCm 7.2 vs RADV, same
binary, warm caches, 10 s clip, 30 steps):

| AR | Synth | AR stage | DiT | Total |
|---|---|---|---|---|
| ROCm0 | ROCm0 | 8.6 s | 44.3 s (738 ms/step) | 57.6 s |
| Vulkan0 | Vulkan0 | 12.0 s | 15.7 s (262 ms/step) | 33.4 s |
| **ROCm0** | **Vulkan0** | **8.4 s** | **15.8 s** | **28.9 s** |

ROCm is ~1.4x faster on the autoregressive stage, RADV is ~2.8x faster on the
DiT, so running each stage where it wins beats either single device by 13%+
over the best of them. The first HIP execution in a process also pays a
one-time kernel JIT cost (~13 s on the AR stage cold); benchmark after a warm
run or the ROCm numbers look far worse than they are.

The switch rides the stage boundary: under EVICT_STRICT the AR handles go out
of scope, the shared backend cache drops to zero residents, and the next
`require` rebuilds it on the other device. Because of that, per-stage backends
are incompatible with `--keep-loaded` (a warning is printed and the run stays
on the first device), and they need a binary with both backends linked:

```
./buildhybrid.sh          # HIP + Vulkan in one binary, gfx1100 by default
./build-hybrid/ace-synth --models models --request song.json --hybrid
```

`--hybrid` hardcodes the AMD pairing measured above; elsewhere use the two
`--*-backend` flags with device names as `ggml_backend_dev_name` reports them
(CUDA0, ROCm0, Vulkan0, CPU). Everything is opt-in: without the flags the
whole run stays on one device, exactly as before.

## Pipeline

```
prompt -> global LM (batch 2: cond, uncond) -> logit CFG -> semantic code c0
       -> RVQ depth decoder, 7 acoustic codebooks per frame
       -> 8 hidden states per 25 Hz frame
       -> condition encoder (mix 8 -> conv1d k3 -> nearest resample to 86.13 Hz)
       -> flow matching DiT, Euler, velocity CFG, 200 frame windows
       -> flow VAE decoder -> 44.1 kHz stereo
```

Memory splits into two stages that never coexist under the default eviction
policy:

| Stage | Resident | Peak |
|---|---|---|
| AR | LM 7.9 GB + depth 0.7 GB + KV cache | ~9 GB + 144 KB per KV position |
| synthesis | DiT 2.5 GB + VAE 0.1 GB | ~3 GB + activations |

The KV cache is sized `prompt + frames + 1` rather than the nominal 10240
context: the reference does the same, and RoPE extrapolates past the nominal
window, which a 5000 token prompt plus 9000 frames needs.

## Notes on fidelity

Three details are easy to get wrong and are pinned by `tests/test-mm3`:

- **`rotary_pos_emb.inv_freq` is loaded, never computed.** The stored table is
  BF16 rounded and differs from `10000^(-2i/32)` by ~3e-3 relative, which is a
  ~0.1 rad phase error at the far end of a window. It is injected into
  `ggml_rope_ext` through `freq_factors`.
- **The pruned head puts the stop token at index 0**, so a sampled index `i > 0`
  is code `i - 1`. There is no vocab mask on this path.
- **Snake alpha is used as stored.** The ACE-Step Oobleck VAE takes `exp(alpha)`
  at load; the DAV checkpoint does not, and applying it would destroy the
  decoder. The weight norm parametrization is still split in the file
  (`weight_g` / `weight_v`), so the loader folds `w = g * v / ||v||` on the host.

The sampler uses ascending sigmas with `x += dt * v`. ComfyUI writes the same
trajectory as descending sigmas, `process_timestep = 1 - sigma`, and a negated
model output; the two are algebraically identical.

Seeds do **not** transfer from ComfyUI: it draws with `torch.multinomial` on a
blake2b-derived generator, this port uses `mt19937_64(lm_seed + i)` plus Philox
for the noise. Parity is defined per component against torch references, not as
a bit-identical output.

## Windowing

The port windows the *pipeline*: the condition is cut into 200 frame windows
with hop 100, the DiT runs per window with a 172 latent overlap blend and a
carry, and the VAE decodes per window with an 86 / 258 latent crop. ComfyUI
instead samples the whole latent and windows attention inside the model. Both
are valid; they are different strategies and must not be mixed.
