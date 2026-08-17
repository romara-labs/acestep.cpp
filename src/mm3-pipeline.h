#pragma once
// mm3-pipeline.h: MiniMax Music 3 generation pipeline
//
// The pipeline owns no GGML module: it borrows them from a ModelStore
// through RAII handles, stage by stage. The AR stage holds
// { MM3_LM, MM3_DEPTH }, the synthesis stage holds { MM3_DIT, MM3_VAE };
// under the default STRICT policy the store unloads a stage's modules when
// its handles go out of scope, so the 8.9 GB of LM weights and the 2.7 GB
// of DiT weights never coexist. See the coexistence groups in
// model-store.h.
//
// mm3_pipeline_configure() records the resolved paths and parameters;
// loads happen inside generate at stage boundaries, through the store.
//
// generate() is synchronous and cancellable: the cancel flag is polled
// between AR frames, between DiT steps, and between VAE windows.
//
// This file is the MiniMax Music 3 path only. Nothing here is reachable
// from the ACE-Step pipelines, which keep their own pipeline-*.cpp.

#include "debug.h"
#include "model-store.h"
#include "request.h"

#include <atomic>
#include <string>
#include <vector>

// MM3 recipe constants, fixed by the checkpoint (see the official
// transformer / condition_encoder / vocoder configs).
static const int MM3_FRAME_RATE     = 25;    // LM frames per second
static const int MM3_MAX_FRAMES     = 9000;  // 360 s cap, reference limit
static const int MM3_CHUNK_FRAMES   = 200;   // condition window
static const int MM3_CHUNK_HOP      = 100;
static const int MM3_OVERLAP_LATENT = 172;
static const int MM3_CROP_LEFT      = 86;
static const int MM3_CROP_RIGHT     = 344 - 86;
static const int MM3_SAMPLE_RATE    = 44100;  // vocoder/config.json
static const int MM3_DEFAULT_STEPS  = 30;     // official ComfyUI workflow
static const int MM3_DEFAULT_TOP_K  = 50;

// Runtime knobs. Applied to the components as they load, so they must be
// set before the first configure and stay fixed for the process lifetime
// (graph caches and store keys bake them in).
struct MM3PipelineParams {
    bool         use_fa        = true;     // flash attention on GPU backends
    bool         use_batch_cfg = true;     // fuse cond and uncond in one decode
    bool         clamp_fp16    = false;    // clamp hidden states to FP16 range
    int          max_seq       = 0;        // LM KV cap, 0 = prompt + frames + 1
    int          max_batch     = 1;        // song batch limit, sizes the 2N KV sets
    const char * dump_dir      = nullptr;  // dump intermediate tensors

    // Hybrid execution: run each stage on a different device. Both NULL
    // (the default) keeps the single auto-best / GGML_BACKEND device for
    // the whole run, exactly as before.
    //
    // Opt-in only, because it is a measured tradeoff rather than a win:
    // on RDNA3 the AR stage is faster under ROCm while the DiT is roughly
    // 3x faster under RADV, so { ar: "ROCm0", synth: "Vulkan0" } beats
    // either device alone. The switch needs EVICT_STRICT: the backend
    // cache is process wide and can only be rebuilt while no module is
    // resident, which is the window a stage boundary opens.
    const char * ar_backend    = nullptr;  // LM + depth decoder
    const char * synth_backend = nullptr;  // DiT + condition encoder + VAE
};

// Resolved paths. lm and dit are GGUFs, vae is a safetensors file.
struct MM3ModelPaths {
    std::string lm;   // text encoder GGUF: LM + depth decoder + tokenizer
    std::string dit;  // diffusion GGUF: DiT + condition encoder
    std::string vae;  // DAV decoder safetensors
};

struct MM3Pipeline {
    ModelStore *      store = nullptr;  // borrowed, owned by the tool
    MM3ModelPaths     wanted;
    MM3PipelineParams params;
    DebugDumper       dumper = {};
};

enum MM3PipelineStatus {
    MM3_PIPELINE_OK        = 0,
    MM3_PIPELINE_FAILED    = 1,
    MM3_PIPELINE_CANCELLED = 2,
};

// Record the resolved paths and parameters for the next generate. No
// module is loaded here; a load failure surfaces as MM3_PIPELINE_FAILED
// from the generate that first requires the failing component.
void mm3_pipeline_configure(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params);

// Full text to audio generation. Seeds must be resolved by the caller
// (request_resolve_seed / request_resolve_lm_seed).
// tracks_out: lm_batch_size * synth_batch_size tracks, each planar stereo
// float [L:T][R:T] at 44100 Hz, full range (normalization and clipping
// belong to the output encoding stage). Song i samples with lm_seed + i,
// variation j with seed + j. Output order is song-major.
// cancel: optional, polled at stage boundaries. NULL disables cancellation.
// codes_out: optional, the audio_codes stream of each song, identical to
// the input codes under replay.
MM3PipelineStatus mm3_pipeline_generate(MM3Pipeline *                     p,
                                        const AceRequest &                req,
                                        std::atomic<bool> *               cancel,
                                        std::vector<std::vector<float>> & tracks_out,
                                        std::vector<std::string> *        codes_out);

// Autoregressive stage only: lm_batch_size code streams out, no synthesis.
// codes_out[i] is the audio_codes string of song i (8 comma separated
// codes per frame). Only requires the AR group from the store.
MM3PipelineStatus mm3_pipeline_lm_generate(MM3Pipeline *              p,
                                           const AceRequest &         req,
                                           std::atomic<bool> *        cancel,
                                           std::vector<std::string> & codes_out);
