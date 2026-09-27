// ace-synth.cpp: ACE-Step synthesis CLI
// Thin wrapper: parses args, scans the model registry, calls pipeline-synth,
// writes output files. Model selection (synth_model, adapter, output_format)
// comes from the request JSON. The registry resolves names to GGUF paths
// under --models <dir> and --adapters <dir>.

#include "audio-io.h"
#include "mm3-pipeline.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "request.h"
#include "synth-batch-runner.h"
#include "task-types.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    AceSynthParams d;
    ace_synth_default_params(&d);

    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> --request <json...> [options]\n"
            "       %s --models <dir> --caption <text> --lyrics <text> [options]   (MiniMax Music 3)\n\n"
            "Required:\n"
            "  --models <dir>          Directory of model files. Scanned flat and in the\n"
            "                          ComfyUI subdirs text_encoders/ diffusion_models/ vae/\n"
            "  --request <json...>     One or more request JSONs (from ace-lm --request)\n\n"
            "Optional:\n"
            "  --adapters <dir>        Directory of adapter files (enables JSON adapter field)\n"
            "  --src-audio <path>      Source audio (WAV or MP3)\n"
            "  --ref-audio <path>      Timbre reference audio (WAV or MP3)\n\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: %d)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: %d)\n"
            "  --keep-loaded           Never evict modules between stages\n\n"
            "MiniMax Music 3 (used when the resolved diffusion model is minimax_music3):\n"
            "  --caption <text>        Style prompt, builds a request without a JSON file\n"
            "  --lyrics <text>         Lyrics, same\n"
            "  --out <path>            Output file (default out.mp3). Also overrides the\n"
            "                          --request basename; extension picks mp3 vs wav\n"
            "  --duration <sec>        Target length (default 60, max 360)\n"
            "  --steps <N>             Euler steps per DiT window (default %d)\n"
            "  --seed <N>              DiT noise seed (-1 = random)\n"
            "  --lm-seed <N>           AR sampling seed (-1 = random)\n"
            "  --lm-cfg <F>            AR logit CFG scale (default 1.5)\n"
            "  --dit-cfg <F>           DiT velocity CFG scale (default 1.7)\n"
            "  --lm-top-k <N>          AR top-k (default %d)\n"
            "  --max-seq <N>           Cap the LM KV cache (default prompt + frames + 1)\n"
            "  --ar-backend <dev>      Run the AR stage (LM + depth) on this device\n"
            "  --synth-backend <dev>   Run synthesis (DiT + VAE) on this device\n"
            "  --hybrid                Shorthand for --ar-backend ROCm0 --synth-backend Vulkan0\n"
            "                          (opt-in: needs a binary built with both backends)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --no-batch-cfg          Split DiT CFG into two separate forwards\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog, prog, d.vae_chunk, d.vae_overlap, MM3_DEFAULT_STEPS, MM3_DEFAULT_TOP_K);
}

// Pick the request output_format from an --out path extension.
static bool out_format_from_path(const char * path, std::string & format) {
    std::string p   = path;
    size_t      dot = p.rfind('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string ext = p.substr(dot);
    if (ext == ".mp3") {
        format = "mp3";
        return true;
    }
    if (ext == ".wav") {
        // keep whatever bit depth the request already asks for
        if (format != "wav16" && format != "wav24" && format != "wav32") {
            format = "wav16";
        }
        return true;
    }
    return false;
}

// MiniMax Music 3 run: resolve the trio from the MM3 registry buckets,
// generate every request, write the tracks plus their replay JSON.
static int run_mm3(const ModelRegistry &            registry,
                   std::vector<AceRequest> &        reqs,
                   const std::vector<std::string> & basenames,
                   const MM3PipelineParams &        params,
                   EvictPolicy                      policy) {
    // Caller guarantees the three buckets are non-empty; a named model that
    // does not resolve is the only failure left.
    const ModelEntry * lm_entry  = &registry.mm3_lm.front();
    const ModelEntry * dit_entry = reqs[0].synth_model.empty() ?
                                       &registry.mm3_dit.front() :
                                       registry_find(registry.mm3_dit, reqs[0].synth_model.c_str());
    const ModelEntry * vae_entry =
        reqs[0].vae.empty() ? &registry.mm3_vae.front() : registry_find(registry.mm3_vae, reqs[0].vae.c_str());
    if (!dit_entry || !vae_entry) {
        fprintf(stderr, "[MM3-Synth] FATAL: requested model name not found in the MiniMax Music 3 registry\n");
        return 1;
    }

    MM3ModelPaths paths;
    paths.lm  = lm_entry->path;
    paths.dit = dit_entry->path;
    paths.vae = vae_entry->path;
    fprintf(stderr, "[MM3-Synth] LM  %s\n[MM3-Synth] DiT %s\n[MM3-Synth] VAE %s\n", paths.lm.c_str(), paths.dit.c_str(),
            paths.vae.c_str());

    ModelStore * store = store_create(policy);
    MM3Pipeline  pipe;
    pipe.store = store;
    mm3_pipeline_configure(&pipe, paths, params);

    int rc = 0;
    for (size_t ri = 0; ri < reqs.size() && rc == 0; ri++) {
        AceRequest & req = reqs[ri];
        request_resolve_seed(&req);
        request_resolve_lm_seed(&req);

        bool      is_mp3  = true;
        WavFormat wav_fmt = WAV_S16;
        if (!audio_parse_format(req.output_format.c_str(), is_mp3, wav_fmt)) {
            fprintf(stderr, "[MM3-Synth] FATAL: invalid output_format '%s'\n", req.output_format.c_str());
            rc = 1;
            break;
        }

        std::vector<std::vector<float>> tracks;
        std::vector<std::string>        codes;
        MM3PipelineStatus               st = mm3_pipeline_generate(&pipe, req, nullptr, tracks, &codes);
        if (st != MM3_PIPELINE_OK) {
            fprintf(stderr, "[MM3-Synth] ERROR: generation failed (status %d)\n", (int) st);
            rc = 1;
            break;
        }

        const int M = req.synth_batch_size < 1 ? 1 : (req.synth_batch_size > 9 ? 9 : req.synth_batch_size);
        for (size_t i = 0; i < tracks.size(); i++) {
            const char * ext = is_mp3 ? ".mp3" : ".wav";
            char         track_path[1024];
            snprintf(track_path, sizeof(track_path), "%s%d%s", basenames[ri].c_str(), (int) i, ext);
            int T_audio = (int) (tracks[i].size() / 2);
            if (!audio_write(track_path, tracks[i].data(), T_audio, MM3_SAMPLE_RATE, req.mp3_bitrate, wav_fmt,
                             req.peak_clip)) {
                fprintf(stderr, "[MM3-Synth] FATAL: failed to write %s\n", track_path);
                rc = 1;
                break;
            }
            fprintf(stderr, "[MM3-Synth] Wrote %s (%.1fs)\n", track_path, (float) T_audio / MM3_SAMPLE_RATE);

            // Replay card: the same request pinned to the codes and the two
            // seeds this track actually consumed.
            AceRequest replay       = req;
            replay.audio_codes      = codes[i / (size_t) M];
            replay.lm_seed          = req.lm_seed + (int64_t) (i / (size_t) M);
            replay.seed             = req.seed + (int64_t) (i % (size_t) M);
            replay.lm_batch_size    = 1;
            replay.synth_batch_size = 1;
            std::string json_path   = std::string(track_path);
            size_t      dot         = json_path.rfind('.');
            json_path               = (dot != std::string::npos ? json_path.substr(0, dot) : json_path) + ".json";
            request_write(&replay, json_path.c_str());
        }
    }

    store_free(store);
    return rc;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    // Defaults live in ace_synth_default_params. CLI locals read from params
    // so there is exactly one place in the codebase that picks the numbers.
    AceSynthParams params;
    ace_synth_default_params(&params);

    std::vector<const char *> request_paths;
    const char *              models_dir     = NULL;
    const char *              adapters_dir   = NULL;
    const char *              src_audio_path = NULL;
    const char *              ref_audio_path = NULL;
    const char *              dump_dir       = NULL;
    bool                      use_fa         = true;
    bool                      use_batch_cfg  = true;
    bool                      clamp_fp16     = false;
    bool                      keep_loaded    = false;
    int                       vae_chunk      = params.vae_chunk;
    int                       vae_overlap    = params.vae_overlap;

    // Flag-built request (MiniMax Music 3 convenience path). cli_req stays
    // unused unless --caption or --lyrics is given.
    AceRequest cli_req;
    request_init(&cli_req);
    bool         have_cli_req  = false;
    const char * cli_out_path  = "out.mp3";
    bool         cli_out_given = false;
    int          max_seq       = 0;
    const char * ar_backend    = nullptr;
    const char * synth_backend = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--caption") && i + 1 < argc) {
            cli_req.caption = argv[++i];
            have_cli_req    = true;
        } else if (!strcmp(argv[i], "--lyrics") && i + 1 < argc) {
            cli_req.lyrics = argv[++i];
            have_cli_req   = true;
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            cli_out_path  = argv[++i];
            cli_out_given = true;
        } else if (!strcmp(argv[i], "--duration") && i + 1 < argc) {
            cli_req.duration = (float) atof(argv[++i]);
            have_cli_req     = true;
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            cli_req.inference_steps = atoi(argv[++i]);
            have_cli_req            = true;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            cli_req.seed = atoll(argv[++i]);
            have_cli_req = true;
        } else if (!strcmp(argv[i], "--lm-seed") && i + 1 < argc) {
            cli_req.lm_seed = atoll(argv[++i]);
            have_cli_req    = true;
        } else if (!strcmp(argv[i], "--lm-cfg") && i + 1 < argc) {
            cli_req.lm_cfg = (float) atof(argv[++i]);
            have_cli_req   = true;
        } else if (!strcmp(argv[i], "--dit-cfg") && i + 1 < argc) {
            cli_req.dit_cfg = (float) atof(argv[++i]);
            have_cli_req    = true;
        } else if (!strcmp(argv[i], "--lm-top-k") && i + 1 < argc) {
            cli_req.lm_top_k = atoi(argv[++i]);
            have_cli_req     = true;
        } else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc) {
            max_seq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ar-backend") && i + 1 < argc) {
            ar_backend = argv[++i];
        } else if (!strcmp(argv[i], "--synth-backend") && i + 1 < argc) {
            synth_backend = argv[++i];
        } else if (!strcmp(argv[i], "--hybrid")) {
            ar_backend    = "ROCm0";
            synth_backend = "Vulkan0";
        } else if (!strcmp(argv[i], "--keep-loaded")) {
            keep_loaded = true;
        } else if (!strcmp(argv[i], "--request")) {
            // Collect all following non-option args
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                request_paths.push_back(argv[++i]);
            }
        } else if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--adapters") && i + 1 < argc) {
            adapters_dir = argv[++i];
        } else if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--ref-audio") && i + 1 < argc) {
            ref_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(argv[i], "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(argv[i], "--no-batch-cfg")) {
            use_batch_cfg = false;
        } else if (!strcmp(argv[i], "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!models_dir) {
        usage(argv[0]);
        return 1;
    }
    if (request_paths.empty() && !have_cli_req) {
        usage(argv[0]);
        return 1;
    }

    // Parse all requests first: the first request drives model selection.
    // With no --request, the flags built one and the output name comes from
    // --out instead of a JSON path.
    std::vector<AceRequest>  reqs;
    std::vector<std::string> basenames;
    if (request_paths.empty()) {
        reqs.push_back(cli_req);
        std::string base = cli_out_path;
        size_t      dot  = base.rfind('.');
        if (dot != std::string::npos) {
            base = base.substr(0, dot);
        }
        basenames.push_back(base);
        if (!out_format_from_path(cli_out_path, reqs[0].output_format)) {
            fprintf(stderr, "[Ace-Synth] FATAL: --out must end in .mp3 or .wav\n");
            return 1;
        }
    } else {
        reqs.resize(request_paths.size());
        basenames.resize(request_paths.size());
        for (size_t ri = 0; ri < request_paths.size(); ri++) {
            const char * rpath = request_paths[ri];
            if (!request_parse(&reqs[ri], rpath)) {
                fprintf(stderr, "[Ace-Synth] FATAL: failed to parse %s\n", rpath);
                return 1;
            }
            request_dump(&reqs[ri], stderr);
            // output basename: strip .json suffix
            basenames[ri] = rpath;
            size_t dot    = basenames[ri].rfind(".json");
            if (dot != std::string::npos) {
                basenames[ri] = basenames[ri].substr(0, dot);
            }
        }
        // An explicit --out wins over the JSON basename for every request,
        // otherwise it is silently ignored and the run overwrites whatever
        // the request path maps to. Its extension also picks the format,
        // same as the flag-built path.
        if (cli_out_given) {
            std::string base = cli_out_path;
            size_t      dot  = base.rfind('.');
            if (dot != std::string::npos) {
                base = base.substr(0, dot);
            }
            for (size_t ri = 0; ri < request_paths.size(); ri++) {
                basenames[ri] = base;
            }
            if (!out_format_from_path(cli_out_path, reqs[0].output_format)) {
                fprintf(stderr, "[Ace-Synth] FATAL: --out must end in .mp3 or .wav\n");
                return 1;
            }
            fprintf(stderr, "[Ace-Synth] --out overrides basename for %zu request(s)\n", reqs.size());
        }
    }
    int batch_n = (int) reqs.size();
    fprintf(stderr, "[Ace-Synth] Batch: %d request(s)\n", batch_n);

    // Scan the registry and resolve model paths from the first request.
    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Ace-Synth] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }
    if (adapters_dir) {
        registry_scan_adapters(&registry, adapters_dir);
    }

    // Architecture dispatch. A synth_model naming an MM3 diffusion GGUF
    // picks MiniMax Music 3 explicitly; otherwise MM3 is used only when the
    // registry has a complete MM3 set and no ACE-Step DiT, so a directory
    // holding both keeps resolving to ACE-Step exactly as before.
    bool use_mm3 = false;
    if (!reqs[0].synth_model.empty()) {
        use_mm3 = registry_find(registry.mm3_dit, reqs[0].synth_model.c_str()) != nullptr;
    } else {
        use_mm3 = registry.dit.empty() && registry_has_mm3(registry);
    }

    if (use_mm3) {
        if (!registry_has_mm3(registry)) {
            fprintf(stderr,
                    "[Ace-Synth] FATAL: MiniMax Music 3 needs all three files "
                    "(text encoder GGUF, diffusion GGUF, DAV safetensors)\n");
            return 1;
        }
        MM3PipelineParams mp;
        mp.use_fa        = use_fa;
        mp.use_batch_cfg = use_batch_cfg;
        mp.clamp_fp16    = clamp_fp16;
        mp.max_seq       = max_seq;
        mp.max_batch     = 1;
        for (const auto & r : reqs) {
            int n = r.lm_batch_size < 1 ? 1 : r.lm_batch_size;
            if (n > mp.max_batch) {
                mp.max_batch = n;
            }
        }
        mp.dump_dir      = dump_dir;
        mp.ar_backend    = ar_backend;
        mp.synth_backend = synth_backend;
        if ((ar_backend || synth_backend) && keep_loaded) {
            fprintf(stderr,
                    "[Ace-Synth] WARNING: per-stage backends need eviction between stages; "
                    "--keep-loaded pins the first device for the whole run\n");
        }
        return run_mm3(registry, reqs, basenames, mp, keep_loaded ? EVICT_NEVER : EVICT_STRICT);
    }

    // ACE-Step path: unchanged from here down.
    for (int ri = 0; ri < batch_n; ri++) {
        if (reqs[ri].caption.empty() && reqs[ri].task_type != TASK_LEGO && reqs[ri].task_type != TASK_EXTRACT &&
            reqs[ri].task_type != TASK_COMPLETE) {
            fprintf(stderr, "[Ace-Synth] FATAL: caption is empty in request %d\n", ri);
            return 1;
        }
    }
    if (registry.dit.empty() || registry.text_enc.empty() || registry.vae.empty()) {
        fprintf(stderr, "[Ace-Synth] FATAL: registry needs DiT, text-encoder and VAE models\n");
        return 1;
    }
    const ModelEntry * dit_entry =
        reqs[0].synth_model.empty() ? &registry.dit[0] : registry_find(registry.dit, reqs[0].synth_model.c_str());
    if (!dit_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: synth_model '%s' not found in registry\n", reqs[0].synth_model.c_str());
        return 1;
    }
    const ModelEntry * vae_entry =
        reqs[0].vae.empty() ? &registry.vae[0] : registry_find(registry.vae, reqs[0].vae.c_str());
    if (!vae_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: vae '%s' not found in registry\n", reqs[0].vae.c_str());
        return 1;
    }
    const AdapterEntry * adapter_entry = NULL;
    if (!reqs[0].adapter.empty()) {
        adapter_entry = registry_find_adapter(registry, reqs[0].adapter.c_str());
        if (!adapter_entry) {
            fprintf(stderr, "[Ace-Synth] FATAL: adapter '%s' not found (use --adapters <dir>)\n",
                    reqs[0].adapter.c_str());
            return 1;
        }
    }

    // Resolve output_format to (is_mp3, wav_fmt).
    bool      is_mp3  = true;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(reqs[0].output_format.c_str(), is_mp3, wav_fmt)) {
        fprintf(stderr, "[Ace-Synth] FATAL: invalid output_format '%s' (use: mp3, wav16, wav24, wav32)\n",
                reqs[0].output_format.c_str());
        return 1;
    }

    // Fill params from registry lookups and CLI flags.
    params.text_encoder_path = registry.text_enc[0].path.c_str();
    params.dit_path          = dit_entry->path.c_str();
    params.vae_path          = vae_entry->path.c_str();
    params.adapter_path      = adapter_entry ? adapter_entry->path.c_str() : NULL;
    params.adapter_scale     = reqs[0].adapter_scale;
    params.use_fa            = use_fa;
    params.use_batch_cfg     = use_batch_cfg;
    params.clamp_fp16        = clamp_fp16;
    params.vae_chunk         = vae_chunk;
    params.vae_overlap       = vae_overlap;
    params.dump_dir          = dump_dir;

    // Local store with the default STRICT policy: at most one GPU module
    // resident at a time for this one-shot CLI. No module sharing across runs,
    // so EVICT_STRICT frees the DiT before the VAE loads, and so on.
    // --keep-loaded trades that for VRAM.
    ModelStore * store = store_create(keep_loaded ? EVICT_NEVER : EVICT_STRICT);
    AceSynth *   ctx   = ace_synth_load(store, &params);
    if (!ctx) {
        store_free(store);
        return 1;
    }

    // Read source audio (cover/lego mode)
    float * src_interleaved = NULL;
    int     src_len         = 0;
    if (src_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --src-audio %s\n", src_audio_path);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Source audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        src_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        src_len = T_audio;
    }

    // Read reference audio (timbre conditioning)
    float * ref_interleaved = NULL;
    int     ref_len         = 0;
    if (ref_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(ref_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --ref-audio %s\n", ref_audio_path);
            free(src_interleaved);
            ace_synth_free(ctx);
            store_free(store);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Reference audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);
        ref_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        ref_len = T_audio;
    }

    // Generate every request in one DiT batch. synth_batch_size expands each
    // request into per-seed variants in groups[0]. Total clamped to DiT max 9.
    int total_alloc = 0;
    for (int ri = 0; ri < batch_n; ri++) {
        int sbs = reqs[ri].synth_batch_size;
        total_alloc += sbs < 1 ? 1 : (sbs > 9 ? 9 : sbs);
    }
    if (total_alloc > 9) {
        fprintf(stderr, "[Ace-Synth] Batch %d exceeds DiT max 9, clamping\n", total_alloc);
        total_alloc = 9;
    }
    std::vector<AceAudio>                all_audio(total_alloc);
    std::vector<std::string>             all_basenames(total_alloc);
    std::vector<int>                     all_synth_indices(total_alloc);
    std::vector<std::vector<AceRequest>> groups(1);
    groups[0].reserve(total_alloc);

    int off = 0;
    for (int ri = 0; ri < batch_n && off < total_alloc; ri++) {
        int sbs = reqs[ri].synth_batch_size;
        if (sbs < 1) {
            sbs = 1;
        }
        if (sbs > 9) {
            sbs = 9;
        }
        if (off + sbs > total_alloc) {
            sbs = total_alloc - off;
        }

        // resolve seed once per original request
        request_resolve_seed(&reqs[ri]);
        const long long base_seed = reqs[ri].seed;

        for (int i = 0; i < sbs; i++) {
            AceRequest r = reqs[ri];
            r.seed       = base_seed + i;
            groups[0].push_back(r);
            all_basenames[off + i]     = basenames[ri];
            all_synth_indices[off + i] = i;
        }
        off += sbs;
    }

    if (total_alloc > 1) {
        fprintf(stderr, "[Ace-Synth] Batch: %d track(s) from %d request(s)\n", total_alloc, batch_n);
    }

    // Two-phase run: DiT resident for all groups, then VAE for all jobs.
    // The CLI does not expose latent IO yet: source and reference are always
    // audio, latent capture is disabled. The server reuses this runner with
    // the full feature.
    const int rc = synth_batch_run(ctx, groups, src_interleaved, src_len, NULL, 0, ref_interleaved, ref_len, NULL, 0,
                                   all_audio.data());
    if (rc != 0) {
        fprintf(stderr, "[Ace-Synth] ERROR: batch run failed\n");
        for (auto & a : all_audio) {
            ace_audio_free(&a);
        }
        free(src_interleaved);
        free(ref_interleaved);
        ace_synth_free(ctx);
        store_free(store);
        return 1;
    }

    // Write output files
    for (int b = 0; b < (int) all_audio.size(); b++) {
        if (!all_audio[b].samples) {
            continue;
        }
        const char * ext = is_mp3 ? ".mp3" : ".wav";
        char         out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s%d%s", all_basenames[b].c_str(), all_synth_indices[b], ext);
        if (!audio_write(out_path, all_audio[b].samples, all_audio[b].n_samples, 48000, groups[0][b].mp3_bitrate,
                         wav_fmt)) {
            fprintf(stderr, "[Ace-Synth Batch%d] FATAL: failed to write %s\n", b, out_path);
        }
        ace_audio_free(&all_audio[b]);
    }

    free(src_interleaved);
    free(ref_interleaved);
    ace_synth_free(ctx);
    store_free(store);
    fprintf(stderr, "[Ace-Synth] All done\n");
    return 0;
}
