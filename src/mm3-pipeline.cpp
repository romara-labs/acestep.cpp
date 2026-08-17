// mm3-pipeline.cpp: full MiniMax Music 3 pipeline, text to stereo float
//
// Stages, mirroring the reference modular pipeline:
//   prompt assembly -> global LM batch 2 [cond, uncond] with logit CFG
//   -> RVQ depth decoder per frame (7 acoustic codebooks, shared CFG)
//   -> condition encoder per 200 frame window
//   -> flow matching DiT (Euler steps, velocity CFG, overlap blending)
//   -> flow VAE decoder -> crop and stitch -> 44.1 kHz stereo.
//
// Sampler convention: ascending sigmas sig[i] = 1 - linspace(1, 1/N, N)[i]
// with sig[N] = 1 and the update x += dt * v. That is the official
// FlowMatchEulerDiscreteScheduler recipe (shift 1.0, invert_sigmas true,
// num_train_timesteps 1) written the other way round, and it is
// algebraically identical to the ComfyUI form (descending sigma,
// process_timestep = 1 - sigma, negated model output).
//
// Windowing follows the pipeline-level strategy: the condition is cut in
// 200 frame windows with hop 100, the DiT runs per window with a 172
// latent overlap blend and a carry, and the VAE decodes per window with an
// 86 / 258 latent crop. The ComfyUI wrapper instead samples the whole
// latent and windows attention inside the model; the two are different
// strategies and must not be mixed.

#include "mm3-pipeline.h"

#include "mm3-prompt.h"
#include "philox.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

// CFG on the semantic logits: the candidate set is the top k of the
// CONDITIONAL branch, the scores are guided = uncond + (cond - uncond)*cfg
// on that set, then softmax + multinomial. Matches `_guided_c0` followed
// by `sample_topk` in the reference (the second top-k there is a no-op
// because the mask already left exactly k candidates).
static int mm3_cfg_sample_cond_ranked(const float *     cond,
                                      const float *     uncond,
                                      int               n,
                                      float             cfg,
                                      int               top_k,
                                      std::mt19937_64 & rng) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve((size_t) n);
    for (int id = 0; id < n; id++) {
        ranked.push_back({ cond[id], id });
    }
    int k = top_k < n ? top_k : n;
    if (k < 1) {
        k = 1;
    }
    std::partial_sort(
        ranked.begin(), ranked.begin() + k, ranked.end(),
        [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
    ranked.resize((size_t) k);

    std::vector<float> guided((size_t) k);
    float              mx = -INFINITY;
    for (int i = 0; i < k; i++) {
        int id             = ranked[(size_t) i].second;
        guided[(size_t) i] = uncond[id] + (cond[id] - uncond[id]) * cfg;
        mx                 = guided[(size_t) i] > mx ? guided[(size_t) i] : mx;
    }
    float sum = 0;
    for (int i = 0; i < k; i++) {
        guided[(size_t) i] = expf(guided[(size_t) i] - mx);
        sum += guided[(size_t) i];
    }
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float                                 r   = uni(rng) * sum;
    float                                 acc = 0;
    for (int i = 0; i < k; i++) {
        acc += guided[(size_t) i];
        if (r <= acc) {
            return ranked[(size_t) i].second;
        }
    }
    return ranked[(size_t) (k - 1)].second;
}

// Depth acoustic CFG: guided = uncond + (cond - uncond) * cfg over the
// whole codebook, then the top k of the GUIDED scores is sampled.
static int mm3_cfg_sample_guided(const float *     cond,
                                 const float *     uncond,
                                 int               n,
                                 float             cfg,
                                 int               top_k,
                                 std::mt19937_64 & rng) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve((size_t) n);
    for (int id = 0; id < n; id++) {
        ranked.push_back({ uncond[id] + (cond[id] - uncond[id]) * cfg, id });
    }
    int k = top_k < n ? top_k : n;
    if (k < 1) {
        k = 1;
    }
    std::partial_sort(
        ranked.begin(), ranked.begin() + k, ranked.end(),
        [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
    ranked.resize((size_t) k);

    std::vector<float> probs((size_t) k);
    float              mx  = ranked[0].first;
    float              sum = 0;
    for (int i = 0; i < k; i++) {
        probs[(size_t) i] = expf(ranked[(size_t) i].first - mx);
        sum += probs[(size_t) i];
    }
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float                                 r   = uni(rng) * sum;
    float                                 acc = 0;
    for (int i = 0; i < k; i++) {
        acc += probs[(size_t) i];
        if (r <= acc) {
            return ranked[(size_t) i].second;
        }
    }
    return ranked[(size_t) (k - 1)].second;
}

static bool mm3_is_cancelled(std::atomic<bool> * cancel) {
    return cancel && cancel->load();
}

// Point the next module load at `device`, if the caller asked for a
// per-stage device at all. The backend cache is process wide, so the
// switch only lands while nothing is resident; when a module is still
// held (EVICT_NEVER, or a caller keeping a handle across the boundary)
// the request is reported and the resident device is kept rather than
// silently ignored.
//
// A NULL device must clear the override, not skip: leaving a stale pin
// from a previous stage would run the next stage on a device nobody
// asked for (AR on ROCm0, then an unpinned synth silently inheriting
// ROCm0 instead of the default choice).
static void mm3_select_backend(const char * device, const char * stage) {
    if (!device) {
        backend_set_override(nullptr);
        return;
    }
    if (!backend_is_idle()) {
        fprintf(stderr, "[MM3] %s wanted %s but %s is still resident, staying on it (needs EVICT_STRICT)\n", stage,
                device, backend_current_name());
        return;
    }
    fprintf(stderr, "[MM3] %s stage on %s\n", stage, device);
    backend_set_override(device);
}

void mm3_pipeline_configure(MM3Pipeline * p, const MM3ModelPaths & paths, const MM3PipelineParams & params) {
    p->wanted = paths;
    p->params = params;
    debug_init(&p->dumper, params.dump_dir);
    if (params.clamp_fp16) {
        fprintf(stderr, "[MM3] FP16 clamp enabled\n");
    }
    if (!params.use_fa) {
        fprintf(stderr, "[MM3] Flash attention disabled\n");
    }
}

// Resolved per-request recipe values: the request carries ACE-Step
// "0 = auto" sentinels, MM3 has its own defaults for them.
struct MM3Recipe {
    int   steps;
    int   top_k;
    float lm_cfg;
    float dit_cfg;
    int   max_frames;
};

static MM3Recipe mm3_resolve_recipe(const AceRequest & req) {
    MM3Recipe r;
    r.steps   = req.inference_steps > 0 ? req.inference_steps : MM3_DEFAULT_STEPS;
    r.top_k   = req.lm_top_k > 0 ? req.lm_top_k : MM3_DEFAULT_TOP_K;
    r.lm_cfg  = req.lm_cfg;
    r.dit_cfg = req.dit_cfg;

    float duration = req.duration > 0.0f ? req.duration : 60.0f;
    r.max_frames   = (int) (duration * MM3_FRAME_RATE);
    if (r.max_frames > MM3_MAX_FRAMES) {
        r.max_frames = MM3_MAX_FRAMES;
    }
    if (r.max_frames < 1) {
        r.max_frames = 1;
    }
    return r;
}

// Require helpers: one place builds the store key of each component from
// the wanted paths and the process-lifetime params, and applies the
// runtime knobs after every require (idempotent on cache hits).
static MM3LM * mm3_require_lm(MM3Pipeline * p, int max_seq, int n_kv_sets) {
    ModelKey k = { MODEL_MM3_LM, p->wanted.lm, max_seq, n_kv_sets, "", 1.0f };
    MM3LM *  m = store_require_mm3_lm(p->store, k);
    if (m) {
        if (!p->params.use_fa) {
            m->use_flash_attn = false;
        }
        m->clamp_fp16 = p->params.clamp_fp16;
    }
    return m;
}

static MM3Depth * mm3_require_depth(MM3Pipeline * p) {
    ModelKey k = { MODEL_MM3_DEPTH, p->wanted.lm, 0, 0, "", 1.0f };
    return store_require_mm3_depth(p->store, k);
}

static MM3DiT * mm3_require_dit(MM3Pipeline * p) {
    ModelKey k = { MODEL_MM3_DIT, p->wanted.dit, 0, 0, "", 1.0f };
    MM3DiT * m = store_require_mm3_dit(p->store, k);
    if (m) {
        if (!p->params.use_fa) {
            m->use_flash_attn = false;
        }
        m->clamp_fp16 = p->params.clamp_fp16;
    }
    return m;
}

static MM3VAE * mm3_require_vae(MM3Pipeline * p) {
    ModelKey k = { MODEL_MM3_VAE, p->wanted.vae, 0, 0, "", 1.0f };
    return store_require_mm3_vae(p->store, k);
}

// Serialize a code stream to the audio_codes wire format: flat comma
// separated, 8 values per frame (semantic then the 7 acoustic codebooks)
static std::string mm3_codes_serialize(const std::vector<int> & codes) {
    std::string out;
    out.reserve(codes.size() * 6);
    char buf[16];
    for (size_t i = 0; i < codes.size(); i++) {
        snprintf(buf, sizeof(buf), i ? ",%d" : "%d", codes[i]);
        out += buf;
    }
    return out;
}

// Parse and validate the audio_codes wire format. Empty result = invalid.
static std::vector<int> mm3_codes_parse(const std::string & s) {
    std::vector<int> out;
    const char *     c = s.c_str();
    while (*c) {
        char * end  = nullptr;
        long   code = strtol(c, &end, 10);
        if (end == c) {
            fprintf(stderr, "[MM3-AR] FATAL: invalid audio_codes near \"%.16s\"\n", c);
            return {};
        }
        out.push_back((int) code);
        c = *end == ',' ? end + 1 : end;
    }
    if (out.size() < 16 || out.size() % 8 != 0) {
        fprintf(stderr, "[MM3-AR] FATAL: audio_codes needs 8 codes per frame, at least 2 frames (got %zu values)\n",
                out.size());
        return {};
    }
    for (size_t i = 0; i < out.size(); i++) {
        int hi = (i % 8 == 0) ? MM3_SEMANTIC_VOCAB : MM3Depth::VOCAB;
        if (out[i] < 0 || out[i] >= hi) {
            fprintf(stderr, "[MM3-AR] FATAL: audio_codes value %d out of range at index %zu\n", out[i], i);
            return {};
        }
    }
    return out;
}

// Build the conditional prompt for a request. Returns an empty vector on
// a tokenizer failure or an over-long prompt.
static std::vector<int> mm3_prompt_ids(BPETokenizer * tok, const AceRequest & req) {
    std::vector<int> ids =
        mm3_build_prompt_ids([&](const std::string & s) { return bpe_encode(tok, s, false); }, req.caption, req.lyrics);
    if ((int) ids.size() > MM3_MAX_PROMPT_TOKENS) {
        fprintf(stderr, "[MM3-Prompt] FATAL: %zu tokens, maximum is %d\n", ids.size(), MM3_MAX_PROMPT_TOKENS);
        return {};
    }
    return ids;
}

// Teacher-forced replay: re-derive the 8 hidden states per frame from an
// explicit code stream, no sampling. The LM runs the whole feedback
// sequence as one full forward (conditional stream only: the hiddens never
// depended on the CFG branch), the depth decoder runs one S=8 causal
// forward per frame, positions 1..7 reproducing the step hiddens.
static MM3PipelineStatus mm3_replay_stage(MM3Pipeline *            p,
                                          MM3LM *                  lm,
                                          MM3Depth *               depth,
                                          const std::vector<int> & cond_ids,
                                          const std::vector<int> & codes,
                                          std::atomic<bool> *      cancel,
                                          std::vector<float> &     frame_hiddens) {
    Timer ar_timer;

    // frame 0 only feeds the first LM feedback, it has no hidden of its own
    int n = (int) (codes.size() / 8) - 1;

    const int H  = lm->cfg.hidden_size;
    const int V  = lm->cfg.head_vocab;
    const int NC = MM3Depth::CODEBOOKS - 1;

    if ((int) cond_ids.size() + n + 1 > lm->cfg.max_seq_len) {
        fprintf(stderr, "[MM3-AR] FATAL: audio_codes carry %d frames, over the KV budget (prompt %zu tokens)\n", n,
                cond_ids.size());
        return MM3_PIPELINE_FAILED;
    }

    mm3_lm_reset_kv(lm, 0);
    std::vector<float> logits((size_t) V);
    mm3_lm_forward(lm, cond_ids.data(), (int) cond_ids.size(), 0, logits.data());

    // Feedback embeddings f_0 .. f_{n-1}, one full-sequence forward
    std::vector<float> embeds((size_t) n * H), emb((size_t) H);
    const float        scale = 1.0f / sqrtf((float) MM3Depth::CODEBOOKS);
    for (int t = 0; t < n; t++) {
        const int * f = codes.data() + (size_t) t * 8;
        mm3_lm_embed_audio_row(lm, f[0], emb.data());
        for (int cb = 1; cb <= NC; cb++) {
            const float * row = depth->audio_embedding_row(cb, f[cb]);
            for (int j = 0; j < H; j++) {
                emb[(size_t) j] += row[j];
            }
        }
        for (int j = 0; j < H; j++) {
            emb[(size_t) j] *= scale;
        }
        memcpy(embeds.data() + (size_t) t * H, emb.data(), (size_t) H * sizeof(float));
    }
    std::vector<float> all_hidden((size_t) n * H);
    mm3_lm_forward(lm, nullptr, n, 0, logits.data(), embeds.data(), nullptr, all_hidden.data());

    // Depth hiddens per frame: the whole step sequence is known upfront, so
    // one causal S=8 forward reproduces the 7 step hiddens
    frame_hiddens.clear();
    frame_hiddens.reserve((size_t) n * 8 * H);
    std::vector<float> seq8((size_t) 8 * H), hid8((size_t) 8 * H), dlogits(MM3Depth::VOCAB);
    for (int k = 1; k <= n; k++) {
        if (mm3_is_cancelled(cancel)) {
            return MM3_PIPELINE_CANCELLED;
        }
        const int *   f  = codes.data() + (size_t) k * 8;
        const float * hk = all_hidden.data() + (size_t) (k - 1) * H;
        memcpy(seq8.data(), hk, (size_t) H * sizeof(float));
        mm3_lm_embed_audio_row(lm, f[0], seq8.data() + H);
        for (int cb = 1; cb < NC; cb++) {
            memcpy(seq8.data() + (size_t) (1 + cb) * H, depth->audio_embedding_row(cb, f[cb]),
                   (size_t) H * sizeof(float));
        }
        if (!depth->forward(seq8.data(), 8, hid8.data(), dlogits.data())) {
            return MM3_PIPELINE_FAILED;
        }
        frame_hiddens.insert(frame_hiddens.end(), hk, hk + H);
        frame_hiddens.insert(frame_hiddens.end(), hid8.begin() + H, hid8.end());
    }
    fprintf(stderr, "[MM3-AR] Replay: %d frames from audio_codes (%.1fs of music), %.1f s\n", n,
            (float) n / MM3_FRAME_RATE, ar_timer.ms() / 1000.0);
    return MM3_PIPELINE_OK;
}

// Autoregressive stage: N songs sampled in one batched pass. Records every
// song's code stream; hiddens_out is optional (null for the codes-only
// entry point).
static MM3PipelineStatus mm3_ar_stage(MM3Pipeline *                     p,
                                      MM3LM *                           lm,
                                      MM3Depth *                        depth,
                                      const std::vector<int> &          cond_ids,
                                      const MM3Recipe &                 recipe,
                                      const AceRequest &                req,
                                      std::atomic<bool> *               cancel,
                                      int                               N,
                                      std::vector<std::vector<float>> * hiddens_out,
                                      std::vector<std::vector<int>> &   codes_out,
                                      std::vector<int> &                n_frames) {
    std::vector<int> uncond_ids = mm3_build_uncond_ids(cond_ids);

    const int H  = lm->cfg.hidden_size;
    const int V  = lm->cfg.head_vocab;  // 16385: index 0 = stop, i > 0 = code i - 1
    const int NC = MM3Depth::CODEBOOKS - 1;

    // The prompt has to fit before anything else: a prefill past the cache
    // bails out early and would leave the logit buffers unwritten, so the
    // first sample would read uninitialized memory. Only reachable when
    // --max-seq is set below the prompt.
    if ((int) cond_ids.size() + 1 > lm->cfg.max_seq_len) {
        fprintf(stderr, "[MM3-AR] FATAL: prompt is %zu tokens but the KV cache holds %d\n", cond_ids.size(),
                lm->cfg.max_seq_len);
        return MM3_PIPELINE_FAILED;
    }

    // KV set blocks: [cond 0..N-1, uncond N..2N-1]. Song i samples with its
    // own stream seeded lm_seed + i, so a song's output only depends on its
    // own seed, never on its batch siblings.
    Timer                        ar_timer;
    std::vector<std::mt19937_64> rng;
    for (int i = 0; i < N; i++) {
        rng.emplace_back((uint64_t) (req.lm_seed + i));
    }

    for (int s = 0; s < 2 * N; s++) {
        mm3_lm_reset_kv(lm, s);
    }

    // Shared prompt: one prefill per CFG branch, replicated to the other
    // songs by KV copy
    Timer              prefill_timer;
    std::vector<float> logits0((size_t) V), logits1((size_t) V), hidden0((size_t) H), hidden1((size_t) H);
    mm3_lm_forward(lm, cond_ids.data(), (int) cond_ids.size(), 0, logits0.data(), nullptr, hidden0.data());
    mm3_lm_forward(lm, uncond_ids.data(), (int) uncond_ids.size(), N, logits1.data(), nullptr, hidden1.data());
    for (int i = 1; i < N; i++) {
        mm3_lm_copy_kv(lm, 0, i);
        mm3_lm_copy_kv(lm, N, N + i);
    }
    fprintf(stderr, "[MM3-AR] Prefill %.0f ms, %zu tokens, CFG=%.2f, top_k=%d, songs=%d\n", prefill_timer.ms(),
            cond_ids.size(), (double) recipe.lm_cfg, recipe.top_k, N);

    // Frame budget: requested duration, the model cap, and the KV room left
    // after the prompt (one decode per frame).
    int max_frames = recipe.max_frames;
    int kv_budget  = lm->cfg.max_seq_len - (int) cond_ids.size() - 1;
    if (max_frames > kv_budget) {
        fprintf(stderr, "[MM3-AR] Frame budget clamped to %d by the KV cache (prompt %zu tokens)\n", kv_budget,
                cond_ids.size());
        max_frames = kv_budget;
    }

    // Current per-stream state in KV set order, seeded from the shared prefill
    std::vector<float> cur_logits((size_t) 2 * N * V), cur_hidden((size_t) 2 * N * H);
    for (int i = 0; i < N; i++) {
        memcpy(cur_logits.data() + (size_t) i * V, logits0.data(), (size_t) V * sizeof(float));
        memcpy(cur_logits.data() + (size_t) (N + i) * V, logits1.data(), (size_t) V * sizeof(float));
        memcpy(cur_hidden.data() + (size_t) i * H, hidden0.data(), (size_t) H * sizeof(float));
        memcpy(cur_hidden.data() + (size_t) (N + i) * H, hidden1.data(), (size_t) H * sizeof(float));
    }

    // A song that ends stays in the batch as a passive row (stable graph
    // shapes, hot CUDA graphs); only its accumulation stops.
    std::vector<int>  frames(N, 0);
    std::vector<bool> done(N, false);
    std::vector<int>  sampled(N);  // pruned head index
    std::vector<int>  c0(N);       // semantic code = index - 1
    std::vector<int>  kv_sets(2 * N);
    for (int s = 0; s < 2 * N; s++) {
        kv_sets[(size_t) s] = s;
    }

    std::vector<float> seqN((size_t) 2 * N * 2 * H), uN((size_t) N * NC);
    std::vector<int>   codesN((size_t) N * NC);
    std::vector<float> collectedN((size_t) N * NC * H);
    std::vector<float> embN((size_t) N * H), feedback((size_t) H), depth_hid((size_t) 8 * H);
    std::vector<float> depth_logits0(MM3Depth::VOCAB), depth_logits1(MM3Depth::VOCAB);
    std::vector<float> seq0, seq1;
    std::vector<float> batch_embeds((size_t) 2 * N * H);

    for (int frame_index = 0; frame_index <= max_frames; frame_index++) {
        if (mm3_is_cancelled(cancel)) {
            return MM3_PIPELINE_CANCELLED;
        }

        bool all_done = true;
        for (int i = 0; i < N; i++) {
            sampled[(size_t) i] =
                mm3_cfg_sample_cond_ranked(cur_logits.data() + (size_t) i * V, cur_logits.data() + (size_t) (N + i) * V,
                                           V, recipe.lm_cfg, recipe.top_k, rng[(size_t) i]);
            // Pruned head: slot 0 is the stop token, code = index - 1
            if (!done[(size_t) i] && sampled[(size_t) i] == 0) {
                fprintf(stderr, "[MM3-AR] Song %d: stop token at frame %d\n", i, frame_index);
                done[(size_t) i] = true;
            }
            c0[(size_t) i] = sampled[(size_t) i] > 0 ? sampled[(size_t) i] - 1 : 0;
            if (!done[(size_t) i]) {
                all_done = false;
            }
        }
        if (all_done) {
            break;
        }

        if (p->params.use_batch_cfg) {
            // Whole frame in one fused graph across all songs: 7 depth steps
            // batch-2N, sampling in graph fed by host-drawn uniforms (per
            // song, same RNG consumption order as the split path)
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            for (int i = 0; i < N; i++) {
                float * emb_i = embN.data() + (size_t) i * H;
                mm3_lm_embed_audio_row(lm, c0[(size_t) i], emb_i);
                memcpy(seqN.data() + (size_t) i * 2 * H, cur_hidden.data() + (size_t) i * H,
                       (size_t) H * sizeof(float));
                memcpy(seqN.data() + ((size_t) i * 2 + 1) * H, emb_i, (size_t) H * sizeof(float));
                memcpy(seqN.data() + (size_t) (N + i) * 2 * H, cur_hidden.data() + (size_t) (N + i) * H,
                       (size_t) H * sizeof(float));
                memcpy(seqN.data() + ((size_t) (N + i) * 2 + 1) * H, emb_i, (size_t) H * sizeof(float));
                for (int j = 0; j < NC; j++) {
                    uN[(size_t) i * NC + j] = uni(rng[(size_t) i]);
                }
            }
            if (!depth->forward_frame(seqN.data(), uN.data(), recipe.lm_cfg, recipe.top_k, N, codesN.data(),
                                      collectedN.data())) {
                return MM3_PIPELINE_FAILED;
            }
        } else {
            // Split reference path: per song, per stream, per step
            for (int i = 0; i < N; i++) {
                float * emb_i = embN.data() + (size_t) i * H;
                mm3_lm_embed_audio_row(lm, c0[(size_t) i], emb_i);
                seq0.assign(cur_hidden.begin() + (size_t) i * H, cur_hidden.begin() + (size_t) (i + 1) * H);
                seq1.assign(cur_hidden.begin() + (size_t) (N + i) * H, cur_hidden.begin() + (size_t) (N + i + 1) * H);
                seq0.insert(seq0.end(), emb_i, emb_i + H);
                seq1.insert(seq1.end(), emb_i, emb_i + H);
                for (int cb = 1; cb < MM3Depth::CODEBOOKS; cb++) {
                    int S = cb + 1;
                    depth->forward(seq0.data(), S, depth_hid.data(), depth_logits0.data());
                    memcpy(collectedN.data() + ((size_t) i * NC + cb - 1) * H, depth_hid.data() + (size_t) (S - 1) * H,
                           (size_t) H * sizeof(float));
                    depth->forward(seq1.data(), S, depth_hid.data(), depth_logits1.data());
                    int code = mm3_cfg_sample_guided(depth_logits0.data(), depth_logits1.data(), MM3Depth::VOCAB,
                                                     recipe.lm_cfg, recipe.top_k, rng[(size_t) i]);
                    codesN[(size_t) i * NC + cb - 1] = code;
                    if (cb < MM3Depth::CODEBOOKS - 1) {
                        const float * row = depth->audio_embedding_row(cb, code);
                        seq0.insert(seq0.end(), row, row + H);
                        seq1.insert(seq1.end(), row, row + H);
                    }
                }
            }
        }

        // The recorded code stream covers one more frame than the hiddens:
        // frame 0 only feeds the first LM feedback
        for (int i = 0; i < N; i++) {
            if (done[(size_t) i]) {
                continue;
            }
            codes_out[(size_t) i].push_back(c0[(size_t) i]);
            codes_out[(size_t) i].insert(codes_out[(size_t) i].end(), codesN.begin() + (size_t) i * NC,
                                         codesN.begin() + (size_t) (i + 1) * NC);
        }

        if (frame_index > 0) {
            for (int i = 0; i < N; i++) {
                if (done[(size_t) i]) {
                    continue;
                }
                if (hiddens_out) {
                    (*hiddens_out)[(size_t) i].insert((*hiddens_out)[(size_t) i].end(),
                                                      cur_hidden.begin() + (size_t) i * H,
                                                      cur_hidden.begin() + (size_t) (i + 1) * H);
                    (*hiddens_out)[(size_t) i].insert((*hiddens_out)[(size_t) i].end(),
                                                      collectedN.begin() + (size_t) i * NC * H,
                                                      collectedN.begin() + (size_t) (i + 1) * NC * H);
                }
                frames[(size_t) i]++;
                if (frames[(size_t) i] >= max_frames) {
                    done[(size_t) i] = true;
                }
            }
            bool budget_done = true;
            for (int i = 0; i < N; i++) {
                if (!done[(size_t) i]) {
                    budget_done = false;
                }
            }
            if (budget_done) {
                break;
            }
        }

        // Frame feedback per song: the sampled c0 embedding kept in embN plus
        // the summed acoustic rows, scaled by 8^-0.5, shared by the song's
        // two CFG streams
        const float fb_scale = 1.0f / sqrtf((float) MM3Depth::CODEBOOKS);
        for (int i = 0; i < N; i++) {
            memcpy(feedback.data(), embN.data() + (size_t) i * H, (size_t) H * sizeof(float));
            for (int cb = 1; cb < MM3Depth::CODEBOOKS; cb++) {
                const float * row = depth->audio_embedding_row(cb, codesN[(size_t) i * NC + cb - 1]);
                for (int j = 0; j < H; j++) {
                    feedback[(size_t) j] += row[j];
                }
            }
            for (int j = 0; j < H; j++) {
                feedback[(size_t) j] *= fb_scale;
            }
            memcpy(batch_embeds.data() + (size_t) i * H, feedback.data(), (size_t) H * sizeof(float));
            memcpy(batch_embeds.data() + (size_t) (N + i) * H, feedback.data(), (size_t) H * sizeof(float));
        }
        if (p->params.use_batch_cfg) {
            mm3_lm_forward_batch(lm, kv_sets.data(), 2 * N, batch_embeds.data(), cur_logits.data(), cur_hidden.data());
        } else {
            for (int s = 0; s < 2 * N; s++) {
                mm3_lm_forward(lm, nullptr, 1, s, cur_logits.data() + (size_t) s * V,
                               batch_embeds.data() + (size_t) s * H, cur_hidden.data() + (size_t) s * H);
            }
        }

        if ((frame_index % 100) == 0) {
            fprintf(stderr, "[MM3-AR] Frame %d/%d\n", frame_index, max_frames);
        }
    }

    n_frames.assign((size_t) N, 0);
    for (int i = 0; i < N; i++) {
        n_frames[(size_t) i] = frames[(size_t) i];
        fprintf(stderr, "[MM3-AR] Song %d: %d frames (%.1fs of music)\n", i, n_frames[(size_t) i],
                (float) n_frames[(size_t) i] / MM3_FRAME_RATE);
        if (n_frames[(size_t) i] == 0) {
            fprintf(stderr, "[MM3-AR] ERROR: song %d generated zero audio frames\n", i);
            return MM3_PIPELINE_FAILED;
        }
    }
    if (p->dumper.enabled && hiddens_out) {
        int shape[3] = { n_frames[0], 8, H };
        debug_dump(&p->dumper, "frame_hiddens", (*hiddens_out)[0].data(), shape, 3);
    }
    {
        int total = 0;
        for (int i = 0; i < N; i++) {
            total += n_frames[(size_t) i];
        }
        fprintf(stderr, "[MM3-AR] %d frames total, %.1f s (%.1f ms/frame)\n", total, ar_timer.ms() / 1000.0,
                total > 0 ? ar_timer.ms() / total : 0.0);
    }
    return MM3_PIPELINE_OK;
}

// KV capacity for a run: prompt + frames + 1, exactly what the reference
// allocates. The nominal 10240 context is not a limit here, RoPE
// extrapolates past it. params.max_seq, when set, caps the result.
//
// Rounded up to a multiple of KV_QUANTUM because the capacity is part of
// the store key: an exact per-prompt value would make every request with a
// different prompt length a distinct module and reload 8.9 GB of weights,
// and under --keep-loaded it would accumulate one resident LM per distinct
// length instead of reusing the one already there. Rounding costs at most
// KV_QUANTUM * 144 KB of cache (144 MB) and makes repeat requests hit.
static int mm3_kv_capacity(const MM3Pipeline * p, size_t prompt_tokens, int max_frames) {
    const int KV_QUANTUM = 1024;

    int need = (int) prompt_tokens + max_frames + 1;
    need     = ((need + KV_QUANTUM - 1) / KV_QUANTUM) * KV_QUANTUM;
    if (p->params.max_seq > 0 && need > p->params.max_seq) {
        fprintf(stderr, "[MM3] KV capacity capped at %d by --max-seq (wanted %d)\n", p->params.max_seq, need);
        need = p->params.max_seq;
    }
    return need;
}

MM3PipelineStatus mm3_pipeline_generate(MM3Pipeline *                     p,
                                        const AceRequest &                req,
                                        std::atomic<bool> *               cancel,
                                        std::vector<std::vector<float>> & tracks_out,
                                        std::vector<std::string> *        codes_out) {
    Timer total_timer;

    int N = req.lm_batch_size < 1 ? 1 : req.lm_batch_size;
    if (N > p->params.max_batch) {
        fprintf(stderr, "[MM3] FATAL: lm_batch_size %d exceeds the batch limit %d\n", N, p->params.max_batch);
        return MM3_PIPELINE_FAILED;
    }

    const MM3Recipe recipe = mm3_resolve_recipe(req);

    // The tokenizer is CPU resident and comes from the LM GGUF metadata, so
    // it loads before any GPU module: the prompt length sizes the KV cache.
    BPETokenizer * tok = store_mm3_bpe(p->store, p->wanted.lm.c_str());
    if (!tok) {
        return MM3_PIPELINE_FAILED;
    }
    std::vector<int> cond_ids = mm3_prompt_ids(tok, req);
    if (cond_ids.empty()) {
        return MM3_PIPELINE_FAILED;
    }
    fprintf(stderr, "[MM3-Prompt] %zu tokens\n", cond_ids.size());

    std::vector<int> replay_codes;
    int              frames_hint = recipe.max_frames;
    if (!req.audio_codes.empty()) {
        replay_codes = mm3_codes_parse(req.audio_codes);
        if (replay_codes.empty()) {
            return MM3_PIPELINE_FAILED;
        }
        frames_hint = (int) (replay_codes.size() / 8);
        // Replay is teacher forced on one code stream: it never samples, so
        // it needs a single conditional KV set. Collapsing N here keeps the
        // store from sizing 2N caches it would not use.
        if (N > 1) {
            fprintf(stderr, "[MM3-AR] audio_codes provided: lm_batch_size ignored\n");
            N = 1;
        }
    }

    std::vector<std::vector<float>> frame_hiddens;
    std::vector<int>                n_frames;
    int                             H = 0;

    // AR group scope: the RAII handles release { MM3_LM, MM3_DEPTH } before
    // the synthesis group is required, so under STRICT the LM weights and
    // the DiT weights never coexist
    {
        mm3_select_backend(p->params.ar_backend, "AR");
        MM3LM * lm = mm3_require_lm(p, mm3_kv_capacity(p, cond_ids.size(), frames_hint), 2 * N);
        if (!lm) {
            return MM3_PIPELINE_FAILED;
        }
        ModelHandle lm_h(p->store, lm);
        MM3Depth *  depth = mm3_require_depth(p);
        if (!depth) {
            return MM3_PIPELINE_FAILED;
        }
        ModelHandle depth_h(p->store, depth);
        H = lm->cfg.hidden_size;

        if (!replay_codes.empty()) {
            // Teacher-forced replay: the codes replace the sampling
            frame_hiddens.resize(1);
            MM3PipelineStatus st = mm3_replay_stage(p, lm, depth, cond_ids, replay_codes, cancel, frame_hiddens[0]);
            if (st != MM3_PIPELINE_OK) {
                return st;
            }
            n_frames.assign(1, (int) (frame_hiddens[0].size() / (8 * (size_t) H)));
            if (p->dumper.enabled) {
                int shape[3] = { n_frames[0], 8, H };
                debug_dump(&p->dumper, "frame_hiddens", frame_hiddens[0].data(), shape, 3);
            }
            if (codes_out) {
                codes_out->assign(1, req.audio_codes);
            }
        } else {
            frame_hiddens.resize((size_t) N);
            std::vector<std::vector<int>> codes((size_t) N);
            MM3PipelineStatus             st =
                mm3_ar_stage(p, lm, depth, cond_ids, recipe, req, cancel, N, &frame_hiddens, codes, n_frames);
            if (st != MM3_PIPELINE_OK) {
                return st;
            }
            if (codes_out) {
                codes_out->resize((size_t) N);
                for (int i = 0; i < N; i++) {
                    (*codes_out)[(size_t) i] = mm3_codes_serialize(codes[(size_t) i]);
                }
            }
        }
    }

    // Synthesis group: { MM3_DIT, MM3_VAE } interleave per window and per
    // song, so both handles live across the whole song loop. The condition
    // encoder rides inside the DiT module.
    mm3_select_backend(p->params.synth_backend, "Synth");
    MM3DiT * dit = mm3_require_dit(p);
    if (!dit) {
        return MM3_PIPELINE_FAILED;
    }
    ModelHandle dit_h(p->store, dit);
    MM3VAE *    vae = mm3_require_vae(p);
    if (!vae) {
        return MM3_PIPELINE_FAILED;
    }
    ModelHandle vae_h(p->store, vae);

    // Windowed flow matching and decode: per song, M noise variations
    // batched in the DiT on the shared condition track (seeds seed + j)
    int M = req.synth_batch_size < 1 ? 1 : (req.synth_batch_size > 9 ? 9 : req.synth_batch_size);
    tracks_out.assign((size_t) N * M, {});
    if (M > 1) {
        fprintf(stderr, "[MM3-Synth] %d variations per song\n", M);
    }
    for (int song = 0; song < N; song++) {
        if (N > 1) {
            fprintf(stderr, "[MM3-Synth] Song %d/%d\n", song + 1, N);
        }
        Timer            synth_timer;
        std::vector<int> chunk_starts;
        if (n_frames[(size_t) song] <= MM3_CHUNK_FRAMES) {
            chunk_starts.push_back(0);
        } else {
            for (int s = 0; s < n_frames[(size_t) song] - MM3_CHUNK_HOP; s += MM3_CHUNK_HOP) {
                chunk_starts.push_back(s);
            }
        }

        // Ascending sigma schedule: linspace(1, 1/steps) inverted, final 1.0
        const int          steps = recipe.steps;
        std::vector<float> sig((size_t) steps + 1);
        for (int i = 0; i < steps; i++) {
            float lin       = steps > 1 ? 1.0f + (1.0f / (float) steps - 1.0f) * (float) i / (float) (steps - 1) : 1.0f;
            sig[(size_t) i] = 1.0f - lin;
        }
        sig[(size_t) steps] = 1.0f;

        // Per-variation window state; the condition track is shared
        std::vector<std::vector<std::vector<float>>> latent_chunks((size_t) M);
        std::vector<int>                             chunk_lat(chunk_starts.size());
        std::vector<std::vector<float>>              prev_latent((size_t) M), noise_prompt((size_t) M), xt((size_t) M);
        std::vector<int64_t>                         noise_index((size_t) M, 0);
        std::vector<float>                           prev_condition, cond_track;
        std::vector<float>                           v_cond, v_uncond;

        for (size_t k = 0; k < chunk_starts.size(); k++) {
            Timer window_timer;
            int   start = chunk_starts[k];
            int   end =
                start + MM3_CHUNK_FRAMES < n_frames[(size_t) song] ? start + MM3_CHUNK_FRAMES : n_frames[(size_t) song];

            std::vector<float> window(frame_hiddens[(size_t) song].begin() + (size_t) start * 8 * H,
                                      frame_hiddens[(size_t) song].begin() + (size_t) end * 8 * H);
            int                T_lat = 0;
            if (!dit->encode_condition(window, end - start, cond_track, T_lat)) {
                return MM3_PIPELINE_FAILED;
            }
            chunk_lat[k] = T_lat;

            int overlap = 0;
            if (!prev_latent[0].empty()) {
                overlap = (int) (prev_latent[0].size() / 128) < T_lat ? (int) (prev_latent[0].size() / 128) : T_lat;
                std::copy(prev_condition.begin(), prev_condition.begin() + (size_t) overlap * 2048, cond_track.begin());
            }

            // Initial noise per variation, Philox stream continued across
            // windows, one seed per variation
            for (int j = 0; j < M; j++) {
                xt[(size_t) j].resize((size_t) T_lat * 128);
                for (float & x : xt[(size_t) j]) {
                    float vals[4];
                    philox_normal4((uint64_t) (req.seed + j), noise_index[(size_t) j]++, 0, vals);
                    x = vals[0];
                }
                noise_prompt[(size_t) j].assign(xt[(size_t) j].begin(),
                                                xt[(size_t) j].begin() + (size_t) overlap * 128);
            }

            std::vector<float> zeros_track(cond_track.size(), 0.0f);
            size_t             lat_sz = xt[0].size();
            std::vector<float> v_cfg;
            std::vector<float> xt2, cond2, v2;
            v_cond.resize(lat_sz);
            v_uncond.resize(lat_sz);
            if (p->params.use_batch_cfg) {
                // All variations and both CFG branches in one batch-2M
                // forward: [cond track x M, zeros x M]
                cond2.resize(cond_track.size() * 2 * (size_t) M);
                for (int j = 0; j < M; j++) {
                    memcpy(cond2.data() + (size_t) j * cond_track.size(), cond_track.data(),
                           cond_track.size() * sizeof(float));
                }
                memset(cond2.data() + (size_t) M * cond_track.size(), 0,
                       (size_t) M * cond_track.size() * sizeof(float));
                xt2.resize(lat_sz * 2 * (size_t) M);
                v2.resize(lat_sz * 2 * (size_t) M);
            }

            bool dump_win = p->dumper.enabled && song == 0 && k == 0;
            if (dump_win) {
                debug_dump_2d(&p->dumper, "noise", xt[0].data(), T_lat, 128);
                v_cfg.resize(lat_sz);
            }

            for (int i = 0; i < steps; i++) {
                if (mm3_is_cancelled(cancel)) {
                    return MM3_PIPELINE_CANCELLED;
                }
                float t = sig[(size_t) i];
                for (int j = 0; j < M; j++) {
                    for (int e = 0; e < overlap * 128; e++) {
                        xt[(size_t) j][(size_t) e] =
                            (1.0f - (1.0f - 1e-6f) * t) * noise_prompt[(size_t) j][(size_t) e] +
                            t * prev_latent[(size_t) j][(size_t) e];
                    }
                }
                float dt = sig[(size_t) i + 1] - sig[(size_t) i];
                if (p->params.use_batch_cfg) {
                    for (int j = 0; j < M; j++) {
                        memcpy(xt2.data() + (size_t) j * lat_sz, xt[(size_t) j].data(), lat_sz * sizeof(float));
                        memcpy(xt2.data() + (size_t) (M + j) * lat_sz, xt[(size_t) j].data(), lat_sz * sizeof(float));
                    }
                    if (!dit->forward(xt2.data(), cond2.data(), T_lat, 2 * M, t, v2.data())) {
                        return MM3_PIPELINE_FAILED;
                    }
                    if (dump_win && i == 0) {
                        dit->dump_named(&p->dumper);
                    }
                    for (int j = 0; j < M; j++) {
                        const float * vc = v2.data() + (size_t) j * lat_sz;
                        const float * vu = v2.data() + (size_t) (M + j) * lat_sz;
                        for (size_t e = 0; e < lat_sz; e++) {
                            float v = vu[e] + (vc[e] - vu[e]) * recipe.dit_cfg;
                            if (dump_win && j == 0) {
                                v_cfg[e] = v;
                            }
                            xt[(size_t) j][e] += dt * v;
                        }
                        if (dump_win && j == 0) {
                            memcpy(v_cond.data(), vc, lat_sz * sizeof(float));
                            memcpy(v_uncond.data(), vu, lat_sz * sizeof(float));
                        }
                    }
                } else {
                    for (int j = 0; j < M; j++) {
                        if (!dit->forward(xt[(size_t) j].data(), cond_track.data(), T_lat, 1, t, v_cond.data())) {
                            return MM3_PIPELINE_FAILED;
                        }
                        if (dump_win && j == 0 && i == 0) {
                            dit->dump_named(&p->dumper);
                        }
                        if (!dit->forward(xt[(size_t) j].data(), zeros_track.data(), T_lat, 1, t, v_uncond.data())) {
                            return MM3_PIPELINE_FAILED;
                        }
                        for (size_t e = 0; e < lat_sz; e++) {
                            float v = v_uncond[e] + (v_cond[e] - v_uncond[e]) * recipe.dit_cfg;
                            if (dump_win && j == 0) {
                                v_cfg[e] = v;
                            }
                            xt[(size_t) j][e] += dt * v;
                        }
                    }
                }
                if (dump_win) {
                    char name[64];
                    snprintf(name, sizeof(name), "dit_step%d_vt_cond", i);
                    debug_dump_2d(&p->dumper, name, v_cond.data(), T_lat, 128);
                    snprintf(name, sizeof(name), "dit_step%d_vt_uncond", i);
                    debug_dump_2d(&p->dumper, name, v_uncond.data(), T_lat, 128);
                    snprintf(name, sizeof(name), "dit_step%d_vt", i);
                    debug_dump_2d(&p->dumper, name, v_cfg.data(), T_lat, 128);
                    snprintf(name, sizeof(name), "dit_step%d_xt", i);
                    debug_dump_2d(&p->dumper, name, xt[0].data(), T_lat, 128);
                }
            }
            if (dump_win) {
                debug_dump_2d(&p->dumper, "dit_x0", xt[0].data(), T_lat, 128);
            }
            // The overlap region is owned by the previous window: restore it
            // verbatim so the stitch has no seam.
            for (int j = 0; j < M; j++) {
                for (int e = 0; e < overlap * 128; e++) {
                    xt[(size_t) j][(size_t) e] = prev_latent[(size_t) j][(size_t) e];
                }
            }

            int os = T_lat - 2 * MM3_OVERLAP_LATENT > 0 ? T_lat - 2 * MM3_OVERLAP_LATENT : 0;
            int oe = T_lat - MM3_OVERLAP_LATENT > os ? T_lat - MM3_OVERLAP_LATENT : os;
            for (int j = 0; j < M; j++) {
                prev_latent[(size_t) j].assign(xt[(size_t) j].begin() + (size_t) os * 128,
                                               xt[(size_t) j].begin() + (size_t) oe * 128);
            }
            prev_condition.assign(cond_track.begin() + (size_t) os * 2048, cond_track.begin() + (size_t) oe * 2048);

            if (p->dumper.enabled && song == 0) {
                char name[64];
                snprintf(name, sizeof(name), "window%zu_cond", k);
                debug_dump_2d(&p->dumper, name, cond_track.data(), T_lat, 2048);
                snprintf(name, sizeof(name), "window%zu_latent", k);
                debug_dump_2d(&p->dumper, name, xt[0].data(), T_lat, 128);
            }
            for (int j = 0; j < M; j++) {
                latent_chunks[(size_t) j].push_back(xt[(size_t) j]);
            }
            fprintf(stderr, "[MM3-DiT] Window %zu/%zu: T=%d, %d steps, %.0f ms (%.1f ms/step)\n", k + 1,
                    chunk_starts.size(), T_lat, steps, window_timer.ms(), window_timer.ms() / steps);
        }
        fprintf(stderr, "[MM3-DiT] CFG=%.2f, %zu windows, %.1f s\n", (double) recipe.dit_cfg, chunk_starts.size(),
                synth_timer.ms() / 1000.0);

        // Decode, crop, stitch each variation
        Timer vae_timer;
        int   T_last = 0;
        for (int j = 0; j < M; j++) {
            std::vector<float> & audio_out = tracks_out[(size_t) song * M + j];
            for (size_t k = 0; k < latent_chunks[(size_t) j].size(); k++) {
                if (mm3_is_cancelled(cancel)) {
                    return MM3_PIPELINE_CANCELLED;
                }
                int                T_lat = chunk_lat[k];
                std::vector<float> chan((size_t) 128 * T_lat);
                for (int t = 0; t < T_lat; t++) {
                    for (int c = 0; c < 128; c++) {
                        chan[(size_t) c * T_lat + t] = latent_chunks[(size_t) j][k][(size_t) t * 128 + c];
                    }
                }
                std::vector<float> wav;
                if (!vae->decode(chan, T_lat, wav)) {
                    return MM3_PIPELINE_FAILED;
                }
                int left  = (k == 0) ? 0 : MM3_CROP_LEFT * MM3VAE::HOP;
                int right = (k + 1 == latent_chunks[(size_t) j].size()) ? 0 : MM3_CROP_RIGHT * MM3VAE::HOP;
                audio_out.insert(audio_out.end(), wav.begin() + (size_t) left * 2, wav.end() - (size_t) right * 2);
            }
            if (p->dumper.enabled && song == 0 && j == 0) {
                debug_dump_2d(&p->dumper, "vae_audio", audio_out.data(), (int) (audio_out.size() / 2), 2);
            }

            // Interleaved [T, 2] -> planar [L:T][R:T] for the output stage
            int T = (int) (audio_out.size() / 2);
            {
                std::vector<float> planar((size_t) T * 2);
                for (int t = 0; t < T; t++) {
                    planar[(size_t) t]              = audio_out[(size_t) t * 2];
                    planar[(size_t) T + (size_t) t] = audio_out[(size_t) t * 2 + 1];
                }
                audio_out.swap(planar);
            }
            T_last = T;
        }
        fprintf(stderr, "[MM3-VAE] Decode: %zu windows -> %.1fs of audio, %.0f ms\n", chunk_starts.size(),
                (float) T_last / (float) MM3_SAMPLE_RATE, vae_timer.ms());
    }
    fprintf(stderr, "[MM3] Done, %.1f s total\n", total_timer.ms() / 1000.0);
    return MM3_PIPELINE_OK;
}

MM3PipelineStatus mm3_pipeline_lm_generate(MM3Pipeline *              p,
                                           const AceRequest &         req,
                                           std::atomic<bool> *        cancel,
                                           std::vector<std::string> & codes_out) {
    Timer total_timer;

    int N = req.lm_batch_size < 1 ? 1 : req.lm_batch_size;
    if (N > p->params.max_batch) {
        fprintf(stderr, "[MM3] FATAL: lm_batch_size %d exceeds the batch limit %d\n", N, p->params.max_batch);
        return MM3_PIPELINE_FAILED;
    }

    const MM3Recipe recipe = mm3_resolve_recipe(req);

    BPETokenizer * tok = store_mm3_bpe(p->store, p->wanted.lm.c_str());
    if (!tok) {
        return MM3_PIPELINE_FAILED;
    }
    std::vector<int> cond_ids = mm3_prompt_ids(tok, req);
    if (cond_ids.empty()) {
        return MM3_PIPELINE_FAILED;
    }
    fprintf(stderr, "[MM3-Prompt] %zu tokens\n", cond_ids.size());

    mm3_select_backend(p->params.ar_backend, "AR");
    MM3LM * lm = mm3_require_lm(p, mm3_kv_capacity(p, cond_ids.size(), recipe.max_frames), 2 * N);
    if (!lm) {
        return MM3_PIPELINE_FAILED;
    }
    ModelHandle lm_h(p->store, lm);
    MM3Depth *  depth = mm3_require_depth(p);
    if (!depth) {
        return MM3_PIPELINE_FAILED;
    }
    ModelHandle depth_h(p->store, depth);

    std::vector<std::vector<int>> codes((size_t) N);
    std::vector<int>              n_frames;
    MM3PipelineStatus st = mm3_ar_stage(p, lm, depth, cond_ids, recipe, req, cancel, N, nullptr, codes, n_frames);
    if (st != MM3_PIPELINE_OK) {
        return st;
    }
    codes_out.resize((size_t) N);
    for (int i = 0; i < N; i++) {
        codes_out[(size_t) i] = mm3_codes_serialize(codes[(size_t) i]);
    }
    fprintf(stderr, "[MM3] Done, %.1f s total\n", total_timer.ms() / 1000.0);
    return MM3_PIPELINE_OK;
}
