#pragma once
// mm3-lm.h: MiniMax Music 3 global LM (Qwen3 8B) with KV cache (GGML)
//
// 36 layers, H = 4096, GQA 32/8 with head_dim 128, QK-RMSNorm, SwiGLU
// 12288, RoPE theta 1e6, RMSNorm eps 1e-6, full (non sliding) attention.
// The ComfyUI packaging ships the projections pre-fused, so qkv_proj
// [4096, 6144] splits by rows into q(4096) || k(1024) || v(1024) and
// gate_up_proj [4096, 24576] into gate || up: both load straight in, no
// gf_load_qkv_fused pass.
//
// Pruned embedding layout. The checkpoint drops the 200k joint table and
// keeps three range specific ones, so there is no single lookup:
//   prompt ids (< 151675)  -> model.embed_tokens_prefill  [4096, 151675]
//   semantic code c0       -> model.embed_tokens_audio    [4096, 16384]
//   frame feedback         -> (emb_audio(c0) + sum_{j=1..7}
//                              audio_extra[(j-1)*1024 + c_j]) * 8^-0.5
// The feedback vector is assembled on the host and enters the graph as
// input_embeds, so the decode path never looks a token id up.
//
// Pruned head. model.lm_head_pruned [4096, 16385] emits 16384 codes plus
// one stop slot: index 0 is the stop token and a sampled index i > 0 is
// code i - 1. There is no vocab mask on this path (the unpruned one would
// mask everything but [151675, 151675+16384) plus <|audio_end|>).
//
// KV cache. Sized by the caller as prompt + frames + 1 rather than the
// nominal 10240 context: the reference does the same and RoPE extrapolates
// past the nominal window, which a 5000 token prompt plus 9000 frames
// needs.

#include "backend.h"
#include "gguf-weights.h"
#include "graph-arena.h"
#include "static-graph.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define MM3LM_MAX_KV_SETS 32  // song batch N * 2 (cond + uncond CFG)
#define MM3LM_MAX_LAYERS  64
#define MM3LM_GRAPH_NODES 16384

// Fixed hyperparameters: the checkpoint is one model and the GGUF carries
// no hparam KVs, so there is nothing to read.
struct MM3LMConfig {
    int   vocab_size        = 151675;  // prompt embedding rows (AUDIO_CODE_OFFSET)
    int   hidden_size       = 4096;
    int   intermediate_size = 12288;
    int   n_heads           = 32;
    int   n_kv_heads        = 8;
    int   head_dim          = 128;
    int   n_layers          = 36;
    float rope_theta        = 1000000.0f;
    float rms_norm_eps      = 1e-6f;
    int   max_seq_len       = 10240;  // KV capacity, set by the caller
    int   head_vocab        = 16385;  // lm_head_pruned: stop + 16384 codes
    int   semantic_vocab    = 16384;
};

struct MM3LMLayer {
    struct ggml_tensor * input_layernorm;      // [H]
    struct ggml_tensor * post_attn_layernorm;  // [H]
    struct ggml_tensor * qkv;                  // [H, (Nh + 2*Nkv)*D] fused
    struct ggml_tensor * o_proj;               // [Nh*D, H]
    struct ggml_tensor * q_norm;               // [D]
    struct ggml_tensor * k_norm;               // [D]
    struct ggml_tensor * gate_up;              // [H, 2*FFN] fused
    struct ggml_tensor * down_proj;            // [FFN, H]
};

// Static batched decode graph, replayed with a fresh input upload per
// step: n_kv_pad advances every 256 steps and the batch base set holds
// within a generation. A prefill forward clobbers the shared sched
// allocation, so it invalidates this cache.
struct MM3LMGraphCache {
    bool                  built        = false;
    int                   key_n_kv_pad = 0;
    int                   key_N        = 0;
    int                   key_s0       = 0;
    struct ggml_cgraph *  gf           = nullptr;
    struct ggml_tensor *  embeds_t     = nullptr;
    struct ggml_tensor *  positions    = nullptr;
    struct ggml_tensor *  kv_rows      = nullptr;
    struct ggml_tensor *  attn_mask    = nullptr;
    struct ggml_tensor *  lgt          = nullptr;
    struct ggml_tensor *  hidden_out   = nullptr;
    StaticGraph           graph;
    std::vector<int>      pos_data;
    std::vector<int64_t>  rows_data;
    std::vector<uint16_t> mask_data;
};

struct MM3LM {
    MM3LMConfig cfg;

    struct ggml_tensor * embed_prefill;  // [H, 151675] prompt tokens
    struct ggml_tensor * embed_audio;    // [H, 16384]  semantic codes
    struct ggml_tensor * lm_head;        // [H, 16385]  stop + codes
    struct ggml_tensor * final_norm;     // [H]
    MM3LMLayer           layers[MM3LM_MAX_LAYERS];

    WeightCtx            wctx           = {};
    ggml_backend_t       backend        = nullptr;
    ggml_backend_t       cpu_backend    = nullptr;
    ggml_backend_sched_t sched          = nullptr;
    bool                 use_flash_attn = false;
    bool                 clamp_fp16     = false;

    // KV cache: 4D per layer [D, max_seq, Nkv, n_sets] plus 3D per-set views
    struct ggml_context * kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    struct ggml_tensor *  kv_k4[MM3LM_MAX_LAYERS];
    struct ggml_tensor *  kv_v4[MM3LM_MAX_LAYERS];
    struct ggml_tensor *  kv_k[MM3LM_MAX_KV_SETS][MM3LM_MAX_LAYERS];
    struct ggml_tensor *  kv_v[MM3LM_MAX_KV_SETS][MM3LM_MAX_LAYERS];
    int                   kv_pos[MM3LM_MAX_KV_SETS];
    int                   n_kv_sets = 0;

    // Persistent graph arenas, one per shape class: stable node addresses
    // across rebuilds keep the backend CUDA graph cache hot.
    GraphArena arena_prefill;
    GraphArena arena_decode;
    GraphArena arena_batch;

    MM3LMGraphCache batch_graph;
};

static void mm3_lm_init_backend(MM3LM * m) {
    BackendPair bp    = backend_init("MM3-LM");
    m->backend        = bp.backend;
    m->cpu_backend    = bp.cpu_backend;
    m->sched          = backend_sched_new(bp, 8192);
    m->use_flash_attn = bp.has_gpu;
    m->clamp_fp16     = false;
}

static void mm3_lm_alloc_kv_cache(MM3LM * m, int n_sets) {
    const MM3LMConfig & c   = m->cfg;
    int                 D   = c.head_dim;
    int                 Nkv = c.n_kv_heads;
    int                 L   = c.n_layers;
    int                 S   = c.max_seq_len;

    m->n_kv_sets = n_sets;

    int                     n_tensors = L * 2 + n_sets * L * 2;
    size_t                  ctx_size  = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params gp        = { ctx_size, NULL, true };
    m->kv_ctx                         = ggml_init(gp);

    for (int l = 0; l < L; l++) {
        m->kv_k4[l] = ggml_new_tensor_4d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv, n_sets);
        m->kv_v4[l] = ggml_new_tensor_4d(m->kv_ctx, GGML_TYPE_F16, D, S, Nkv, n_sets);
        char name[64];
        snprintf(name, sizeof(name), "kv_k4_%d", l);
        ggml_set_name(m->kv_k4[l], name);
        snprintf(name, sizeof(name), "kv_v4_%d", l);
        ggml_set_name(m->kv_v4[l], name);

        for (int s = 0; s < n_sets; s++) {
            size_t off = (size_t) s * D * S * Nkv * ggml_type_size(GGML_TYPE_F16);
            m->kv_k[s][l] =
                ggml_view_3d(m->kv_ctx, m->kv_k4[l], D, S, Nkv, m->kv_k4[l]->nb[1], m->kv_k4[l]->nb[2], off);
            m->kv_v[s][l] =
                ggml_view_3d(m->kv_ctx, m->kv_v4[l], D, S, Nkv, m->kv_v4[l]->nb[1], m->kv_v4[l]->nb[2], off);
        }
    }
    for (int s = 0; s < n_sets; s++) {
        m->kv_pos[s] = 0;
    }

    m->kv_buf = ggml_backend_alloc_ctx_tensors(m->kv_ctx, m->backend);
    if (!m->kv_buf) {
        fprintf(stderr, "[MM3-LM-KV] FATAL: failed to allocate KV cache\n");
        exit(1);
    }
    // The attention window is padded past kv_pos, so the masked tail must
    // read finite values, never uninitialized F16 bit patterns.
    ggml_backend_buffer_clear(m->kv_buf, 0);

    size_t kv_bytes = (size_t) n_sets * L * 2 * D * S * Nkv * ggml_type_size(GGML_TYPE_F16);
    fprintf(stderr, "[MM3-LM-KV] %d sets x %d layers, seq %d, %.1f MB\n", n_sets, L, S,
            (float) kv_bytes / (1024 * 1024));
}

static void mm3_lm_reset_kv(MM3LM * m, int kv_set) {
    m->kv_pos[kv_set] = 0;
}

static void mm3_lm_copy_kv(MM3LM * m, int src, int dst) {
    for (int l = 0; l < m->cfg.n_layers; l++) {
        ggml_backend_tensor_copy(m->kv_k[src][l], m->kv_k[dst][l]);
        ggml_backend_tensor_copy(m->kv_v[src][l], m->kv_v[dst][l]);
    }
    m->kv_pos[dst] = m->kv_pos[src];
}

// max_seq_len must cover prompt + frames + 1; n_kv_sets is 2 * songs.
static bool mm3_lm_load(MM3LM * m, const char * gguf_path, int max_seq_len, int n_kv_sets) {
    *m = {};

    mm3_lm_init_backend(m);

    GGUFModel gf;
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[MM3-LM] FATAL: cannot load %s\n", gguf_path);
        return false;
    }
    if (max_seq_len > 0) {
        m->cfg.max_seq_len = max_seq_len;
    }
    const MM3LMConfig & c = m->cfg;

    // embed_prefill + embed_audio + lm_head + final_norm, then 8 per layer:
    // 2 norms, fused qkv, o_proj, q_norm, k_norm, fused gate_up, down_proj
    int n_tensors = 4 + c.n_layers * 8;
    wctx_init(&m->wctx, n_tensors);

    m->embed_prefill = gf_load_tensor(&m->wctx, gf, "model.embed_tokens_prefill.weight");
    m->embed_audio   = gf_load_tensor(&m->wctx, gf, "model.embed_tokens_audio.weight");
    m->lm_head       = gf_load_tensor(&m->wctx, gf, "model.lm_head_pruned.weight");
    m->final_norm    = gf_load_tensor_f32(&m->wctx, gf, "model.norm.weight");

    for (int i = 0; i < c.n_layers; i++) {
        MM3LMLayer & ly        = m->layers[i];
        std::string  pfx       = "model.layers." + std::to_string(i) + ".";
        ly.input_layernorm     = gf_load_tensor_f32(&m->wctx, gf, pfx + "input_layernorm.weight");
        ly.post_attn_layernorm = gf_load_tensor_f32(&m->wctx, gf, pfx + "post_attention_layernorm.weight");
        ly.qkv                 = gf_load_tensor(&m->wctx, gf, pfx + "self_attn.qkv_proj.weight");
        ly.o_proj              = gf_load_tensor(&m->wctx, gf, pfx + "self_attn.o_proj.weight");
        ly.q_norm              = gf_load_tensor_f32(&m->wctx, gf, pfx + "self_attn.q_norm.weight");
        ly.k_norm              = gf_load_tensor_f32(&m->wctx, gf, pfx + "self_attn.k_norm.weight");
        ly.gate_up             = gf_load_tensor(&m->wctx, gf, pfx + "mlp.gate_up_proj.weight");
        ly.down_proj           = gf_load_tensor(&m->wctx, gf, pfx + "mlp.down_proj.weight");
    }

    // Shape sanity: a silently different packaging would produce garbage
    // audio instead of an error.
    if (m->embed_prefill->ne[1] != c.vocab_size || m->embed_audio->ne[1] != c.semantic_vocab ||
        m->lm_head->ne[1] != c.head_vocab) {
        fprintf(stderr, "[MM3-LM] FATAL: unexpected embedding shapes (%lld / %lld / %lld)\n",
                (long long) m->embed_prefill->ne[1], (long long) m->embed_audio->ne[1], (long long) m->lm_head->ne[1]);
        gf_close(&gf);
        return false;
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    mm3_lm_alloc_kv_cache(m, n_kv_sets > 0 ? n_kv_sets : 1);

    if (!graph_arena_init(&m->arena_prefill, MM3LM_GRAPH_NODES) ||
        !graph_arena_init(&m->arena_decode, MM3LM_GRAPH_NODES) ||
        !graph_arena_init(&m->arena_batch, MM3LM_GRAPH_NODES)) {
        return false;
    }

    fprintf(stderr, "[MM3-LM] Loaded: %dL H=%d GQA %d/%d D=%d, head %d, flash_attn=%d\n", c.n_layers, c.hidden_size,
            c.n_heads, c.n_kv_heads, c.head_dim, c.head_vocab, m->use_flash_attn);
    return true;
}

// Graph helpers

static struct ggml_tensor * mm3_lm_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

static struct ggml_tensor * mm3_lm_rms_norm(struct ggml_context * ctx,
                                            struct ggml_tensor *  x,
                                            struct ggml_tensor *  w,
                                            float                 eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), mm3_lm_f32(ctx, w));
}

// F32 manual attention fallback. Works for 3D [D, S, X] and 4D [D, S, X, N],
// returning the flash_attn_ext layout (dims 1 and 2 swapped vs input).
static struct ggml_tensor * mm3_lm_attn_f32(struct ggml_context * ctx,
                                            struct ggml_tensor *  q,
                                            struct ggml_tensor *  k,
                                            struct ggml_tensor *  v,
                                            struct ggml_tensor *  mask,
                                            float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// gate_up is [gate(FFN) || up(FFN)]: ggml_swiglu on the fused tensor is
// silu(first half) * second half, exactly `down(silu(gate) * up)`.
static struct ggml_tensor * mm3_lm_build_mlp(struct ggml_context * ctx, MM3LMLayer * ly, struct ggml_tensor * x) {
    struct ggml_tensor * gu = ggml_mul_mat(ctx, ly->gate_up, x);
    return ggml_mul_mat(ctx, ly->down_proj, ggml_swiglu(ctx, gu));
}

// Self-attention with KV cache write + read. The T fresh K/V rows write at
// the positions carried by kv_rows via set_rows, so the graph topology is
// identical across decode steps and a captured CUDA graph replays as is.
static struct ggml_tensor * mm3_lm_build_attn(struct ggml_context * ctx,
                                              struct ggml_cgraph *  gf,
                                              const MM3LMConfig &   c,
                                              MM3LMLayer *          ly,
                                              struct ggml_tensor *  x,
                                              struct ggml_tensor *  positions,
                                              struct ggml_tensor *  mask,
                                              struct ggml_tensor *  kv_rows,
                                              struct ggml_tensor *  cache_k,
                                              struct ggml_tensor *  cache_v,
                                              int                   n_kv_pad,
                                              int                   n_tokens,
                                              bool                  use_flash_attn,
                                              bool                  clamp_fp16) {
    int D   = c.head_dim;
    int Nh  = c.n_heads;
    int Nkv = c.n_kv_heads;
    int S   = n_tokens;

    int q_dim  = Nh * D;
    int kv_dim = Nkv * D;

    struct ggml_tensor * qkv = ggml_mul_mat(ctx, ly->qkv, x);
    struct ggml_tensor * q   = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim, S, qkv->nb[1], 0));
    struct ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t) q_dim * qkv->nb[0]));
    struct ggml_tensor * v =
        ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, S, qkv->nb[1], (size_t) (q_dim + kv_dim) * qkv->nb[0]));

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);

    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, mm3_lm_f32(ctx, ly->q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, mm3_lm_f32(ctx, ly->k_norm));

    q = ggml_rope_ext(ctx, q, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    if (clamp_fp16) {
        v = ggml_clamp(ctx, v, -65504.0f, 65504.0f);
    }

    size_t nb1 = (size_t) D * ggml_type_size(GGML_TYPE_F16);
    size_t nb2 = (size_t) D * c.max_seq_len * ggml_type_size(GGML_TYPE_F16);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, k, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, v, kv_rows));

    struct ggml_tensor * k_full = ggml_view_3d(ctx, cache_k, D, n_kv_pad, Nkv, nb1, nb2, 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, cache_v, D, n_kv_pad, Nkv, nb1, nb2, 0);

    float                scale = 1.0f / sqrtf((float) D);
    struct ggml_tensor * attn  = use_flash_attn ? ggml_flash_attn_ext(ctx, q, k_full, v_full, mask, scale, 0.0f, 0.0f) :
                                                  mm3_lm_attn_f32(ctx, q, k_full, v_full, mask, scale);
    if (use_flash_attn) {
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    }

    attn = ggml_reshape_2d(ctx, attn, Nh * D, S);
    return ggml_mul_mat(ctx, ly->o_proj, attn);
}

// Prefill / single stream forward.
// token_ids[n_tokens] are prompt ids resolved through embed_tokens_prefill;
// input_embeds replaces the lookup with raw [n_tokens, H] embeddings.
// logits receives head_vocab floats for the last token. out_hidden gets the
// last final-norm hidden state, out_hidden_all every position.
static void mm3_lm_forward(MM3LM *       m,
                           const int *   token_ids,
                           int           n_tokens,
                           int           kv_set,
                           float *       logits,
                           const float * input_embeds   = nullptr,
                           float *       out_hidden     = nullptr,
                           float *       out_hidden_all = nullptr) {
    if (m->batch_graph.graph.sched_allocated) {
        static_graph_release(&m->batch_graph.graph, m->sched);
        m->batch_graph.built = false;
    }

    const MM3LMConfig & c      = m->cfg;
    int                 H      = c.hidden_size;
    int                 kv_pos = m->kv_pos[kv_set];
    int                 kv_len = kv_pos + n_tokens;

    if (kv_len > c.max_seq_len) {
        fprintf(stderr, "[MM3-LM] FATAL: kv_len %d > max_seq %d\n", kv_len, c.max_seq_len);
        return;
    }

    const int kv_pad_raw = (int) GGML_PAD(kv_len, 256);
    const int n_kv_pad   = kv_pad_raw < c.max_seq_len ? kv_pad_raw : c.max_seq_len;

    GraphArena *          arena = (n_tokens > 1) ? &m->arena_prefill : &m->arena_decode;
    struct ggml_context * ctx   = graph_arena_begin(arena);
    struct ggml_cgraph *  gf    = ggml_new_graph_custom(ctx, MM3LM_GRAPH_NODES, false);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, n_tokens);
    ggml_set_name(mask, "causal_mask");
    ggml_set_input(mask);

    struct ggml_tensor * token_ids_t = nullptr;
    struct ggml_tensor * embeds_t    = nullptr;
    struct ggml_tensor * hidden      = nullptr;
    if (input_embeds) {
        embeds_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_tokens);
        ggml_set_name(embeds_t, "input_embeds");
        ggml_set_input(embeds_t);
        hidden = embeds_t;
    } else {
        token_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(token_ids_t, "token_ids");
        ggml_set_input(token_ids_t);
        hidden = ggml_get_rows(ctx, m->embed_prefill, token_ids_t);
    }

    struct ggml_tensor * kv_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_name(kv_rows, "kv_rows");
    ggml_set_input(kv_rows);

    for (int l = 0; l < c.n_layers; l++) {
        MM3LMLayer *         ly   = &m->layers[l];
        struct ggml_tensor * norm = mm3_lm_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);
        struct ggml_tensor * attn =
            mm3_lm_build_attn(ctx, gf, c, ly, norm, positions, mask, kv_rows, m->kv_k[kv_set][l], m->kv_v[kv_set][l],
                              n_kv_pad, n_tokens, m->use_flash_attn, m->clamp_fp16);
        hidden = ggml_add(ctx, hidden, attn);
        if (m->clamp_fp16) {
            hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
        }
        norm                     = mm3_lm_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
        struct ggml_tensor * mlp = mm3_lm_build_mlp(ctx, ly, norm);
        hidden                   = ggml_add(ctx, hidden, mlp);
        if (m->clamp_fp16) {
            hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
        }
    }

    hidden = mm3_lm_rms_norm(ctx, hidden, m->final_norm, c.rms_norm_eps);

    // Whole-sequence hiddens, exposed for the teacher-forced replay
    struct ggml_tensor * all_hidden = nullptr;
    if (n_tokens > 1) {
        all_hidden = ggml_cont(ctx, hidden);
        ggml_set_name(all_hidden, "all_hidden");
        ggml_set_output(all_hidden);
        ggml_build_forward_expand(gf, all_hidden);
        hidden = ggml_view_1d(ctx, hidden, H, (int64_t) (n_tokens - 1) * H * sizeof(float));
    }

    struct ggml_tensor * hidden_out = ggml_cont(ctx, hidden);
    ggml_set_name(hidden_out, "last_hidden");
    ggml_set_output(hidden_out);
    ggml_build_forward_expand(gf, hidden_out);

    struct ggml_tensor * lgt = ggml_mul_mat(ctx, m->lm_head, hidden_out);
    ggml_set_name(lgt, "logits");
    ggml_set_output(lgt);
    ggml_build_forward_expand(gf, lgt);

    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[MM3-LM] FATAL: failed to allocate graph (%d tokens)\n", n_tokens);
        exit(1);
    }

    if (input_embeds) {
        ggml_backend_tensor_set(embeds_t, input_embeds, 0, (size_t) n_tokens * H * sizeof(float));
    } else {
        ggml_backend_tensor_set(token_ids_t, token_ids, 0, n_tokens * sizeof(int));
    }

    {
        std::vector<int>     pos_data(n_tokens);
        std::vector<int64_t> rows_data(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            pos_data[i]  = kv_pos + i;
            rows_data[i] = (int64_t) (kv_pos + i);
        }
        ggml_backend_tensor_set(positions, pos_data.data(), 0, n_tokens * sizeof(int));
        ggml_backend_tensor_set(kv_rows, rows_data.data(), 0, n_tokens * sizeof(int64_t));
    }

    {
        // Row i (query at kv_pos + i) attends columns [0, kv_pos + i];
        // everything past that carries neg inf, padded tail included.
        std::vector<uint16_t> mask_data((size_t) n_kv_pad * n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            int query_abs_pos = kv_pos + i;
            for (int j = 0; j < n_kv_pad; j++) {
                mask_data[(size_t) i * n_kv_pad + j] = ggml_fp32_to_fp16((j <= query_abs_pos) ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(mask, mask_data.data(), 0, (size_t) n_kv_pad * n_tokens * sizeof(uint16_t));
    }

    ggml_backend_sched_graph_compute(m->sched, gf);

    ggml_backend_tensor_get(lgt, logits, 0, (size_t) c.head_vocab * sizeof(float));
    if (out_hidden) {
        ggml_backend_tensor_get(hidden_out, out_hidden, 0, (size_t) H * sizeof(float));
    }
    if (out_hidden_all) {
        ggml_backend_tensor_get(all_hidden ? all_hidden : hidden_out, out_hidden_all, 0,
                                (size_t) n_tokens * H * sizeof(float));
    }

    m->kv_pos[kv_set] += n_tokens;
}

// Batched decode: N sequences, one feedback embedding each, consecutive KV
// sets [s0, s0 + N). logits is [N * head_vocab], out_hidden [N * H].
static void mm3_lm_forward_batch(MM3LM *       m,
                                 const int *   kv_sets,
                                 int           N,
                                 const float * input_embeds,
                                 float *       logits,
                                 float *       out_hidden) {
    const MM3LMConfig & c   = m->cfg;
    int                 H   = c.hidden_size;
    int                 D   = c.head_dim;
    int                 Nh  = c.n_heads;
    int                 Nkv = c.n_kv_heads;

    int max_kv_len = 0;
    for (int i = 0; i < N; i++) {
        int kl = m->kv_pos[kv_sets[i]] + 1;
        if (kl > max_kv_len) {
            max_kv_len = kl;
        }
        if (kl > c.max_seq_len) {
            fprintf(stderr, "[MM3-LM] FATAL: kv_len %d > max_seq %d (set %d)\n", kl, c.max_seq_len, kv_sets[i]);
            exit(1);
        }
    }

    const int kv_pad_raw = (int) GGML_PAD(max_kv_len, 256);
    const int n_kv_pad   = kv_pad_raw < c.max_seq_len ? kv_pad_raw : c.max_seq_len;

    struct ggml_cgraph * gf        = nullptr;
    struct ggml_tensor * positions = nullptr;
    struct ggml_tensor * attn_mask = nullptr;
    struct ggml_tensor * kv_rows   = nullptr;
    struct ggml_tensor * lgt       = nullptr;

    const int  s0         = kv_sets[0];
    const bool need_build = !m->batch_graph.built || m->batch_graph.key_n_kv_pad != n_kv_pad ||
                            m->batch_graph.key_N != N || m->batch_graph.key_s0 != s0;
    if (need_build) {
        static_graph_release(&m->batch_graph.graph, m->sched);
        m->batch_graph.built      = false;
        struct ggml_context * ctx = graph_arena_begin(&m->arena_batch);
        gf                        = ggml_new_graph_custom(ctx, MM3LM_GRAPH_NODES, false);

        struct ggml_tensor * embeds_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        ggml_set_name(embeds_t, "input_embeds");
        ggml_set_input(embeds_t);

        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv_pad, 1, 1, N);
        ggml_set_name(attn_mask, "attn_mask");
        ggml_set_input(attn_mask);

        kv_rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I64, 1, 1, N);
        ggml_set_name(kv_rows, "kv_rows");
        ggml_set_input(kv_rows);

        struct ggml_tensor * hidden = embeds_t;

        for (int l = 0; l < c.n_layers; l++) {
            MM3LMLayer *         ly   = &m->layers[l];
            struct ggml_tensor * norm = mm3_lm_rms_norm(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);

            int                  q_dim  = Nh * D;
            int                  kv_dim = Nkv * D;
            struct ggml_tensor * qkv    = ggml_mul_mat(ctx, ly->qkv, norm);
            struct ggml_tensor * q      = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim, N, qkv->nb[1], 0));
            struct ggml_tensor * k =
                ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, N, qkv->nb[1], (size_t) q_dim * qkv->nb[0]));
            struct ggml_tensor * v =
                ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, N, qkv->nb[1], (size_t) (q_dim + kv_dim) * qkv->nb[0]));

            q = ggml_reshape_3d(ctx, q, D, Nh, N);
            k = ggml_reshape_3d(ctx, k, D, Nkv, N);
            v = ggml_reshape_3d(ctx, v, D, Nkv, N);

            q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
            q = ggml_mul(ctx, q, mm3_lm_f32(ctx, ly->q_norm));
            k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
            k = ggml_mul(ctx, k, mm3_lm_f32(ctx, ly->k_norm));

            q = ggml_rope_ext(ctx, q, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                              0.0f);
            k = ggml_rope_ext(ctx, k, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                              0.0f);

            q = ggml_cont(ctx, q);
            k = ggml_cont(ctx, k);
            v = ggml_cont(ctx, v);
            if (m->clamp_fp16) {
                v = ggml_clamp(ctx, v, -65504.0f, 65504.0f);
            }

            float  scale  = 1.0f / sqrtf((float) D);
            size_t off_s0 = (size_t) s0 * m->kv_k4[l]->nb[3];

            struct ggml_tensor * k_sets = ggml_view_4d(ctx, m->kv_k4[l], D, c.max_seq_len, Nkv, N, m->kv_k4[l]->nb[1],
                                                       m->kv_k4[l]->nb[2], m->kv_k4[l]->nb[3], off_s0);
            struct ggml_tensor * v_sets = ggml_view_4d(ctx, m->kv_v4[l], D, c.max_seq_len, Nkv, N, m->kv_v4[l]->nb[1],
                                                       m->kv_v4[l]->nb[2], m->kv_v4[l]->nb[3], off_s0);

            struct ggml_tensor * k_new = ggml_reshape_4d(ctx, k, D, 1, Nkv, N);
            struct ggml_tensor * v_new = ggml_reshape_4d(ctx, v, D, 1, Nkv, N);

            ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_sets, k_new, kv_rows));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_sets, v_new, kv_rows));

            struct ggml_tensor * q4 = ggml_reshape_4d(ctx, q, D, 1, Nh, N);

            struct ggml_tensor * k_batch = ggml_view_4d(ctx, m->kv_k4[l], D, n_kv_pad, Nkv, N, m->kv_k4[l]->nb[1],
                                                        m->kv_k4[l]->nb[2], m->kv_k4[l]->nb[3], off_s0);
            struct ggml_tensor * v_batch = ggml_view_4d(ctx, m->kv_v4[l], D, n_kv_pad, Nkv, N, m->kv_v4[l]->nb[1],
                                                        m->kv_v4[l]->nb[2], m->kv_v4[l]->nb[3], off_s0);

            struct ggml_tensor * attn_result =
                m->use_flash_attn ? ggml_flash_attn_ext(ctx, q4, k_batch, v_batch, attn_mask, scale, 0.0f, 0.0f) :
                                    mm3_lm_attn_f32(ctx, q4, k_batch, v_batch, attn_mask, scale);
            if (m->use_flash_attn) {
                ggml_flash_attn_ext_set_prec(attn_result, GGML_PREC_F32);
            }

            struct ggml_tensor * attn_cat = ggml_reshape_2d(ctx, attn_result, Nh * D, N);
            hidden                        = ggml_add(ctx, hidden, ggml_mul_mat(ctx, ly->o_proj, attn_cat));
            if (m->clamp_fp16) {
                hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
            }

            norm   = mm3_lm_rms_norm(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
            hidden = ggml_add(ctx, hidden, mm3_lm_build_mlp(ctx, ly, norm));
            if (m->clamp_fp16) {
                hidden = ggml_clamp(ctx, hidden, -65504.0f, 65504.0f);
            }
        }

        hidden = mm3_lm_rms_norm(ctx, hidden, m->final_norm, c.rms_norm_eps);
        ggml_set_name(hidden, "hidden_out");
        ggml_set_output(hidden);
        ggml_build_forward_expand(gf, hidden);

        lgt = ggml_mul_mat(ctx, m->lm_head, hidden);  // [head_vocab, N]
        ggml_set_name(lgt, "logits");
        ggml_set_output(lgt);
        ggml_build_forward_expand(gf, lgt);

        if (!static_graph_alloc(&m->batch_graph.graph, m->backend, m->sched, gf)) {
            fprintf(stderr, "[MM3-LM] FATAL: failed to allocate batch graph (N=%d)\n", N);
            exit(1);
        }

        m->batch_graph.gf           = gf;
        m->batch_graph.embeds_t     = embeds_t;
        m->batch_graph.hidden_out   = hidden;
        m->batch_graph.positions    = positions;
        m->batch_graph.kv_rows      = kv_rows;
        m->batch_graph.attn_mask    = attn_mask;
        m->batch_graph.lgt          = lgt;
        m->batch_graph.key_n_kv_pad = n_kv_pad;
        m->batch_graph.key_N        = N;
        m->batch_graph.key_s0       = s0;
        m->batch_graph.pos_data.resize((size_t) N);
        m->batch_graph.rows_data.resize((size_t) N);
        m->batch_graph.mask_data.resize((size_t) n_kv_pad * (size_t) N);
        m->batch_graph.built = true;
    } else {
        gf        = m->batch_graph.gf;
        positions = m->batch_graph.positions;
        kv_rows   = m->batch_graph.kv_rows;
        attn_mask = m->batch_graph.attn_mask;
        lgt       = m->batch_graph.lgt;
    }

    ggml_backend_tensor_set(m->batch_graph.embeds_t, input_embeds, 0, (size_t) N * H * sizeof(float));

    for (int i = 0; i < N; i++) {
        m->batch_graph.pos_data[(size_t) i]  = m->kv_pos[kv_sets[i]];
        m->batch_graph.rows_data[(size_t) i] = (int64_t) m->kv_pos[kv_sets[i]];
    }
    ggml_backend_tensor_set(positions, m->batch_graph.pos_data.data(), 0, (size_t) N * sizeof(int));
    ggml_backend_tensor_set(kv_rows, m->batch_graph.rows_data.data(), 0, (size_t) N * sizeof(int64_t));

    for (int i = 0; i < N; i++) {
        int kvl = m->kv_pos[kv_sets[i]] + 1;
        for (int j = 0; j < n_kv_pad; j++) {
            m->batch_graph.mask_data[(size_t) i * (size_t) n_kv_pad + (size_t) j] =
                ggml_fp32_to_fp16((j < kvl) ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(attn_mask, m->batch_graph.mask_data.data(), 0,
                            m->batch_graph.mask_data.size() * sizeof(uint16_t));

    static_graph_compute(&m->batch_graph.graph, m->backend, m->sched, gf);

    ggml_backend_tensor_get(lgt, logits, 0, (size_t) c.head_vocab * N * sizeof(float));
    if (out_hidden) {
        ggml_backend_tensor_get(m->batch_graph.hidden_out, out_hidden, 0, (size_t) H * N * sizeof(float));
    }

    for (int i = 0; i < N; i++) {
        m->kv_pos[kv_sets[i]]++;
    }
}

// One row of embed_tokens_audio as host F32, dequantizing through the ggml
// type traits (the shipped checkpoint is Q8_0).
static void mm3_lm_embed_audio_row(MM3LM * m, int code, float * out) {
    int                  H         = m->cfg.hidden_size;
    size_t               row_bytes = ggml_row_size(m->embed_audio->type, H);
    std::vector<uint8_t> raw(row_bytes);
    ggml_backend_tensor_get(m->embed_audio, raw.data(), (size_t) code * row_bytes, row_bytes);
    if (m->embed_audio->type == GGML_TYPE_F32) {
        memcpy(out, raw.data(), (size_t) H * sizeof(float));
        return;
    }
    ggml_get_type_traits(m->embed_audio->type)->to_float(raw.data(), out, H);
}

static void mm3_lm_free(MM3LM * m) {
    static_graph_release(&m->batch_graph.graph, m->sched);
    graph_arena_free(&m->arena_batch);
    graph_arena_free(&m->arena_decode);
    graph_arena_free(&m->arena_prefill);
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    if (m->kv_buf) {
        ggml_backend_buffer_free(m->kv_buf);
    }
    if (m->kv_ctx) {
        ggml_free(m->kv_ctx);
    }
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
