#pragma once
// mm3-dit.h: MiniMax Music 3 flow matching transformer + condition encoder
//
// 36 self attention blocks, model dim 2048 (32 heads x 64), fused SwiGLU FF
// (ff.ff.0.proj emits value then gate, output = ff.ff.2(value * silu(gate))),
// LayerNorm with bias (gamma/beta), partial NeoX RoPE on the first 32 dims
// per head, bidirectional attention.
//
// The input concatenates [latent(128), zeros(128), condition(2048)] along
// channels (the zeros slot carries context latents during training),
// preprocess_conv and postprocess_conv are residual k=1 convs
// (conv(x) + x). The timestep enters as one prefix token: trained Fourier
// features -> linear -> SiLU -> linear, prepended before the blocks and
// dropped after them. The timestep embedding runs on CPU (4.7M MACs), the
// transformer runs as a cached GGML graph on the compute backend.
//
// In the ComfyUI packaging the condition encoder rides in the same GGUF as
// the DiT, so this module owns both:
//   cond_layer_logits [8] / cond_layer_scale [1]
//     -> softmax * scale, folded into a fixed 8 way mix at load
//   latent_conditioners.0.{weight [3, 4096, 2048], bias [2048]}
//     -> conv1d k=3 pad=1, 4096 -> 2048
//   then nearest resample 25 Hz -> 86.1328125 Hz
//      (latent_length = int(frames * 44100 / 24000 * 960 / 512))
//
// RoPE note: `transformer.rotary_pos_emb.inv_freq` is stored in the file
// with BF16 rounded values that differ from 10000^(-2i/32) by ~3e-3
// relative. Recomputing it analytically drifts the phase by ~0.1 rad at
// the far end of a window, so the table is loaded and injected into
// ggml_rope_ext through freq_factors: ggml divides its own analytic theta
// by freq_factors[i], so freq_factors[i] = analytic[i] / file[i] cancels
// the analytic term and leaves exactly the table value.
//
// Sign convention: the ComfyUI wrapper negates the transformer output and
// pairs it with a descending sigma schedule. This port keeps the
// minimaxmusic.cpp convention (ascending sigma, `x += dt * v`, no
// negation), which is algebraically the same trajectory.

#include "backend.h"
#include "debug.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

struct MM3DiTBlock {
    struct ggml_tensor * pre_norm_w, *pre_norm_b;
    struct ggml_tensor * wqkv;                 // [2048, 6144] fused q||k||v, no bias
    struct ggml_tensor * wo;                   // [2048, 2048], no bias
    struct ggml_tensor * ff_norm_w, *ff_norm_b;
    struct ggml_tensor * ff_in_w, *ff_in_b;    // [2048, 16384] value||gate
    struct ggml_tensor * ff_out_w, *ff_out_b;  // [8192, 2048]
};

struct MM3DiT {
    static const int DIM         = 2048;
    static const int IN_CH       = 128;
    static const int COND_DIM    = 2048;
    static const int CONCAT_CH   = 2 * IN_CH + COND_DIM;  // 2304
    static const int N_LAYERS    = 36;
    static const int N_HEADS     = 32;
    static const int HEAD_DIM    = 64;
    static const int FF_INNER    = 8192;
    static const int ROTARY_DIM  = 32;
    static const int FOURIER     = 256;
    static const int COND_HIDDEN = 4096;
    static const int COND_LAYERS = 8;

    // DiT
    struct ggml_tensor * pre_w;    // preprocess_conv [2304, 2304] reshaped at load
    struct ggml_tensor * in_w;     // transformer.project_in [2304, 2048]
    MM3DiTBlock          blocks[N_LAYERS];
    struct ggml_tensor * out_w;    // transformer.project_out [2048, 128]
    struct ggml_tensor * post_w;   // postprocess_conv [128, 128] reshaped at load
    struct ggml_tensor * rope_ff;  // [16] freq_factors carrying the file inv_freq

    // Condition encoder (same GGUF)
    float                mix[COND_LAYERS];  // softmax(cond_layer_logits) * cond_layer_scale
    struct ggml_tensor * cond_w;            // latent_conditioners.0.weight [3, 4096, 2048] F16
    struct ggml_tensor * cond_b;            // latent_conditioners.0.bias [2048]

    // Timestep embedding, host side F32 copies
    std::vector<float> fourier_w;     // [128]
    std::vector<float> te1_w, te1_b;  // [2048, 256], [2048]
    std::vector<float> te2_w, te2_b;  // [2048, 2048], [2048]

    ggml_backend_t       backend        = nullptr;
    ggml_backend_t       cpu_backend    = nullptr;
    BackendPair          bp             = {};
    ggml_backend_sched_t sched          = nullptr;
    ggml_backend_sched_t sched_cond     = nullptr;
    WeightCtx            wctx           = {};
    bool                 use_flash_attn = false;
    bool                 clamp_fp16     = false;  // sub-Ampere CUDA FP16 accumulation overflow

    // DiT graph cache: rebuilt only when T or B changes
    struct ggml_context * graph_ctx    = nullptr;
    uint8_t *             graph_buf    = nullptr;
    struct ggml_cgraph *  graph        = nullptr;
    struct ggml_tensor *  in_xt        = nullptr;
    struct ggml_tensor *  in_cond      = nullptr;
    struct ggml_tensor *  in_temb      = nullptr;
    struct ggml_tensor *  in_pos       = nullptr;
    struct ggml_tensor *  graph_output = nullptr;
    int                   graph_T      = 0;
    int                   graph_B      = 0;

    // Condition projection graph cache: rebuilt only when n_frames changes
    struct ggml_context * cgraph_ctx    = nullptr;
    uint8_t *             cgraph_buf    = nullptr;
    struct ggml_cgraph *  cgraph        = nullptr;
    struct ggml_tensor *  cgraph_input  = nullptr;
    struct ggml_tensor *  cgraph_output = nullptr;
    int                   cgraph_T      = 0;

    bool load(const char * gguf_path);

    // hidden_states: [n_frames, 8, 4096] frame-major, layer-major inside a
    // frame (the reference concat layout). condition: [n_latents, 2048]
    // frame-major on the latent track.
    bool encode_condition(const std::vector<float> & hidden_states,
                          int                        n_frames,
                          std::vector<float> &       condition,
                          int &                      n_latents);

    // One denoising evaluation for a batch of B sequences sharing the
    // schedule step. xt: B contiguous [T, 128] time-major blocks, cond:
    // B contiguous [T, 2048] time-major blocks (zeros for unconditional
    // CFG branches), t in [0, 1] (0 = noise, 1 = data). Writes B
    // contiguous velocity [T, 128] time-major blocks.
    bool forward(const float * xt, const float * cond, int T, int B, float t, float * velocity);

    // Dump the named probe tensors of the last computed graph (cossim harness)
    void dump_named(const DebugDumper * dbg);

    void free();
};

// Number of VAE latents a run of 25 Hz LM frames maps to.
// Matches `latent_length()` in the ComfyUI reference.
static inline int mm3_latent_length(int n_frames) {
    int n = (int) ((double) n_frames * 44100.0 / 24000.0 * 960.0 / 512.0);
    return n < 1 ? 1 : n;
}

inline bool MM3DiT::load(const char * gguf_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[MM3-DiT] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    bp             = backend_init("MM3-DiT");
    backend        = bp.backend;
    cpu_backend    = bp.cpu_backend;
    sched          = backend_sched_new(bp, 4096);
    sched_cond     = backend_sched_new(bp, 64);
    use_flash_attn = bp.has_gpu;

    wctx_init(&wctx, 512);

    const std::string dt = "diffusion_transformer.";

    // k=1 convs [1, C, C] flattened to 2D [C_in, C_out] for mul_mat
    static const int64_t pre_shape[2]  = { CONCAT_CH, CONCAT_CH };
    static const int64_t post_shape[2] = { IN_CH, IN_CH };
    pre_w                              = gf_load_tensor(&wctx, gf, dt + "preprocess_conv.weight", pre_shape, 2);
    post_w                             = gf_load_tensor(&wctx, gf, dt + "postprocess_conv.weight", post_shape, 2);
    in_w                               = gf_load_tensor(&wctx, gf, dt + "transformer.project_in.weight");
    out_w                              = gf_load_tensor(&wctx, gf, dt + "transformer.project_out.weight");

    for (int i = 0; i < N_LAYERS; i++) {
        MM3DiTBlock & b   = blocks[i];
        std::string   pfx = dt + "transformer.layers." + std::to_string(i) + ".";
        b.pre_norm_w      = gf_load_tensor_f32(&wctx, gf, pfx + "pre_norm.gamma");
        b.pre_norm_b      = gf_load_tensor_f32(&wctx, gf, pfx + "pre_norm.beta");
        b.wqkv            = gf_load_tensor(&wctx, gf, pfx + "self_attn.to_qkv.weight");
        b.wo              = gf_load_tensor(&wctx, gf, pfx + "self_attn.to_out.weight");
        b.ff_norm_w       = gf_load_tensor_f32(&wctx, gf, pfx + "ff_norm.gamma");
        b.ff_norm_b       = gf_load_tensor_f32(&wctx, gf, pfx + "ff_norm.beta");
        b.ff_in_w         = gf_load_tensor(&wctx, gf, pfx + "ff.ff.0.proj.weight");
        b.ff_in_b         = gf_load_tensor_f32(&wctx, gf, pfx + "ff.ff.0.proj.bias");
        b.ff_out_w        = gf_load_tensor(&wctx, gf, pfx + "ff.ff.2.weight");
        b.ff_out_b        = gf_load_tensor_f32(&wctx, gf, pfx + "ff.ff.2.bias");
    }

    // Condition projection: the GGUF already stores [k, in, out] = the ggml
    // conv_1d kernel layout, so it only needs the F32 -> F16 cast.
    std::vector<float> cw;
    if (!gf_host_f32(gf, "latent_conditioners.0.weight", cw)) {
        fprintf(stderr, "[MM3-DiT] FATAL: missing latent_conditioners.0.weight\n");
        gf_close(&gf);
        return false;
    }
    cond_w = ggml_new_tensor_3d(wctx.ctx, GGML_TYPE_F16, 3, COND_HIDDEN, COND_DIM);
    ggml_set_name(cond_w, "latent_conditioners.0.weight");
    cond_b = gf_load_tensor_f32(&wctx, gf, "latent_conditioners.0.bias");

    // RoPE inv_freq -> freq_factors. ggml computes theta iteratively as
    // theta *= theta_scale and then divides by freq_factors[i], so the
    // divisor is built with the same accumulation to cancel exactly.
    std::vector<float> inv_freq;
    if (!gf_host_f32(gf, dt + "transformer.rotary_pos_emb.inv_freq", inv_freq) ||
        inv_freq.size() != (size_t) (ROTARY_DIM / 2)) {
        fprintf(stderr, "[MM3-DiT] FATAL: rotary_pos_emb.inv_freq missing or wrong size\n");
        gf_close(&gf);
        return false;
    }
    rope_ff = ggml_new_tensor_1d(wctx.ctx, GGML_TYPE_F32, ROTARY_DIM / 2);
    ggml_set_name(rope_ff, "rope_freq_factors");

    std::vector<float> ff(ROTARY_DIM / 2);
    {
        const float theta_scale = powf(10000.0f, -2.0f / (float) ROTARY_DIM);
        float       analytic    = 1.0f;
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            ff[i] = inv_freq[i] != 0.0f ? analytic / inv_freq[i] : 1.0f;
            analytic *= theta_scale;
        }
    }

    if (!wctx_alloc(&wctx, backend)) {
        gf_close(&gf);
        return false;
    }

    // Post-alloc host uploads for the two tensors built by hand
    {
        std::vector<ggml_fp16_t> cw16(cw.size());
        ggml_fp32_to_fp16_row(cw.data(), cw16.data(), (int64_t) cw.size());
        ggml_backend_tensor_set(cond_w, cw16.data(), 0, cw16.size() * sizeof(ggml_fp16_t));
    }
    ggml_backend_tensor_set(rope_ff, ff.data(), 0, ff.size() * sizeof(float));

    // Condition mix: softmax(logits) * scale, folded once at load
    std::vector<float> logits, scale;
    if (!gf_host_f32(gf, "cond_layer_logits", logits) || !gf_host_f32(gf, "cond_layer_scale", scale) ||
        logits.size() != (size_t) COND_LAYERS || scale.empty()) {
        fprintf(stderr, "[MM3-DiT] FATAL: missing cond_layer_logits / cond_layer_scale\n");
        gf_close(&gf);
        return false;
    }
    {
        float mx = logits[0];
        for (int i = 1; i < COND_LAYERS; i++) {
            mx = logits[i] > mx ? logits[i] : mx;
        }
        float sum = 0;
        for (int i = 0; i < COND_LAYERS; i++) {
            mix[i] = expf(logits[i] - mx);
            sum += mix[i];
        }
        for (int i = 0; i < COND_LAYERS; i++) {
            mix[i] = mix[i] / sum * scale[0];
        }
    }

    // Timestep MLP stays on the host: 4.7M MACs once per step
    gf_host_f32(gf, dt + "timestep_features.weight", fourier_w);
    gf_host_f32(gf, dt + "to_timestep_embed.0.weight", te1_w);
    gf_host_f32(gf, dt + "to_timestep_embed.0.bias", te1_b);
    gf_host_f32(gf, dt + "to_timestep_embed.2.weight", te2_w);
    gf_host_f32(gf, dt + "to_timestep_embed.2.bias", te2_b);
    if (fourier_w.size() != (size_t) (FOURIER / 2) || te1_w.size() != (size_t) DIM * FOURIER ||
        te2_w.size() != (size_t) DIM * DIM) {
        fprintf(stderr, "[MM3-DiT] FATAL: timestep embedding tensors have unexpected sizes\n");
        gf_close(&gf);
        return false;
    }

    fprintf(stderr, "[MM3-DiT] Loaded: %d layers, dim %d, flash_attn=%d, cond mix[0]=%.6f\n", N_LAYERS, DIM,
            use_flash_attn, mix[0]);
    gf_close(&gf);
    return true;
}

// LayerNorm with weight and bias, eps 1e-5
static struct ggml_tensor * mm3_dit_layer_norm(struct ggml_context * ctx,
                                               struct ggml_tensor *  x,
                                               struct ggml_tensor *  w,
                                               struct ggml_tensor *  b) {
    struct ggml_tensor * n = ggml_norm(ctx, x, 1e-5f);
    n                      = ggml_mul(ctx, n, w);
    return ggml_add(ctx, n, b);
}

// F32 manual attention (CPU fallback), bidirectional, no mask
static struct ggml_tensor * mm3_dit_attn_f32(struct ggml_context * ctx,
                                             struct ggml_tensor *  q,
                                             struct ggml_tensor *  k,
                                             struct ggml_tensor *  v,
                                             float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static struct ggml_tensor * mm3_dit_block(struct ggml_context * ctx,
                                          MM3DiT *              m,
                                          MM3DiTBlock *         b,
                                          struct ggml_tensor *  x,  // [2048, S, B]
                                          struct ggml_tensor *  pos,
                                          int                   S,
                                          int                   B,
                                          const char *          sa_name) {
    const int D  = MM3DiT::HEAD_DIM;
    const int Nh = MM3DiT::N_HEADS;

    // Attention
    struct ggml_tensor * h = mm3_dit_layer_norm(ctx, x, b->pre_norm_w, b->pre_norm_b);

    // Fused QKV: [6144, S, B] split by rows into q || k || v (2048 each)
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, b->wqkv, h);
    struct ggml_tensor * q   = ggml_cont(ctx, ggml_view_3d(ctx, qkv, MM3DiT::DIM, S, B, qkv->nb[1], qkv->nb[2], 0));
    struct ggml_tensor * k   = ggml_cont(
        ctx, ggml_view_3d(ctx, qkv, MM3DiT::DIM, S, B, qkv->nb[1], qkv->nb[2], (size_t) MM3DiT::DIM * qkv->nb[0]));
    struct ggml_tensor * v = ggml_cont(
        ctx, ggml_view_3d(ctx, qkv, MM3DiT::DIM, S, B, qkv->nb[1], qkv->nb[2], (size_t) 2 * MM3DiT::DIM * qkv->nb[0]));

    q = ggml_reshape_4d(ctx, q, D, Nh, S, B);
    k = ggml_reshape_4d(ctx, k, D, Nh, S, B);
    v = ggml_reshape_4d(ctx, v, D, Nh, S, B);

    // Partial NeoX RoPE: only the first 32 dims of each head rotate, and
    // the frequencies come from the file table through freq_factors.
    q = ggml_rope_ext(ctx, q, pos, m->rope_ff, MM3DiT::ROTARY_DIM, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f,
                      0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, m->rope_ff, MM3DiT::ROTARY_DIM, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f,
                      0.0f, 0.0f);

    // [D, Nh, S] -> [D, S, Nh]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    float                scale = 1.0f / sqrtf((float) D);
    struct ggml_tensor * attn;
    if (m->use_flash_attn) {
        k    = ggml_cast(ctx, k, GGML_TYPE_F16);
        v    = ggml_cast(ctx, v, GGML_TYPE_F16);
        attn = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = mm3_dit_attn_f32(ctx, q, k, v, scale);
    }
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, B);

    struct ggml_tensor * sa_out = ggml_mul_mat(ctx, b->wo, attn);
    if (sa_name) {
        ggml_set_name(sa_out, sa_name);
        ggml_set_output(sa_out);
    }
    x = ggml_add(ctx, x, sa_out);

    // FF: proj emits [value(8192), gate(8192)], out = ff.2(value * silu(gate))
    h                          = mm3_dit_layer_norm(ctx, x, b->ff_norm_w, b->ff_norm_b);
    struct ggml_tensor * ff    = ggml_mul_mat(ctx, b->ff_in_w, h);
    ff                         = ggml_add(ctx, ff, b->ff_in_b);
    struct ggml_tensor * value = ggml_cont(ctx, ggml_view_3d(ctx, ff, MM3DiT::FF_INNER, S, B, ff->nb[1], ff->nb[2], 0));
    struct ggml_tensor * gate  = ggml_cont(ctx, ggml_view_3d(ctx, ff, MM3DiT::FF_INNER, S, B, ff->nb[1], ff->nb[2],
                                                             (size_t) MM3DiT::FF_INNER * ff->nb[0]));
    struct ggml_tensor * act   = ggml_swiglu_split(ctx, gate, value);  // silu(gate) * value
    struct ggml_tensor * out   = ggml_mul_mat(ctx, b->ff_out_w, act);
    out                        = ggml_add(ctx, out, b->ff_out_b);

    return ggml_add(ctx, x, out);
}

static struct ggml_tensor * mm3_dit_build_graph(struct ggml_context * ctx, MM3DiT * m, int T, int B) {
    int S = T + 1;

    // Channel concat [latent, zeros, condition] -> [2304, T]
    struct ggml_tensor * zeros = ggml_scale(ctx, m->in_xt, 0.0f);
    struct ggml_tensor * x     = ggml_concat(ctx, m->in_xt, zeros, 0);
    x                          = ggml_concat(ctx, x, m->in_cond, 0);

    // Residual k=1 convs as mul_mat
    x = ggml_add(ctx, ggml_mul_mat(ctx, m->pre_w, x), x);
    ggml_set_name(x, "hidden_after_preprocess");
    ggml_set_output(x);

    // project_in then prepend the timestep token
    x = ggml_mul_mat(ctx, m->in_w, x);
    ggml_set_name(x, "hidden_after_proj_in");
    ggml_set_output(x);
    x = ggml_concat(ctx, m->in_temb, x, 1);

    for (int i = 0; i < MM3DiT::N_LAYERS; i++) {
        x = mm3_dit_block(ctx, m, &m->blocks[i], x, m->in_pos, S, B, i == 0 ? "layer0_sa_output" : NULL);
        if (m->clamp_fp16) {
            x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
        }
        // Named probes at key depths for the cossim harness
        if (i == 0 || i == 6 || i == 12 || i == 18 || i == MM3DiT::N_LAYERS - 1) {
            char lname[64];
            snprintf(lname, sizeof(lname), "hidden_after_layer%d", i);
            ggml_set_name(x, lname);
            ggml_set_output(x);
        }
    }

    // Drop the timestep token, project back to latent channels
    struct ggml_tensor * tokens = ggml_view_3d(ctx, x, MM3DiT::DIM, T, B, x->nb[1], x->nb[2], x->nb[1]);
    struct ggml_tensor * y      = ggml_mul_mat(ctx, m->out_w, ggml_cont(ctx, tokens));
    y                           = ggml_add(ctx, ggml_mul_mat(ctx, m->post_w, y), y);
    return y;  // [128, T, B]
}

// CPU timestep embedding: Fourier features -> linear -> SiLU -> linear
static void mm3_dit_time_embed(MM3DiT * m, float t, std::vector<float> & temb) {
    float emb[MM3DiT::FOURIER];
    for (int i = 0; i < MM3DiT::FOURIER / 2; i++) {
        float angle                  = 2.0f * (float) M_PI * t * m->fourier_w[i];
        emb[i]                       = cosf(angle);
        emb[MM3DiT::FOURIER / 2 + i] = sinf(angle);
    }
    std::vector<float> h(MM3DiT::DIM);
    for (int j = 0; j < MM3DiT::DIM; j++) {
        float acc = m->te1_b[j];
        for (int i = 0; i < MM3DiT::FOURIER; i++) {
            acc += m->te1_w[(size_t) j * MM3DiT::FOURIER + i] * emb[i];
        }
        h[j] = acc / (1.0f + expf(-acc));  // SiLU
    }
    temb.resize(MM3DiT::DIM);
    for (int j = 0; j < MM3DiT::DIM; j++) {
        float acc = m->te2_b[j];
        for (int i = 0; i < MM3DiT::DIM; i++) {
            acc += m->te2_w[(size_t) j * MM3DiT::DIM + i] * h[i];
        }
        temb[j] = acc;
    }
}

inline bool MM3DiT::encode_condition(const std::vector<float> & hidden_states,
                                     int                        n_frames,
                                     std::vector<float> &       condition,
                                     int &                      n_latents) {
    if (hidden_states.size() != (size_t) n_frames * COND_LAYERS * COND_HIDDEN) {
        fprintf(stderr, "[MM3-Cond] hidden size %zu does not match T=%d\n", hidden_states.size(), n_frames);
        return false;
    }

    // Mix: [T, 8, 4096] -> ggml input [T, 4096] channel-major
    std::vector<float> mixed((size_t) COND_HIDDEN * n_frames);
    for (int t = 0; t < n_frames; t++) {
        const float * frame = hidden_states.data() + (size_t) t * COND_LAYERS * COND_HIDDEN;
        for (int h = 0; h < COND_HIDDEN; h++) {
            float acc = 0;
            for (int l = 0; l < COND_LAYERS; l++) {
                acc += mix[l] * frame[(size_t) l * COND_HIDDEN + h];
            }
            mixed[(size_t) h * n_frames + t] = acc;
        }
    }

    // Projection graph: conv1d k=3 pad=1, [T, 4096] -> [T, 2048]
    if (cgraph_T != n_frames) {
        if (cgraph_ctx) {
            ggml_backend_sched_reset(sched_cond);
            ggml_free(cgraph_ctx);
            std::free(cgraph_buf);
            cgraph_ctx = nullptr;
            cgraph_buf = nullptr;
        }

        size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
        cgraph_buf      = (uint8_t *) malloc(ctx_size);
        if (!cgraph_buf) {
            fprintf(stderr, "[MM3-Cond] FATAL: OOM allocating graph context for T=%d\n", n_frames);
            cgraph_T = 0;
            return false;
        }
        struct ggml_init_params p   = { ctx_size, cgraph_buf, true };
        struct ggml_context *   ctx = ggml_init(p);

        cgraph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, COND_HIDDEN);
        ggml_set_name(cgraph_input, "cond_input");
        ggml_set_input(cgraph_input);

        struct ggml_tensor * y   = ggml_conv_1d(ctx, cond_w, cgraph_input, 1, 1, 1);
        y                        = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, cond_b, 1, COND_DIM);
        cgraph_output            = ggml_add(ctx, y, b2d);
        if (clamp_fp16) {
            cgraph_output = ggml_clamp(ctx, cgraph_output, -65504.0f, 65504.0f);
        }
        ggml_set_name(cgraph_output, "cond_output");
        ggml_set_output(cgraph_output);

        cgraph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(cgraph, cgraph_output);

        if (!ggml_backend_sched_alloc_graph(sched_cond, cgraph)) {
            fprintf(stderr, "[MM3-Cond] FATAL: graph alloc failed for T=%d\n", n_frames);
            ggml_free(ctx);
            std::free(cgraph_buf);
            cgraph_ctx = NULL;
            cgraph_buf = NULL;
            cgraph_T   = 0;
            return false;
        }

        cgraph_ctx = ctx;
        cgraph_T   = n_frames;
    }

    ggml_backend_tensor_set(cgraph_input, mixed.data(), 0, mixed.size() * sizeof(float));
    ggml_backend_sched_graph_compute(sched_cond, cgraph);

    std::vector<float> proj((size_t) COND_DIM * n_frames);
    ggml_backend_tensor_get(cgraph_output, proj.data(), 0, proj.size() * sizeof(float));

    // Nearest resample to the latent track, output frame-major [n_latents, 2048]
    n_latents = mm3_latent_length(n_frames);
    condition.resize((size_t) n_latents * COND_DIM);
    for (int i = 0; i < n_latents; i++) {
        int src = (int) ((double) i * n_frames / n_latents);
        for (int c = 0; c < COND_DIM; c++) {
            condition[(size_t) i * COND_DIM + c] = proj[(size_t) c * n_frames + src];
        }
    }
    return true;
}

inline bool MM3DiT::forward(const float * xt, const float * cond, int T, int B, float t, float * velocity) {
    if (graph_T != T || graph_B != B) {
        if (graph_ctx) {
            ggml_backend_sched_reset(sched);
            ggml_free(graph_ctx);
            std::free(graph_buf);
            graph_ctx = nullptr;
            graph_buf = nullptr;
        }

        size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
        graph_buf       = (uint8_t *) malloc(ctx_size);
        if (!graph_buf) {
            fprintf(stderr, "[MM3-DiT] FATAL: OOM allocating graph context for T=%d\n", T);
            graph_T = 0;
            return false;
        }
        struct ggml_init_params p   = { ctx_size, graph_buf, true };
        struct ggml_context *   ctx = ggml_init(p);

        in_xt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, IN_CH, T, B);
        in_cond = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, COND_DIM, T, B);
        in_temb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DIM, 1, B);
        ggml_set_name(in_temb, "temb_t");
        in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T + 1);
        ggml_set_input(in_xt);
        ggml_set_input(in_cond);
        ggml_set_input(in_temb);
        ggml_set_input(in_pos);

        graph_output = mm3_dit_build_graph(ctx, this, T, B);
        ggml_set_name(graph_output, "dit_output");
        ggml_set_output(graph_output);

        graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph, graph_output);

        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            fprintf(stderr, "[MM3-DiT] FATAL: graph alloc failed for T=%d B=%d\n", T, B);
            ggml_free(ctx);
            std::free(graph_buf);
            graph_ctx = NULL;
            graph_buf = NULL;
            graph_T   = 0;
            graph_B   = 0;
            return false;
        }

        graph_ctx = ctx;
        graph_T   = T;
        graph_B   = B;
        fprintf(stderr, "[MM3-DiT] Graph: %d nodes, T=%d, B=%d\n", ggml_graph_n_nodes(graph), T, B);
    }

    // The schedule step is shared: one embedding replicated per element
    std::vector<float> temb1;
    mm3_dit_time_embed(this, t, temb1);
    std::vector<float> temb((size_t) DIM * B);
    for (int b = 0; b < B; b++) {
        memcpy(temb.data() + (size_t) b * DIM, temb1.data(), DIM * sizeof(float));
    }

    std::vector<int32_t> pos(T + 1);
    for (int i = 0; i <= T; i++) {
        pos[i] = i;
    }

    ggml_backend_tensor_set(in_xt, xt, 0, (size_t) IN_CH * T * B * sizeof(float));
    ggml_backend_tensor_set(in_cond, cond, 0, (size_t) COND_DIM * T * B * sizeof(float));
    ggml_backend_tensor_set(in_temb, temb.data(), 0, (size_t) DIM * B * sizeof(float));
    ggml_backend_tensor_set(in_pos, pos.data(), 0, (T + 1) * sizeof(int32_t));

    ggml_backend_sched_graph_compute(sched, graph);

    ggml_backend_tensor_get(graph_output, velocity, 0, (size_t) IN_CH * T * B * sizeof(float));
    return true;
}

inline void MM3DiT::free() {
    if (graph_ctx) {
        ggml_free(graph_ctx);
        std::free(graph_buf);
        graph_ctx = nullptr;
        graph_buf = nullptr;
        graph_T   = 0;
        graph_B   = 0;
    }
    if (cgraph_ctx) {
        ggml_free(cgraph_ctx);
        std::free(cgraph_buf);
        cgraph_ctx = nullptr;
        cgraph_buf = nullptr;
        cgraph_T   = 0;
    }
    if (sched) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (sched_cond) {
        ggml_backend_sched_free(sched_cond);
        sched_cond = nullptr;
    }
    wctx_free(&wctx);
    // backends are refcounted and shared across all modules
    backend_release(backend, cpu_backend);
    backend     = nullptr;
    cpu_backend = nullptr;
}

// Reads each named probe tensor from the cached graph and dumps it
// time-major [ne1, ne0], matching the torch reference hook layouts.
inline void MM3DiT::dump_named(const DebugDumper * dbg) {
    if (!dbg || !dbg->enabled || !graph) {
        return;
    }
    const char * names[] = { "temb_t",
                             "hidden_after_preprocess",
                             "hidden_after_proj_in",
                             "layer0_sa_output",
                             "hidden_after_layer0",
                             "hidden_after_layer6",
                             "hidden_after_layer12",
                             "hidden_after_layer18",
                             "hidden_after_layer35" };
    for (const char * name : names) {
        struct ggml_tensor * t = ggml_graph_get_tensor(graph, name);
        if (!t) {
            continue;
        }
        int64_t            n0 = t->ne[0];
        int64_t            n1 = t->ne[1];
        std::vector<float> buf((size_t) n0 * n1);
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t) n0 * n1 * sizeof(float));
        if (n1 <= 1) {
            debug_dump_1d(dbg, name, buf.data(), (int) n0);
        } else {
            debug_dump_2d(dbg, name, buf.data(), (int) n1, (int) n0);
        }
    }
}

static void mm3_dit_free(MM3DiT * m) {
    m->free();
}
