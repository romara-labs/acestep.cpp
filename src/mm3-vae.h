#pragma once
// mm3-vae.h: MiniMax Music 3 flow VAE decoder (DAC style upsampling stack)
//
// Decodes the 128 channel flow VAE latent at 86.13 Hz into 44.1 kHz audio.
// The 128 latent channels are two independent 64 channel tracks, one per
// stereo side: the decoder runs once per side and emits a mono waveform.
//
// Topology (vocoder/, ComfyUI `minimax_music3_dav.safetensors` names):
//   dec_in_proj:      conv1d 64 -> 1024 k=1
//   decoder.model.0:  conv_in, conv1d 1024 -> 1536 k=7
//   decoder.model.1..4: blocks 0..3
//       .block.0        snake1 alpha
//       .block.1        conv_transpose1d stride s, k = 2s, pad = ceil(s/2)
//       .block.2/3/4    res units, dilation 1 / 3 / 9
//     strides [8, 8, 4, 2], dims 1536 -> 768 -> 384 -> 192 -> 96
//   ResUnit: skip -> snake1 -> conv1 k=7 dil -> snake2 -> conv2 k=1 -> + skip
//   decoder.model.5:  snake_out
//   decoder.model.6:  conv_out, conv1d 96 -> 1 k=7, then tanh
//
// Snake: y = x + sin(alpha * x)^2 / (alpha + 1e-9), emitted as the 5-op
// decomposition (mul -> sin -> sqr -> mul -> add) that backends pattern-match
// into their fused snake kernel. alpha is used as stored: this checkpoint is
// not the Oobleck one, there is no exp() on load.
// ConvTranspose1d: mul_mat on the pre-permuted [IC, K*OC] weight + col2im_1d.
//
// Unlike the ACE-Step VAE this file reads a *safetensors* checkpoint with
// the weight norm parametrization still split, so the loader folds
// w = g * v / ||v|| per output row on the host before uploading. There is
// no encoder in the file (MM3 is text to music only), so decode only.

#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "safetensors.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct MM3VAEResUnit {
    struct ggml_tensor *s1a, *s1i;  // snake1 alpha, 1/(alpha + 1e-9) [1, C]
    struct ggml_tensor *c1w, *c1b;  // conv1 [7, C, C], bias [C]
    struct ggml_tensor *s2a, *s2i;  // snake2
    struct ggml_tensor *c2w, *c2b;  // conv2 [1, C, C], bias [C]
    int                 dilation;
};

struct MM3VAEBlock {
    struct ggml_tensor *sa, *si;    // snake1 alpha, inv [1, in_ch]
    struct ggml_tensor *ctw, *ctb;  // conv_t F16 [IC, K*OC] pre-permuted, bias [out_ch]
    int                 in_ch, out_ch, stride, kernel;
    MM3VAEResUnit       ru[3];
};

struct MM3VAE {
    static const int LATENT_CH   = 128;  // both sides
    static const int SIDE_CH     = 64;   // per side
    static const int HOP         = 512;
    static const int SAMPLE_RATE = 44100;

    struct ggml_tensor *piw, *pib;  // dec_in_proj [1, 64, 1024], bias [1024]
    struct ggml_tensor *c1w, *c1b;  // conv_in [7, 1024, 1536], bias [1536]
    MM3VAEBlock         blk[4];
    struct ggml_tensor *sa, *si;    // snake_out alpha, inv [1, 96]
    struct ggml_tensor *c2w, *c2b;  // conv_out [7, 96, 1], bias [1]

    ggml_backend_t        backend     = nullptr;
    ggml_backend_t        cpu_backend = nullptr;
    ggml_backend_sched_t  sched       = nullptr;
    ggml_backend_buffer_t buf         = nullptr;
    struct ggml_context * weight_ctx  = nullptr;

    // Graph cache: rebuilt only when T_latent changes
    struct ggml_context * graph_ctx    = nullptr;
    uint8_t *             graph_buf    = nullptr;
    struct ggml_cgraph *  graph        = nullptr;
    struct ggml_tensor *  graph_input  = nullptr;
    struct ggml_tensor *  graph_output = nullptr;
    int                   graph_T      = 0;

    bool load(const char * safetensors_path);

    // Decodes the latent [128, T] channel-major (torch memory order) into
    // interleaved stereo samples at 44.1 kHz (T * 512 frames per side).
    bool decode(const std::vector<float> & latent, int n_latents, std::vector<float> & audio);

    void free();
};

// Safetensors helpers

static const STEntry * mm3_st_find(const STFile & st, const std::string & name) {
    for (const auto & e : st.entries) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// Read one entry into a host F32 vector. F32 passthrough, F16/BF16 widened.
static bool mm3_st_f32(const STFile & st, const std::string & name, std::vector<float> & out) {
    const STEntry * e = mm3_st_find(st, name);
    if (!e) {
        fprintf(stderr, "[MM3-VAE] FATAL: missing tensor %s\n", name.c_str());
        return false;
    }
    size_t n = 1;
    for (int i = 0; i < e->n_dims; i++) {
        n *= (size_t) e->shape[i];
    }
    out.resize(n);
    const void * raw = st_data(st, *e);
    if (e->dtype == "F32") {
        memcpy(out.data(), raw, n * sizeof(float));
        return true;
    }
    if (e->dtype == "F16") {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, out.data(), (int64_t) n);
        return true;
    }
    if (e->dtype == "BF16") {
        const uint16_t * p = (const uint16_t *) raw;
        for (size_t i = 0; i < n; i++) {
            out[i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &p[i]);
        }
        return true;
    }
    fprintf(stderr, "[MM3-VAE] FATAL: unsupported dtype %s for %s\n", e->dtype.c_str(), name.c_str());
    return false;
}

// Fold the weight norm parametrization of `prefix`: reads weight_g [R,1,1]
// and weight_v [R, ...], writes w[r] = g[r] * v[r] / ||v[r]||_2 into out.
// R is the torch dim-0 length (output channels for Conv1d, *input*
// channels for ConvTranspose1d, which is what nn.utils weight_norm
// normalizes over by default). Also reports the trailing shape so callers
// can reinterpret the flat buffer.
static bool mm3_st_fold_weight_norm(const STFile &       st,
                                    const std::string &  prefix,
                                    std::vector<float> & out,
                                    int64_t              shape[3]) {
    const STEntry * ve = mm3_st_find(st, prefix + ".weight_v");
    if (!ve || ve->n_dims != 3) {
        fprintf(stderr, "[MM3-VAE] FATAL: %s.weight_v missing or not 3D\n", prefix.c_str());
        return false;
    }
    std::vector<float> g, v;
    if (!mm3_st_f32(st, prefix + ".weight_g", g) || !mm3_st_f32(st, prefix + ".weight_v", v)) {
        return false;
    }
    shape[0] = ve->shape[0];
    shape[1] = ve->shape[1];
    shape[2] = ve->shape[2];

    int64_t R    = shape[0];
    size_t  span = (size_t) shape[1] * (size_t) shape[2];
    if (g.size() != (size_t) R) {
        fprintf(stderr, "[MM3-VAE] FATAL: %s.weight_g has %zu entries, expected %lld\n", prefix.c_str(), g.size(),
                (long long) R);
        return false;
    }

    out.resize((size_t) R * span);
    for (int64_t r = 0; r < R; r++) {
        const float * src = v.data() + (size_t) r * span;
        double        ss  = 0.0;
        for (size_t i = 0; i < span; i++) {
            ss += (double) src[i] * (double) src[i];
        }
        float   norm  = (float) sqrt(ss);
        float   scale = norm > 0.0f ? g[(size_t) r] / norm : 0.0f;
        float * dst   = out.data() + (size_t) r * span;
        for (size_t i = 0; i < span; i++) {
            dst[i] = src[i] * scale;
        }
    }
    return true;
}

// Conv1d weight -> F16 tensor. torch [OC, IC, K] row-major has exactly the
// ggml [K, IC, OC] memory order, so the folded buffer is uploaded as is.
static bool mm3_vae_load_conv(struct ggml_tensor * dst, const STFile & st, const std::string & prefix) {
    std::vector<float> w;
    int64_t            shape[3];
    if (!mm3_st_fold_weight_norm(st, prefix, w, shape)) {
        return false;
    }
    if ((size_t) ggml_nelements(dst) != w.size()) {
        fprintf(stderr, "[MM3-VAE] FATAL: %s has %zu elements, tensor wants %lld\n", prefix.c_str(), w.size(),
                (long long) ggml_nelements(dst));
        return false;
    }
    std::vector<ggml_fp16_t> w16(w.size());
    ggml_fp32_to_fp16_row(w.data(), w16.data(), (int64_t) w.size());
    ggml_backend_tensor_set(dst, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
    return true;
}

// ConvTranspose1d weight, torch [IC, OC, K] -> F16 [IC, K*OC] pre-permuted
// for mul_mat, col rows ordered oc-major k-minor to match col2im_1d.
static bool mm3_vae_load_conv_t(struct ggml_tensor * dst, const STFile & st, const std::string & prefix) {
    std::vector<float> wv;
    int64_t            shape[3];
    if (!mm3_st_fold_weight_norm(st, prefix, wv, shape)) {
        return false;
    }
    const float * w  = wv.data();
    int           IC = (int) shape[0];
    int           OC = (int) shape[1];
    int           K  = (int) shape[2];

    std::vector<ggml_fp16_t> p((size_t) IC * K * OC);
    for (int ic = 0; ic < IC; ic++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                p[(size_t) (oc * K + k) * IC + ic] = ggml_fp32_to_fp16(w[((size_t) ic * OC + oc) * K + k]);
            }
        }
    }
    if ((size_t) ggml_nelements(dst) != p.size()) {
        fprintf(stderr, "[MM3-VAE] FATAL: %s conv_t size mismatch\n", prefix.c_str());
        return false;
    }
    ggml_backend_tensor_set(dst, p.data(), 0, p.size() * sizeof(ggml_fp16_t));
    return true;
}

static bool mm3_vae_load_f32(struct ggml_tensor * dst, const STFile & st, const std::string & name) {
    std::vector<float> w;
    if (!mm3_st_f32(st, name, w)) {
        return false;
    }
    ggml_backend_tensor_set(dst, w.data(), 0, w.size() * sizeof(float));
    return true;
}

// Snake alpha [1, C, 1] -> two F32 [1, C]: alpha and 1/(alpha + 1e-9)
static bool mm3_vae_load_snake(struct ggml_tensor * dst_a,
                               struct ggml_tensor * dst_i,
                               const STFile &       st,
                               const std::string &  name) {
    std::vector<float> a;
    if (!mm3_st_f32(st, name, a)) {
        return false;
    }
    std::vector<float> inv(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        inv[i] = 1.0f / (a[i] + 1e-9f);
    }
    ggml_backend_tensor_set(dst_a, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(dst_i, inv.data(), 0, inv.size() * sizeof(float));
    return true;
}

inline bool MM3VAE::load(const char * safetensors_path) {
    STFile st = {};
    if (!st_open(&st, safetensors_path)) {
        fprintf(stderr, "[MM3-VAE] FATAL: cannot load %s\n", safetensors_path);
        return false;
    }

    static const int strides[]   = { 8, 8, 4, 2 };
    static const int in_ch[]     = { 1536, 768, 384, 192 };
    static const int out_ch[]    = { 768, 384, 192, 96 };
    static const int dilations[] = { 1, 3, 9 };

    // Phase 1: tensor metadata (no_alloc context)
    size_t                  ctx_size = ggml_tensor_overhead() * 128;
    struct ggml_init_params p        = { ctx_size, NULL, true };
    weight_ctx                       = ggml_init(p);
    struct ggml_context * ctx        = weight_ctx;

    piw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, SIDE_CH, 1024);
    pib = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 1024, 1536);
    c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1536);

    for (int i = 0; i < 4; i++) {
        MM3VAEBlock & b = blk[i];
        b.in_ch         = in_ch[i];
        b.out_ch        = out_ch[i];
        b.stride        = strides[i];
        b.kernel        = strides[i] * 2;
        int C           = out_ch[i];
        b.sa            = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.si            = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, in_ch[i]);
        b.ctw           = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, in_ch[i], b.kernel * out_ch[i]);
        b.ctb           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch[i]);
        for (int r = 0; r < 3; r++) {
            MM3VAEResUnit & ru = b.ru[r];
            ru.dilation        = dilations[r];
            ru.s1a             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1i             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w             = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2i             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w             = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
    }
    sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 96);
    si  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 96);
    c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 96, 1);
    c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    // Phase 2: backend buffer
    BackendPair bp = backend_init("MM3-VAE");
    backend        = bp.backend;
    cpu_backend    = bp.cpu_backend;
    sched          = backend_sched_new(bp, 8192);
    buf            = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "[MM3-VAE] FATAL: failed to allocate weight buffer\n");
        st_close(&st);
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    fprintf(stderr, "[MM3-VAE] Backend: %s, Weight buffer: %.1f MB\n", ggml_backend_name(backend),
            (float) ggml_backend_buffer_get_size(buf) / (1024 * 1024));

    // Phase 3: load weights, folding the weight norm parametrization
    bool ok = true;
    // dec_in_proj is a plain Conv1d (no weight norm)
    {
        std::vector<float> w;
        ok = ok && mm3_st_f32(st, "dec_in_proj.weight", w);
        if (ok) {
            std::vector<ggml_fp16_t> w16(w.size());
            ggml_fp32_to_fp16_row(w.data(), w16.data(), (int64_t) w.size());
            ggml_backend_tensor_set(piw, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
        }
    }
    ok = ok && mm3_vae_load_f32(pib, st, "dec_in_proj.bias");
    ok = ok && mm3_vae_load_conv(c1w, st, "decoder.model.0");
    ok = ok && mm3_vae_load_f32(c1b, st, "decoder.model.0.bias");

    for (int i = 0; i < 4 && ok; i++) {
        MM3VAEBlock & b       = blk[i];
        std::string   blk_pfx = "decoder.model." + std::to_string(i + 1);
        ok                    = ok && mm3_vae_load_snake(b.sa, b.si, st, blk_pfx + ".block.0.alpha");
        ok                    = ok && mm3_vae_load_conv_t(b.ctw, st, blk_pfx + ".block.1");
        ok                    = ok && mm3_vae_load_f32(b.ctb, st, blk_pfx + ".block.1.bias");
        for (int r = 0; r < 3 && ok; r++) {
            MM3VAEResUnit & ru = b.ru[r];
            std::string     rp = blk_pfx + ".block." + std::to_string(r + 2) + ".block.";
            ok                 = ok && mm3_vae_load_snake(ru.s1a, ru.s1i, st, rp + "0.alpha");
            ok                 = ok && mm3_vae_load_conv(ru.c1w, st, rp + "1");
            ok                 = ok && mm3_vae_load_f32(ru.c1b, st, rp + "1.bias");
            ok                 = ok && mm3_vae_load_snake(ru.s2a, ru.s2i, st, rp + "2.alpha");
            ok                 = ok && mm3_vae_load_conv(ru.c2w, st, rp + "3");
            ok                 = ok && mm3_vae_load_f32(ru.c2b, st, rp + "3.bias");
        }
    }
    ok = ok && mm3_vae_load_snake(sa, si, st, "decoder.model.5.alpha");
    ok = ok && mm3_vae_load_conv(c2w, st, "decoder.model.6");
    ok = ok && mm3_vae_load_f32(c2b, st, "decoder.model.6.bias");

    st_close(&st);
    if (!ok) {
        return false;
    }

    fprintf(stderr, "[MM3-VAE] Loaded: 4 blocks, upsample=%dx, weight-norm folded at load\n", HOP);
    return true;
}

// Graph building

// Snake activation, 5-op decomposition for backend pattern fusion
// y = x + sin(alpha * x)^2 * inv, x: [T, C], alpha/inv: [1, C]
static struct ggml_tensor * mm3_vae_snake(struct ggml_context * ctx,
                                          struct ggml_tensor *  x,
                                          struct ggml_tensor *  alpha,
                                          struct ggml_tensor *  inv) {
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    struct ggml_tensor * s  = ggml_sin(ctx, ax);
    struct ggml_tensor * s2 = ggml_sqr(ctx, s);
    struct ggml_tensor * d  = ggml_mul(ctx, s2, inv);
    return ggml_add(ctx, x, d);
}

// Conv1d + bias: data [T, IC] -> [T_out, OC]
static struct ggml_tensor * mm3_vae_conv1d(struct ggml_context * ctx,
                                           struct ggml_tensor *  w,  // [K, IC, OC] F16
                                           struct ggml_tensor *  b,  // [OC] or NULL
                                           struct ggml_tensor *  x,  // [T, IC]
                                           int                   stride,
                                           int                   padding,
                                           int                   dilation) {
    struct ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    y                      = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}

// ConvTranspose1d via GEMM + col2im_1d
// w: [IC, K*OC] pre-permuted at load, x: [T_in, IC] -> [T_out, OC]
static struct ggml_tensor * mm3_vae_conv_t1d(struct ggml_context * ctx,
                                             struct ggml_tensor *  w,
                                             struct ggml_tensor *  b,
                                             struct ggml_tensor *  x,
                                             int                   stride,
                                             int                   padding,
                                             int                   oc) {
    struct ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor * col = ggml_mul_mat(ctx, w, xt);
    struct ggml_tensor * y   = ggml_col2im_1d(ctx, col, stride, oc, padding);
    if (b) {
        struct ggml_tensor * b2d = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        y                        = ggml_add(ctx, y, b2d);
    }
    return y;
}

static struct ggml_tensor * mm3_vae_res_unit(struct ggml_context * ctx, MM3VAEResUnit * ru, struct ggml_tensor * x) {
    struct ggml_tensor * skip = x;
    int                  pad  = 3 * ru->dilation;
    x                         = mm3_vae_snake(ctx, x, ru->s1a, ru->s1i);
    x                         = mm3_vae_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x                         = mm3_vae_snake(ctx, x, ru->s2a, ru->s2i);
    x                         = mm3_vae_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

// One stereo side: latent [T, 64] -> waveform [T * 512, 1]
static struct ggml_tensor * mm3_vae_build_graph(struct ggml_context * ctx, MM3VAE * m, struct ggml_tensor * latent) {
    struct ggml_tensor * x = mm3_vae_conv1d(ctx, m->piw, m->pib, latent, 1, 0, 1);
    x                      = mm3_vae_conv1d(ctx, m->c1w, m->c1b, x, 1, 3, 1);

    for (int i = 0; i < 4; i++) {
        MM3VAEBlock & b = m->blk[i];
        x               = mm3_vae_snake(ctx, x, b.sa, b.si);
        // torch pad = ceil(stride / 2), identical to (kernel - stride) / 2
        // for kernel = 2 * stride
        int pad         = (b.kernel - b.stride) / 2;
        x               = mm3_vae_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad, b.out_ch);
        for (int r = 0; r < 3; r++) {
            x = mm3_vae_res_unit(ctx, &b.ru[r], x);
        }
    }

    x = mm3_vae_snake(ctx, x, m->sa, m->si);
    x = mm3_vae_conv1d(ctx, m->c2w, m->c2b, x, 1, 3, 1);
    return ggml_tanh(ctx, x);
}

// Ensures the graph is cached for T_latent, runs one side, returns T_audio or -1
static int mm3_vae_compute(MM3VAE * m, const float * latent_side, int T_latent) {
    if (m->graph_T != T_latent) {
        if (m->graph_ctx) {
            ggml_backend_sched_reset(m->sched);
            ggml_free(m->graph_ctx);
            std::free(m->graph_buf);
            m->graph_ctx = nullptr;
            m->graph_buf = nullptr;
        }

        size_t ctx_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
        m->graph_buf    = (uint8_t *) malloc(ctx_size);
        if (!m->graph_buf) {
            fprintf(stderr, "[MM3-VAE] FATAL: OOM allocating graph context for T=%d\n", T_latent);
            m->graph_T = 0;
            return -1;
        }
        struct ggml_init_params p   = { ctx_size, m->graph_buf, true };
        struct ggml_context *   ctx = ggml_init(p);

        m->graph_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_latent, MM3VAE::SIDE_CH);
        ggml_set_name(m->graph_input, "mm3_vae_input");
        ggml_set_input(m->graph_input);

        m->graph_output = mm3_vae_build_graph(ctx, m, m->graph_input);
        ggml_set_name(m->graph_output, "mm3_vae_output");
        ggml_set_output(m->graph_output);

        m->graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(m->graph, m->graph_output);

        if (!ggml_backend_sched_alloc_graph(m->sched, m->graph)) {
            fprintf(stderr, "[MM3-VAE] FATAL: graph alloc failed for T=%d\n", T_latent);
            ggml_free(ctx);
            std::free(m->graph_buf);
            m->graph_ctx = NULL;
            m->graph_buf = NULL;
            m->graph_T   = 0;
            return -1;
        }

        m->graph_ctx = ctx;
        m->graph_T   = T_latent;
        fprintf(stderr, "[MM3-VAE] Graph: %d nodes, T_latent=%d\n", ggml_graph_n_nodes(m->graph), T_latent);
    }

    ggml_backend_tensor_set(m->graph_input, latent_side, 0, (size_t) MM3VAE::SIDE_CH * T_latent * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, m->graph);

    return (int) m->graph_output->ne[0];
}

inline bool MM3VAE::decode(const std::vector<float> & latent, int n_latents, std::vector<float> & audio) {
    if ((int) latent.size() != LATENT_CH * n_latents) {
        fprintf(stderr, "[MM3-VAE] latent size %zu does not match T=%d\n", latent.size(), n_latents);
        return false;
    }

    int T_audio = n_latents * HOP;
    audio.resize((size_t) T_audio * 2);
    std::vector<float> side(T_audio);

    for (int ch = 0; ch < 2; ch++) {
        int T_out = mm3_vae_compute(this, latent.data() + (size_t) ch * SIDE_CH * n_latents, n_latents);
        if (T_out != T_audio) {
            fprintf(stderr, "[MM3-VAE] decode failed on side %d (T_out=%d, want %d)\n", ch, T_out, T_audio);
            return false;
        }
        ggml_backend_tensor_get(graph_output, side.data(), 0, (size_t) T_audio * sizeof(float));
        for (int t = 0; t < T_audio; t++) {
            audio[(size_t) t * 2 + ch] = side[t];
        }
    }

    fprintf(stderr, "[MM3-VAE] Decoded: T_latent=%d -> T_audio=%d (%.2fs @ %d Hz)\n", n_latents, T_audio,
            (float) T_audio / (float) SAMPLE_RATE, SAMPLE_RATE);
    return true;
}

inline void MM3VAE::free() {
    if (graph_ctx) {
        ggml_free(graph_ctx);
        std::free(graph_buf);
        graph_ctx = nullptr;
        graph_buf = nullptr;
        graph_T   = 0;
    }
    if (sched) {
        ggml_backend_sched_free(sched);
        sched = nullptr;
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
        buf = nullptr;
    }
    if (weight_ctx) {
        ggml_free(weight_ctx);
        weight_ctx = nullptr;
    }
    // backends are refcounted and shared across all modules
    backend_release(backend, cpu_backend);
    backend     = nullptr;
    cpu_backend = nullptr;
}

static void mm3_vae_free(MM3VAE * m) {
    m->free();
}
