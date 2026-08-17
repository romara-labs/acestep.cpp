// test-mm3.cpp: load and smoke test every MiniMax Music 3 module against
// the real checkpoint files, plus the empirical fixtures from the port
// notes (condition mix, rotary inv_freq, tokenizer vocab).
//
//   test-mm3 <models_dir> [--skip-lm] [--dump-vae <prefix>]
//
// The LM is 8.9 GB, so --skip-lm keeps the fast structural checks usable
// on a small machine. --dump-vae writes <prefix>-in.f32 and <prefix>-out.f32
// for tests/mm3-vae-ref.py, which recomputes the decoder in numpy straight
// from the safetensors and reports the cosine similarity.

#include "mm3-depth.h"
#include "mm3-dit.h"
#include "mm3-lm.h"
#include "mm3-pipeline.h"
#include "mm3-prompt.h"
#include "mm3-vae.h"
#include "model-registry.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    fprintf(stderr, "  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_fail++;
    }
}

static void check_close(float got, float want, float tol, const char * what) {
    bool ok = fabsf(got - want) <= tol;
    fprintf(stderr, "  [%s] %s (got %.8f, want %.8f)\n", ok ? "PASS" : "FAIL", what, (double) got, (double) want);
    if (!ok) {
        g_fail++;
    }
}

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <models_dir> [--skip-lm]\n", argv[0]);
        return 1;
    }
    const char * models_dir = argv[1];
    bool         skip_lm    = false;
    const char * dump_vae   = nullptr;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--skip-lm")) {
            skip_lm = true;
        } else if (!strcmp(argv[i], "--dump-vae") && i + 1 < argc) {
            dump_vae = argv[++i];
        }
    }

    // 1. Registry classification
    fprintf(stderr, "\n=== registry ===\n");
    ModelRegistry reg;
    registry_scan(&reg, models_dir);
    check(reg.mm3_lm.size() == 1, "one MM3 text encoder GGUF found");
    check(reg.mm3_dit.size() == 1, "one MM3 diffusion GGUF found");
    check(reg.mm3_vae.size() == 1, "one MM3 DAV safetensors found");
    check(registry_has_mm3(reg), "registry reports a complete MM3 set");
    if (!registry_has_mm3(reg)) {
        return 1;
    }
    const std::string lm_path  = reg.mm3_lm[0].path;
    const std::string dit_path = reg.mm3_dit[0].path;
    const std::string vae_path = reg.mm3_vae[0].path;

    // 2. Tokenizer out of the embedded tokenizer_json tensor
    fprintf(stderr, "\n=== tokenizer ===\n");
    {
        BPETokenizer tok;
        bool         ok = load_bpe_from_mm3_gguf(&tok, lm_path.c_str());
        check(ok, "tokenizer_json parsed and control ids validated");
        if (ok) {
            check(tok.n_vocab == MM3_AUDIO_CODE_OFFSET, "vocab is 151675 (151643 base + 32 added)");
            check(tok.merges.size() == 151387, "151387 merges");
            std::vector<int> ids =
                mm3_build_prompt_ids([&](const std::string & s) { return bpe_encode(&tok, s, false); },
                                     "a calm piano ballad", "[verse]\nhello world");
            check(ids.size() > 9, "prompt assembles past the 9 control tokens");
            check(ids.front() == MM3_IM_START, "prompt starts with <|im_start|>");
            check(ids[1] == MM3_CAPTION_START, "second token is <|caption_start|>");
            check(ids[ids.size() - 3] == MM3_LYRICS_END, "third from last is <|lyrics_end|>");
            check(ids[ids.size() - 2] == MM3_IM_END, "second from last is <|im_end|>");
            check(ids.back() == MM3_AUDIO_START, "prompt ends with <|audio_start|>");

            std::vector<int> unc = mm3_build_uncond_ids(ids);
            check(unc.front() == MM3_IM_START, "uncond keeps <|im_start|>");
            check(unc[unc.size() - 2] == MM3_IM_END, "uncond keeps <|im_end|>");
            check(unc.back() == MM3_AUDIO_START, "uncond keeps <|audio_start|>");
            bool all_cfg = true;
            for (size_t i = 1; i + 2 < unc.size(); i++) {
                if (unc[i] != MM3_AUDIO_CFG) {
                    all_cfg = false;
                }
            }
            check(all_cfg, "uncond replaces [1, n-2) with <|audio_cfg|>");
        }
    }

    // 3. VAE decoder: weight norm folding, graph, tiny decode
    fprintf(stderr, "\n=== mm3-vae ===\n");
    {
        MM3VAE vae;
        bool   ok = vae.load(vae_path.c_str());
        check(ok, "DAV decoder loads from safetensors");
        if (ok) {
            const int          T = 8;
            std::vector<float> latent((size_t) 128 * T, 0.0f);
            // a mild non-zero signal so the graph is not trivially constant
            for (size_t i = 0; i < latent.size(); i++) {
                latent[i] = 0.1f * sinf((float) i * 0.037f);
            }
            std::vector<float> audio;
            bool               dec = vae.decode(latent, T, audio);
            check(dec, "decode runs");
            check(audio.size() == (size_t) T * MM3VAE::HOP * 2, "output is T * 512 stereo samples");
            check(all_finite(audio), "output is finite");
            float peak = 0;
            for (float x : audio) {
                peak = fabsf(x) > peak ? fabsf(x) : peak;
            }
            check(peak > 0.0f && peak <= 1.0f, "output is non-silent and inside the tanh range");
            fprintf(stderr, "  (peak %.6f)\n", (double) peak);

            if (dump_vae) {
                std::string in_path  = std::string(dump_vae) + "-in.f32";
                std::string out_path = std::string(dump_vae) + "-out.f32";
                FILE *      fi       = fopen(in_path.c_str(), "wb");
                FILE *      fo       = fopen(out_path.c_str(), "wb");
                if (fi && fo) {
                    fwrite(latent.data(), sizeof(float), latent.size(), fi);
                    fwrite(audio.data(), sizeof(float), audio.size(), fo);
                    fprintf(stderr, "  (dumped %s and %s)\n", in_path.c_str(), out_path.c_str());
                }
                if (fi) {
                    fclose(fi);
                }
                if (fo) {
                    fclose(fo);
                }
            }
            vae.free();
        }
    }

    // 4. DiT + condition encoder: fixtures then a tiny forward
    fprintf(stderr, "\n=== mm3-dit ===\n");
    {
        MM3DiT dit;
        bool   ok = dit.load(dit_path.c_str());
        check(ok, "DiT + condition encoder load");
        if (ok) {
            // softmax(cond_layer_logits) * cond_layer_scale, from the port notes
            check_close(dit.mix[0], 0.06196913f, 1e-6f, "cond mix[0] (LM hidden dominates at 90.65%)");
            check_close(dit.mix[1], 0.00088394f, 1e-7f, "cond mix[1]");
            check_close(dit.mix[7], 0.00097082f, 1e-7f, "cond mix[7]");

            // freq_factors must reconstruct the file inv_freq table:
            // analytic[i] / freq_factors[i] == inv_freq[i]
            std::vector<float> ff(MM3DiT::ROTARY_DIM / 2);
            ggml_backend_tensor_get(dit.rope_ff, ff.data(), 0, ff.size() * sizeof(float));
            const float file_inv_freq[4] = { 1.0f, 0.5625f, 0.31640625f, 0.177734375f };
            float       analytic         = 1.0f;
            const float theta_scale      = powf(10000.0f, -2.0f / (float) MM3DiT::ROTARY_DIM);
            for (int i = 0; i < 4; i++) {
                char name[96];
                snprintf(name, sizeof(name), "rope inv_freq[%d] reconstructed from freq_factors", i);
                check_close(analytic / ff[(size_t) i], file_inv_freq[i], 1e-6f, name);
                analytic *= theta_scale;
            }

            // condition encoder over 8 frames
            const int          F = 8;
            std::vector<float> hidden((size_t) F * 8 * 4096);
            for (size_t i = 0; i < hidden.size(); i++) {
                hidden[i] = 0.01f * sinf((float) i * 0.0013f);
            }
            std::vector<float> cond;
            int                T_lat = 0;
            check(dit.encode_condition(hidden, F, cond, T_lat), "condition encoder runs");
            check(T_lat == mm3_latent_length(F), "latent length matches int(frames * 44100/24000 * 960/512)");
            check(cond.size() == (size_t) T_lat * 2048, "condition track is [n_latents, 2048]");
            check(all_finite(cond), "condition is finite");

            // one DiT evaluation on that window
            std::vector<float> xt((size_t) T_lat * 128), vel((size_t) T_lat * 128);
            for (size_t i = 0; i < xt.size(); i++) {
                xt[i] = 0.5f * cosf((float) i * 0.021f);
            }
            check(dit.forward(xt.data(), cond.data(), T_lat, 1, 0.3f, vel.data()), "DiT forward runs");
            check(all_finite(vel), "velocity is finite");
            double norm = 0;
            for (float x : vel) {
                norm += (double) x * x;
            }
            norm = sqrt(norm / (double) vel.size());
            check(norm > 1e-4, "velocity is non-degenerate");
            fprintf(stderr, "  (velocity rms %.6f, T_lat %d)\n", norm, T_lat);
            dit.free();
        }
    }

    // 5. Depth decoder: every sequence length the frame loop uses
    fprintf(stderr, "\n=== mm3-depth ===\n");
    {
        MM3Depth depth;
        bool     ok = depth.load(lm_path.c_str());
        check(ok, "depth decoder + audio_extra_embedding load");
        if (ok) {
            check(depth.audio_emb.size() == (size_t) 7168 * 4096, "host audio embedding table is [7168, 4096]");
            std::vector<float> seq((size_t) MM3Depth::MAX_SEQ * MM3Depth::DIM);
            for (size_t i = 0; i < seq.size(); i++) {
                seq[i] = 0.02f * sinf((float) i * 0.0007f);
            }
            bool all_ok = true;
            for (int S = 2; S <= MM3Depth::MAX_SEQ; S++) {
                std::vector<float> hid((size_t) S * MM3Depth::DIM), lg(MM3Depth::VOCAB);
                if (!depth.forward(seq.data(), S, hid.data(), lg.data()) || !all_finite(hid) || !all_finite(lg)) {
                    all_ok = false;
                }
            }
            check(all_ok, "forward runs finite for S = 2..8");

            // fused CFG frame path, one song
            const int          NC = MM3Depth::CODEBOOKS - 1;
            std::vector<float> seq_init((size_t) 2 * 2 * MM3Depth::DIM);
            for (size_t i = 0; i < seq_init.size(); i++) {
                seq_init[i] = 0.02f * cosf((float) i * 0.0009f);
            }
            std::vector<float> rnd((size_t) NC, 0.5f);
            std::vector<int>   codes((size_t) NC, -1);
            std::vector<float> cond_hid((size_t) NC * MM3Depth::DIM);
            check(depth.forward_frame(seq_init.data(), rnd.data(), 1.5f, 50, 1, codes.data(), cond_hid.data()),
                  "fused CFG frame runs");
            bool codes_ok = true;
            for (int c : codes) {
                if (c < 0 || c >= MM3Depth::VOCAB) {
                    codes_ok = false;
                }
            }
            check(codes_ok, "fused path samples 7 codes inside [0, 1024)");
            check(all_finite(cond_hid), "fused path hidden states are finite");
            depth.free();
        }
    }

    // 6. Global LM: pruned tables, prefill, one batched decode step
    if (!skip_lm) {
        fprintf(stderr, "\n=== mm3-lm ===\n");
        MM3LM lm;
        bool  ok = mm3_lm_load(&lm, lm_path.c_str(), 512, 2);
        check(ok, "pruned Qwen3 8B loads");
        if (ok) {
            check(lm.cfg.head_vocab == 16385, "lm_head_pruned is 16385 wide (stop + 16384 codes)");
            const int          H   = lm.cfg.hidden_size;
            std::vector<int>   ids = { MM3_IM_START,   MM3_CAPTION_START, MM3_CAPTION_END, MM3_LYRICS_START,
                                       MM3_LYRICS_END, MM3_IM_END,        MM3_AUDIO_START };
            std::vector<float> logits((size_t) lm.cfg.head_vocab), hidden((size_t) H);
            mm3_lm_reset_kv(&lm, 0);
            mm3_lm_forward(&lm, ids.data(), (int) ids.size(), 0, logits.data(), nullptr, hidden.data());
            check(all_finite(logits), "prefill logits are finite");
            check(all_finite(hidden), "prefill hidden state is finite");
            check(lm.kv_pos[0] == (int) ids.size(), "KV position advanced by the prompt length");

            std::vector<float> emb((size_t) H);
            mm3_lm_embed_audio_row(&lm, 1234, emb.data());
            check(all_finite(emb), "embed_tokens_audio row reads back finite");

            // batched decode over both CFG sets
            mm3_lm_reset_kv(&lm, 1);
            mm3_lm_forward(&lm, ids.data(), (int) ids.size(), 1, logits.data());
            std::vector<int>   kv_sets = { 0, 1 };
            std::vector<float> embeds((size_t) 2 * H);
            for (int b = 0; b < 2; b++) {
                memcpy(embeds.data() + (size_t) b * H, emb.data(), (size_t) H * sizeof(float));
            }
            std::vector<float> blogits((size_t) 2 * lm.cfg.head_vocab), bhidden((size_t) 2 * H);
            mm3_lm_forward_batch(&lm, kv_sets.data(), 2, embeds.data(), blogits.data(), bhidden.data());
            check(all_finite(blogits), "batched decode logits are finite");
            check(all_finite(bhidden), "batched decode hidden states are finite");
            mm3_lm_free(&lm);
        }
    } else {
        fprintf(stderr, "\n=== mm3-lm (skipped) ===\n");
    }

    fprintf(stderr, "\n%s: %d failure(s)\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
