#pragma once
// model-store.h: centralised ownership of GGML modules
//
// VRAM policy doctrine. READ THIS BEFORE CHANGING ANYTHING IN THIS FILE.
//
//   --keep-loaded (EVICT_NEVER)
//       Everything stays in VRAM. No reload, ever. The user is telling us
//       they have the budget for the full working set. Do not second-guess
//       them by adding smart eviction rules.
//
//   default (EVICT_STRICT)
//       Maximum VRAM optimisation. At most one GPU module resident at a
//       time. VAE tiles never coexist with DiT weights or LM weights by
//       construction, because only one module is ever loaded. No special
//       case needed.
//
//   coexistence groups (MiniMax Music 3 only)
//       The MM3 pipeline interleaves modules at a granularity where
//       evicting between them would thrash, so eviction consults a group
//       id instead of the bare kind:
//         AR    = { MM3_LM, MM3_DEPTH }   interleaved per frame
//         SYNTH = { MM3_DIT, MM3_VAE }    interleaved per window
//       Every ACE-Step kind is its own singleton group, so the ACE path
//       keeps exactly the "at most one module" behaviour it has today:
//       modules of different kinds still evict each other, and two
//       modules of the same kind with different keys still do too.
//
//   invariant held under BOTH policies
//       Exactly ONE LM instance for the whole process. ace_lm (generate)
//       and ace_understand must share the same LM: duplicating it would
//       waste gigabytes for no gain. This is enforced by making the
//       ModelKey identical across both pipelines (same path, same
//       max_seq, same n_kv_sets).
//
// A ModelStore holds the GGML module instances that the pipelines need
// (Qwen3 LM, DiT, VAE encoder, VAE decoder, FSQ tokenizer, etc). Pipelines
// ask the store for a module by key and return it when done. The store
// decides what stays in VRAM and what gets evicted, following the policy
// above set at creation time.
//
// Keys
//   A module is uniquely identified by (kind, path, extras). Two requires
//   with the same key return the same instance. Two requires with different
//   extras (for instance two DiTs with different adapters) are two distinct
//   modules. The LM key deliberately fixes n_kv_sets at 2 * max_batch for
//   both ace_lm and ace_understand so they share one instance.
//
// Refcounting
//   Each module has a refcount. require increments it, release decrements.
//   In EVICT_STRICT, a module with refcount > 0 cannot be evicted: a
//   conflicting require is a programming error (asserts). This catches
//   accidental overlap between modules that must not coexist.
//
// Thread safety
//   All public entry points take a single mutex. Load / unload / hit
//   decisions are serialised. Compute itself runs outside the lock.

#include "bpe.h"
#include "cond-enc.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "metadata-fsm.h"
#include "mm3-depth.h"
#include "mm3-dit.h"
#include "mm3-lm.h"
#include "mm3-vae.h"
#include "qwen3-enc.h"
#include "qwen3-lm.h"
#include "vae-enc.h"
#include "vae.h"

#include <cstddef>
#include <string>

struct ModelStore;

enum ModelKind {
    MODEL_LM,         // Qwen3LM        from acestep-5Hz-lm-*.gguf
    MODEL_TEXT_ENC,   // Qwen3GGML      from Qwen3-Embedding-*.gguf
    MODEL_COND_ENC,   // CondGGML       from acestep-v15-*.gguf (cond_enc.*)
    MODEL_DIT,        // DiTGGML        from acestep-v15-*.gguf
    MODEL_VAE_ENC,    // VAEEncoder     from vae.gguf (encoder.*)
    MODEL_VAE_DEC,    // VAEGGML        from vae.gguf (decoder.*)
    MODEL_FSQ_TOK,    // TokGGML        from acestep-v15-*.gguf (tokenizer.*)
    MODEL_FSQ_DETOK,  // DetokGGML      from acestep-v15-*.gguf (detokenizer.*)

    // MiniMax Music 3. The LM and the depth decoder share one GGUF (the
    // text encoder file), the DiT and the condition encoder share another,
    // and the DAV decoder is a safetensors file.
    MODEL_MM3_LM,     // MM3LM          from minimax_music3_text_encoder_*.gguf
    MODEL_MM3_DEPTH,  // MM3Depth       from the same text encoder GGUF
    MODEL_MM3_DIT,    // MM3DiT         from MiniMax-Music3-*.gguf
    MODEL_MM3_VAE,    // MM3VAE         from minimax_music3_dav.safetensors
};

struct ModelKey {
    ModelKind   kind;
    std::string path;  // GGUF path the module is loaded from
    // LM-only extras (ignored for other kinds):
    int         max_seq;    // KV cache length
    int         n_kv_sets;  // number of KV sets (1 or 2*max_batch with CFG)
    // DiT-only extras (ignored for other kinds):
    std::string adapter_path;   // "" when no adapter
    float       adapter_scale;  // 1.0f default, significant when adapter_path is set
};

enum EvictPolicy {
    EVICT_STRICT,  // default: at most one GPU module resident at a time
    EVICT_NEVER,   // --keep-loaded: never evict, accumulate
};

// DiT metadata cached on the CPU: needed by text encoding and T resolution
// before the DiT itself is loaded on the GPU.
struct DiTMeta {
    DiTGGMLConfig      cfg;
    std::vector<float> silence_full;   // [15000, 64] f32, from silence_latent tensor
    std::vector<float> null_cond_cpu;  // [hidden_size] f32, empty when the model has none
    bool               is_turbo;
};

ModelStore * store_create(EvictPolicy policy);
void         store_free(ModelStore * s);

// Typed GPU module accessors. Each returns a pointer owned by the store;
// never free it yourself. Returns NULL on load failure.
//
// After require, the module stays resident with a refcount > 0 until the
// matching release. In EVICT_STRICT, require evicts every other GPU module
// whose refcount is zero; if any conflicting module has refcount > 0 the
// store aborts (a programming error in the caller).
Qwen3LM *    store_require_lm(ModelStore * s, const ModelKey & k);
Qwen3GGML *  store_require_text_enc(ModelStore * s, const ModelKey & k);
CondGGML *   store_require_cond_enc(ModelStore * s, const ModelKey & k);
DiTGGML *    store_require_dit(ModelStore * s, const ModelKey & k);
VAEEncoder * store_require_vae_enc(ModelStore * s, const ModelKey & k);
VAEGGML *    store_require_vae_dec(ModelStore * s, const ModelKey & k);
TokGGML *    store_require_fsq_tok(ModelStore * s, const ModelKey & k);
DetokGGML *  store_require_fsq_detok(ModelStore * s, const ModelKey & k);

// MiniMax Music 3 modules. MM3_LM reads max_seq / n_kv_sets from the key
// exactly like MODEL_LM does; the other three ignore the extras.
MM3LM *    store_require_mm3_lm(ModelStore * s, const ModelKey & k);
MM3Depth * store_require_mm3_depth(ModelStore * s, const ModelKey & k);
MM3DiT *   store_require_mm3_dit(ModelStore * s, const ModelKey & k);
MM3VAE *   store_require_mm3_vae(ModelStore * s, const ModelKey & k);

// Release decrements the refcount for the module behind this handle.
// Pass exactly the pointer returned by require. After release, the pointer
// must not be used: in EVICT_STRICT it may be unloaded immediately.
void store_release(ModelStore * s, void * handle);

// CPU-resident accessors. Loaded on first call, kept forever, never evicted.
// All small (a few MB total). Return NULL on load failure.
BPETokenizer *  store_bpe(ModelStore * s, const char * lm_path);
// MM3 tokenizer: same cache, but built from the `tokenizer_json` tensor of
// the MM3 text encoder GGUF instead of tokenizer.ggml.tokens/merges KVs.
BPETokenizer *  store_mm3_bpe(ModelStore * s, const char * lm_path);
const float *   store_silence(ModelStore * s, const char * dit_path);
MetadataFSM *   store_fsm(ModelStore * s, const char * lm_path, int vocab_size);
const DiTMeta * store_dit_meta(ModelStore * s, const char * dit_path);

// Observability: sum of currently resident GPU module weight buffers, and
// the count of loaded GPU modules. Used by test-model-store to assert
// eviction policy invariants.
size_t store_vram_bytes(const ModelStore * s);
int    store_gpu_module_count(const ModelStore * s);

// RAII helper. Builds on top of store_release, nothing else.
struct ModelHandle {
    ModelStore * store;
    void *       ptr;

    ModelHandle(ModelStore * s, void * p) : store(s), ptr(p) {}

    ~ModelHandle() {
        if (store && ptr) {
            store_release(store, ptr);
        }
    }

    // non-copyable, movable
    ModelHandle(const ModelHandle &)             = delete;
    ModelHandle & operator=(const ModelHandle &) = delete;

    ModelHandle(ModelHandle && o) noexcept : store(o.store), ptr(o.ptr) {
        o.store = nullptr;
        o.ptr   = nullptr;
    }
};
