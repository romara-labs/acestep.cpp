#pragma once
// progress.h: process-wide generation progress hook
//
// Pipelines report coarse stage progress (stage label + cur/total) from
// their inner loops, at the same boundaries where cancel is polled. The
// hook is installed by the embedder right before a job runs and cleared
// after; CLI tools never install it and progress_report() is a no-op.
//
// Contract: progress_set / progress_report are only ever called from the
// single worker thread that runs the pipelines, so the hook state needs
// no synchronization.
//
// Stage labels (stable, chosen by the pipelines):
//   "encode"         VAE encode tiles (source / timbre / understand)
//   "vae"            VAE decode tiles
//   "dit"            DiT denoise (steps or windows)
//   "lm-metadata"    LM phase 1 (metadata + lyrics tokens)
//   "lm-codes"       LM phase 2 (audio code tokens)
//   "lm-understand"  LM understand decode
//   "ar"             MiniMax Music 3 autoregressive frames

typedef void (*progress_fn)(void * data, const char * stage, int cur, int total);

inline progress_fn g_progress_fn   = nullptr;
inline void *      g_progress_data = nullptr;

inline void progress_set(progress_fn fn, void * data) {
    g_progress_fn   = fn;
    g_progress_data = data;
}

// Report position cur/total inside a stage. cur is clamped to [0, total]
// by the caller convention; a total <= 0 means "stage finished" and is
// reported as 100%.
inline void progress_report(const char * stage, int cur, int total) {
    if (g_progress_fn) {
        g_progress_fn(g_progress_data, stage, cur, total);
    }
}
