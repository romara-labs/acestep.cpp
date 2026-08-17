#!/bin/bash
# Download pre-quantized ACE-Step GGUF models from HuggingFace
#
# Usage: ./models.sh [options]
#   default:    Q8_0 turbo essentials (text-encoder + LM-4B + DiT-turbo + VAE)
#   --all:      all models, all quants
#   --quant X:  use quant X (Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16)
#   --lm SIZE:  LM size (0.6B, 1.7B, 4B, default: 4B)
#   --sft:      include SFT DiT variant
#   --base:     include base DiT variant
#   --shifts:   include shift1/shift3/continuous DiT variants
#   --mm3:      MiniMax Music 3 instead of ACE-Step (3 files, ComfyUI layout)

set -eu

REPO="Serveurperso/ACE-Step-1.5-GGUF"
DIR="models"
QUANT="Q8_0"
LM_SIZE="4B"
ALL=0
SFT=0
BASE=0
SHIFTS=0
MM3=0

while [ $# -gt 0 ]; do
    case "$1" in
        --all)    ALL=1 ;;
        --quant)  QUANT="$2"; shift ;;
        --lm)     LM_SIZE="$2"; shift ;;
        --sft)    SFT=1 ;;
        --base)   BASE=1 ;;
        --shifts) SHIFTS=1 ;;
        --mm3)    MM3=1 ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$DIR"

dl() {
    local file="$1"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file"
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
}

# MiniMax Music 3: three files in the ComfyUI component layout that
# registry_scan looks for under --models.
#   text_encoders/    global LM (Qwen3 8B, pruned embeddings) + RVQ depth
#                     decoder + the embedded HF tokenizer.json
#   diffusion_models/ flow matching DiT + condition encoder
#   vae/              DAV decoder (safetensors, weight norm folded at load)
# The DAV file is the authoritative one from Comfy-Org; the two GGUFs are
# quantizations of that repo's *_pruned_* / dit safetensors.
if [ "$MM3" = 1 ]; then
    MM3_GGUF_REPO="molbal/Minimax-Music3-GGUF"
    MM3_VAE_REPO="Comfy-Org/MiniMax-Music-3"
    case "$QUANT" in
        Q4_0|Q4_K_M) MM3_QUANT="Q4_0" ;;
        BF16)        MM3_QUANT="BF16" ;;
        *)           MM3_QUANT="Q8_0" ;;
    esac

    mkdir -p "$DIR/text_encoders" "$DIR/diffusion_models" "$DIR/vae"

    dl_mm3() {
        local repo="$1" file="$2" subdir="$3"
        local base
        base="$(basename "$file")"
        if [ -f "$DIR/$subdir/$base" ]; then
            echo "[OK] $subdir/$base"
            return
        fi
        echo "[Download] $subdir/$base <- $repo"
        hf download --quiet "$repo" "$file" --local-dir "$DIR/$subdir"
        # flatten: the vae file arrives under its repo subdirectory
        if [ "$file" != "$base" ] && [ -f "$DIR/$subdir/$file" ]; then
            mv "$DIR/$subdir/$file" "$DIR/$subdir/$base"
        fi
    }

    # the DiT GGUF ships BF16 rather than Q8_CR at BF16 quant
    if [ "$MM3_QUANT" = "BF16" ]; then
        dl_mm3 "$MM3_GGUF_REPO" "minimax_music3_dit_BF16.gguf" "diffusion_models"
    else
        dl_mm3 "$MM3_GGUF_REPO" "minimax_music3_dit_${MM3_QUANT}.gguf" "diffusion_models"
    fi
    dl_mm3 "$MM3_GGUF_REPO" "minimax_music3_text_encoder_pruned_${MM3_QUANT}.gguf" "text_encoders"
    dl_mm3 "$MM3_VAE_REPO" "vae/minimax_music3_dav.safetensors" "vae"

    echo "[Done] MiniMax Music 3 ready in $DIR/"
    echo "[Done] Run: ./build/ace-synth --models $DIR --caption \"...\" --lyrics \"...\" --out song.mp3"
    exit 0
fi

# Resolve quant to best available for each model type.
# Matches quantize.sh matrix exactly:
#   Embedding/LM-small: BF16, Q8_0
#   LM-4B:              BF16, Q5_K_M, Q6_K, Q8_0  (Q4_K_M breaks audio codes)
#   DiT:                BF16, Q4_K_M, Q5_K_M, Q6_K, Q8_0
# Order: Q4_K_M < Q5_K_M < Q6_K < Q8_0 < BF16
# If requested quant unavailable, picks next larger available.
resolve_quant() {
    local requested="$1" model_type="$2"
    case "$model_type" in
        emb|lm_small)
            case "$requested" in
                BF16) echo "BF16" ;;
                *)    echo "Q8_0" ;;
            esac ;;
        lm_4B)
            case "$requested" in
                BF16)              echo "BF16" ;;
                Q8_0)              echo "Q8_0" ;;
                Q6_K)              echo "Q6_K" ;;
                Q5_K_M|Q4_K_M)    echo "Q5_K_M" ;;
                *)                 echo "Q8_0" ;;
            esac ;;
        dit)
            echo "$requested" ;;
    esac
}

# VAE is always BF16 (small, quality-critical)
dl "vae-BF16.gguf"

# Text encoder
dl "Qwen3-Embedding-0.6B-$(resolve_quant "$QUANT" emb).gguf"

# LM
if [ "$LM_SIZE" = "4B" ]; then
    dl "acestep-5Hz-lm-4B-$(resolve_quant "$QUANT" lm_4B).gguf"
else
    dl "acestep-5Hz-lm-${LM_SIZE}-$(resolve_quant "$QUANT" lm_small).gguf"
fi

# DiT turbo (always included)
dl "acestep-v15-turbo-${QUANT}.gguf"

# Optional DiT variants
if [ "$SFT" = 1 ] || [ "$ALL" = 1 ]; then
    dl "acestep-v15-sft-${QUANT}.gguf"
fi
if [ "$BASE" = 1 ] || [ "$ALL" = 1 ]; then
    dl "acestep-v15-base-${QUANT}.gguf"
fi
if [ "$SHIFTS" = 1 ] || [ "$ALL" = 1 ]; then
    dl "acestep-v15-turbo-shift1-${QUANT}.gguf"
    dl "acestep-v15-turbo-shift3-${QUANT}.gguf"
    dl "acestep-v15-turbo-continuous-${QUANT}.gguf"
fi

# --all: every model with its valid quants (matches quantize.sh)
if [ "$ALL" = 1 ]; then
    # Embedding: BF16 + Q8_0 only
    dl "Qwen3-Embedding-0.6B-BF16.gguf"

    # Small/medium LM: BF16 + Q8_0 only (too small for aggressive quant)
    for lm in 0.6B 1.7B; do
        dl "acestep-5Hz-lm-${lm}-BF16.gguf"
        dl "acestep-5Hz-lm-${lm}-Q8_0.gguf"
    done

    # Large LM: BF16 + Q5_K_M/Q6_K/Q8_0 (Q4_K_M breaks audio codes)
    for q in BF16 Q5_K_M Q6_K Q8_0; do
        dl "acestep-5Hz-lm-4B-${q}.gguf"
    done

    # DiT variants: BF16 + Q4_K_M/Q5_K_M/Q6_K/Q8_0
    for dit in turbo sft base turbo-shift1 turbo-shift3 turbo-continuous; do
        for q in BF16 Q4_K_M Q5_K_M Q6_K Q8_0; do
            dl "acestep-v15-${dit}-${q}.gguf"
        done
    done
fi

echo "[Done] Models ready in $DIR/"
