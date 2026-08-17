#pragma once
// model-registry.h: scan directories for GGUF models and adapters.
//
// Reads only GGUF headers (no weight data) to classify each file by its
// general.architecture KV into lm/dit/text-enc/vae buckets.
// Adapter entries are .safetensors files or PEFT directories.
//
// Usage:
//   ModelRegistry reg;
//   registry_scan(&reg, "./models");
//   registry_scan_adapters(&reg, "./adapters");
//   const ModelEntry * dit = registry_find(reg.dit, "acestep-v15-turbo-Q8_0.gguf");

#include "gguf.h"
#include "safetensors.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dirent.h>
#    include <sys/stat.h>
#endif

struct ModelEntry {
    std::string name;  // filename (e.g. "acestep-v15-turbo-Q8_0.gguf")
    std::string path;  // full path
};

struct AdapterEntry {
    std::string name;  // filename or directory name (e.g. "singer.safetensors" or "my-adapter")
    std::string path;  // full path (file or PEFT directory)
};

struct ModelRegistry {
    std::vector<ModelEntry>   lm;
    std::vector<ModelEntry>   dit;
    std::vector<ModelEntry>   text_enc;
    std::vector<ModelEntry>   vae;
    std::vector<AdapterEntry> adapters;

    // MiniMax Music 3 buckets. Both MM3 GGUFs declare the same
    // general.architecture ("minimax_music3"), so they are told apart by
    // tensor name probing, and the MM3 VAE is a safetensors file rather
    // than a GGUF. Kept in their own buckets so nothing about the
    // ACE-Step resolution path changes.
    std::vector<ModelEntry> mm3_lm;   // text encoder GGUF: LM + depth + tokenizer
    std::vector<ModelEntry> mm3_dit;  // diffusion GGUF: DiT + condition encoder
    std::vector<ModelEntry> mm3_vae;  // DAV decoder safetensors
};

// True when the registry holds a complete MiniMax Music 3 model set.
static bool registry_has_mm3(const ModelRegistry & reg) {
    return !reg.mm3_lm.empty() && !reg.mm3_dit.empty() && !reg.mm3_vae.empty();
}

// find an entry by name in a bucket. returns NULL if not found.
static const ModelEntry * registry_find(const std::vector<ModelEntry> & bucket, const char * name) {
    for (const auto & e : bucket) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// find an adapter entry by name. returns NULL if not found.
static const AdapterEntry * registry_find_adapter(const ModelRegistry & reg, const char * name) {
    for (const auto & e : reg.adapters) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// classify a GGUF file by reading its header.
// returns: "LM", "DiT", "Text-Enc", "VAE", "MM3-LM", "MM3-DiT",
// or "" if unrecognized.
static std::string registry_classify_gguf(const char * path) {
    struct gguf_init_params params = { true, nullptr };
    struct gguf_context *   ctx    = gguf_init_from_file(path, params);
    if (!ctx) {
        return "";
    }

    std::string arch;
    int64_t     idx = gguf_find_key(ctx, "general.architecture");
    if (idx >= 0) {
        arch = gguf_get_val_str(ctx, idx);
    }

    // MiniMax Music 3 ships both components under one architecture string,
    // so probe for a signature tensor of each, the same way ComfyUI's
    // model_detection does. Header only: no weight data is touched.
    std::string mm3;
    if (arch == "minimax_music3") {
        if (gguf_find_tensor(ctx, "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight") >= 0 &&
            gguf_find_tensor(ctx, "latent_conditioners.0.weight") >= 0) {
            mm3 = "MM3-DiT";
        } else if (gguf_find_tensor(ctx, "model.audio_decoder.projection.weight") >= 0 &&
                   gguf_find_tensor(ctx, "model.layers.0.self_attn.qkv_proj.weight") >= 0) {
            mm3 = "MM3-LM";
        }
    }

    gguf_free(ctx);
    if (!mm3.empty()) {
        return mm3;
    }

    // map GGUF architecture string to bucket name
    if (arch == "acestep-lm") {
        return "LM";
    }
    if (arch == "acestep-dit") {
        return "DiT";
    }
    if (arch == "acestep-text-enc") {
        return "Text-Enc";
    }
    if (arch == "acestep-vae") {
        return "VAE";
    }
    return "";
}

// Does this .safetensors file hold the MiniMax Music 3 DAV decoder?
// Same signature ComfyUI's sd.py uses: dec_in_proj.weight plus the still
// unfolded weight norm parametrization of the first decoder conv.
// Reads the JSON header only.
static bool registry_is_mm3_vae(const char * path) {
    STFile st = {};
    if (!st_open(&st, path)) {
        return false;
    }
    bool has_proj = false, has_conv = false;
    for (const auto & e : st.entries) {
        if (e.name == "dec_in_proj.weight") {
            has_proj = true;
        } else if (e.name == "decoder.model.0.weight_g") {
            has_conv = true;
        }
    }
    st_close(&st);
    return has_proj && has_conv;
}

// check if a string ends with a suffix
static bool str_ends_with(const std::string & s, const char * suffix) {
    size_t slen = strlen(suffix);
    return s.size() >= slen && s.compare(s.size() - slen, slen, suffix) == 0;
}

#ifdef _WIN32

// scan a directory for files matching a pattern (Windows)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

// list subdirectories (Windows)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    std::string      pattern = std::string(dir) + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && strcmp(fd.cFileName, ".") != 0 &&
            strcmp(fd.cFileName, "..") != 0) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static bool registry_is_file(const char * path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

#else

// scan a directory for files (POSIX)
static void registry_list_dir(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        // skip directories
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

// list subdirectories (POSIX)
static void registry_list_subdirs(const char * dir, std::vector<std::string> * names) {
    DIR * d = opendir(dir);
    if (!d) {
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(d)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string full = std::string(dir) + "/" + entry->d_name;
        struct stat sb;
        if (stat(full.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            names->push_back(entry->d_name);
        }
    }
    closedir(d);
}

static bool registry_is_file(const char * path) {
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISREG(sb.st_mode);
}

#endif

// path separator
#ifdef _WIN32
#    define REGISTRY_SEP "\\"
#else
#    define REGISTRY_SEP "/"
#endif

// scan one flat directory, classify each model file, append to the buckets.
// returns the number of files classified.
static int registry_scan_flat(ModelRegistry * reg, const char * models_dir) {
    std::vector<std::string> files;
    registry_list_dir(models_dir, &files);
    std::sort(files.begin(), files.end());

    int count = 0;
    for (const auto & fname : files) {
        const bool is_gguf = str_ends_with(fname, ".gguf");
        const bool is_st   = str_ends_with(fname, ".safetensors");
        if (!is_gguf && !is_st) {
            continue;
        }

        std::string full = std::string(models_dir) + REGISTRY_SEP + fname;
        std::string type;
        if (is_gguf) {
            type = registry_classify_gguf(full.c_str());
        } else if (registry_is_mm3_vae(full.c_str())) {
            // The only safetensors a models dir is expected to hold: the
            // MM3 DAV decoder. Anything else stays an adapter candidate.
            type = "MM3-VAE";
        }
        if (type.empty()) {
            if (is_gguf) {
                fprintf(stderr, "[Registry] WARNING: skipping %s (unknown architecture)\n", fname.c_str());
            }
            continue;
        }

        ModelEntry entry = { fname, full };
        if (type == "LM") {
            reg->lm.push_back(entry);
        } else if (type == "DiT") {
            reg->dit.push_back(entry);
        } else if (type == "Text-Enc") {
            reg->text_enc.push_back(entry);
        } else if (type == "VAE") {
            reg->vae.push_back(entry);
        } else if (type == "MM3-LM") {
            reg->mm3_lm.push_back(entry);
        } else if (type == "MM3-DiT") {
            reg->mm3_dit.push_back(entry);
        } else if (type == "MM3-VAE") {
            reg->mm3_vae.push_back(entry);
        }

        fprintf(stderr, "[Registry] %s -> %s\n", fname.c_str(), type.c_str());
        count++;
    }
    return count;
}

// scan a directory for model files, classify each by architecture.
// The directory itself is scanned flat (the ACE-Step layout), then the
// three ComfyUI component subdirectories are scanned too when present, so
// a MiniMax Music 3 checkout laid out as text_encoders/ diffusion_models/
// vae/ resolves from a single --models argument.
// returns true if at least one model was found.
static bool registry_scan(ModelRegistry * reg, const char * models_dir) {
    int count = registry_scan_flat(reg, models_dir);

    static const char * comfy_subdirs[] = { "text_encoders", "diffusion_models", "vae", "llm", "llms" };
    for (const char * sub : comfy_subdirs) {
        std::string path = std::string(models_dir) + REGISTRY_SEP + sub;
        count += registry_scan_flat(reg, path.c_str());
    }

    return count > 0;
}

// scan a directory for adapters.
// - .safetensors files: ComfyUI single-file format (alpha baked in)
// - subdirectories containing adapter_model.safetensors: PEFT format
// returns true if at least one adapter was found.
static bool registry_scan_adapters(ModelRegistry * reg, const char * adapters_dir) {
    int count = 0;

    // single .safetensors files
    std::vector<std::string> files;
    registry_list_dir(adapters_dir, &files);
    std::sort(files.begin(), files.end());
    for (const auto & fname : files) {
        if (!str_ends_with(fname, ".safetensors")) {
            continue;
        }
        std::string full = std::string(adapters_dir) + REGISTRY_SEP + fname;
        reg->adapters.push_back({ fname, full });
        fprintf(stderr, "[Registry] Adapter: %s (ComfyUI)\n", fname.c_str());
        count++;
    }

    // PEFT directories (contain adapter_model.safetensors)
    std::vector<std::string> subdirs;
    registry_list_subdirs(adapters_dir, &subdirs);
    std::sort(subdirs.begin(), subdirs.end());
    for (const auto & dname : subdirs) {
        std::string adapter =
            std::string(adapters_dir) + REGISTRY_SEP + dname + REGISTRY_SEP + "adapter_model.safetensors";
        if (registry_is_file(adapter.c_str())) {
            std::string full = std::string(adapters_dir) + REGISTRY_SEP + dname;
            reg->adapters.push_back({ dname, full });
            fprintf(stderr, "[Registry] Adapter: %s (PEFT)\n", dname.c_str());
            count++;
        }
    }

    return count > 0;
}
