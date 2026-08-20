/*
 * deepseek4-shrink.c -- DS4 GGUF width-shrink tool (D144W4 pre-Gate milestone).
 *
 * Reads an existing DS4 GGUF (e.g. the REAP-144 prune), slices the routed
 * expert FFN tensors (blk.N.ffn_gate_exps / ffn_up_exps / ffn_down_exps,
 * plus mtp.N.* variants) along the per-expert hidden dimension to
 * ne[1]/divide rows, re-quantizes the kept rows with the same block type
 * (using an optional importance matrix), and emits a new GGUF with every
 * other tensor copied verbatim.
 *
 * The quantization side reuses quants.c (the DS4 GGUF quantization facade,
 * byte-layout compatible with the DS4 recipes).  The dequantization side
 * is ported from ds4.c because the DS4 IQ2_XXS layout is local to this
 * engine: the 256-entry grid, sign masks, and per-group scale nibbles must
 * decode exactly like ds4_vec_dot_iq2_xxs_f32(), or the emitted tensors
 * would be garbage to the engine.  The tables are shared via
 * iq2xxs_tables.inc, generated from ds4.c.
 *
 * The output GGUF layout (metadata KV passthrough, tensor directory,
 * 32-byte alignment, offset-per-tensor data) matches deepseek4-quantize.c.
 *
 * usage: deepseek4-shrink --in MODEL.gguf --out OUT.gguf [options]
 */

#include "quants.h"

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#error "deepseek4-shrink.c currently targets POSIX systems"
#endif

#define QK_K 256

#define DS4_KV_QUANTIZE_IMATRIX_FILE      "quantize.imatrix.file"
#define DS4_KV_QUANTIZE_IMATRIX_DATASET   "quantize.imatrix.dataset"
#define DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES "quantize.imatrix.entries_count"
#define DS4_GGUF_DEFAULT_ALIGNMENT 32
#define DS4_SHRINK_MAX_LAYERS 4096

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

static void die(const char *msg) {
    fprintf(stderr, "deepseek4-shrink: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "deepseek4-shrink: %s: %s: %s\n", what, path, strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    if (n == 0) n = 1;
    void *p = malloc(n);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    char *p = strdup(s);
    if (!p) die("out of memory");
    return p;
}

static bool str_starts(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

/* =====================================================================
 * GGUF reading (metadata + tensor directory), mirrors deepseek4-quantize.c
 */

typedef struct {
    char *name;
    int n_dims;
    int64_t ne[DS4Q_MAX_DIMS];
    ds4q_type type;
    uint64_t old_offset; /* offset into the source data section */
    size_t size;
    int64_t new_offset;  /* output plan */
    bool shrink;         /* output plan: tensor is being width-shrunk */
    bool emit;           /* output plan: tensor is present in the output file */
} tensor_meta;

typedef struct {
    char *path;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint8_t *kv_raw;
    size_t kv_raw_len;
    size_t alignment;
    int n_experts; /* deepseek4.expert_count, -1 if absent */
    int n_ff_exp;  /* deepseek4.expert_feed_forward_length, -1 if absent */
    size_t data_offset;
    tensor_meta *tensors;
} gguf_file;

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static uint32_t read_u32_le_fp(FILE *fp, const char *what) {
    uint32_t v;
    if (fread(&v, sizeof(v), 1, fp) != 1) die(what);
    return v;
}

static uint64_t read_u64_le_fp(FILE *fp, const char *what) {
    uint64_t v;
    if (fread(&v, sizeof(v), 1, fp) != 1) die(what);
    return v;
}

static uint64_t read_checked_len_fp(FILE *fp, const char *what) {
    uint64_t n = read_u64_le_fp(fp, what);
    if (n > (uint64_t)(1u << 31)) die("absurd GGUF length");
    return n;
}

static char *read_gguf_string_fp(FILE *fp) {
    uint64_t n = read_checked_len_fp(fp, "GGUF string length");
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short GGUF string read");
    s[n] = '\0';
    return s;
}

static void skip_bytes_fp(FILE *fp, uint64_t n) {
    if (fseeko(fp, (off_t)n, SEEK_CUR) != 0) die("GGUF seek failed");
}

static void skip_gguf_value_fp(FILE *fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = read_u64_le_fp(fp, "GGUF string length");
        skip_bytes_fp(fp, n);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_u32_le_fp(fp, "GGUF array type");
        uint64_t n = read_u64_le_fp(fp, "GGUF array count");
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = read_u64_le_fp(fp, "GGUF array string length");
                skip_bytes_fp(fp, len);
            }
        } else {
            size_t sz = gguf_scalar_size(elem_type);
            if (!sz) die("unsupported GGUF array type");
            skip_bytes_fp(fp, n * sz);
        }
        return;
    }
    size_t sz = gguf_scalar_size(type);
    if (!sz) die("unsupported GGUF value type");
    skip_bytes_fp(fp, sz);
}

static size_t gguf_string_size(const char *s) {
    return sizeof(uint64_t) + strlen(s);
}

/* Rewrite a UINT32 KV value in the raw KV byte blob in place.  The value is
 * the same size as before, so no record or file offsets shift.  Returns the
 * new value on success, or -1 if the key is absent or not a UINT32. */
static int64_t kv_patch_u32(uint8_t *kv, size_t kv_len, const char *key, uint32_t value) {
    size_t p = 0;
    while (p < kv_len) {
        if (kv_len - p < 8) return -1;
        uint64_t klen = 0;
        for (int b = 0; b < 8; b++) klen |= (uint64_t)kv[p + b] << (8 * b);
        p += 8;
        if (klen > kv_len - p) return -1;
        char *k = xmalloc((size_t)klen + 1);
        memcpy(k, kv + p, (size_t)klen);
        k[klen] = '\0';
        p += (size_t)klen;
        if (kv_len - p < 4) { free(k); return -1; }
        uint32_t type = 0;
        for (int b = 0; b < 4; b++) type |= (uint32_t)kv[p + b] << (8 * b);
        p += 4;
        if (strcmp(k, key) == 0) {
            if (type != GGUF_TYPE_UINT32) { free(k); return -1; }
            if (kv_len - p < 4) { free(k); return -1; }
            for (int b = 0; b < 4; b++) kv[p + b] = (uint8_t)((value >> (8 * b)) & 0xff);
            free(k);
            return (int64_t)value;
        }
        free(k);
        /* skip value by type (arrays included: tokenizer KVs precede the
         * FFN-width key and must be walked, not rejected) */
        size_t vsz = 0;
        switch (type) {
            case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: vsz = 1; break;
            case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: vsz = 2; break;
            case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: vsz = 4; break;
            case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: vsz = 8; break;
            case GGUF_TYPE_STRING: {
                if (kv_len - p < 8) return -1;
                uint64_t slen = 0;
                for (int b = 0; b < 8; b++) slen |= (uint64_t)kv[p + b] << (8 * b);
                p += 8;
                vsz = (size_t)slen;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                if (kv_len - p < 12) return -1;
                uint32_t et = 0;
                for (int b = 0; b < 4; b++) et |= (uint32_t)kv[p + b] << (8 * b);
                uint64_t n = 0;
                for (int b = 0; b < 8; b++) n |= (uint64_t)kv[p + 4 + b] << (8 * b);
                p += 12;
                if (et == GGUF_TYPE_STRING) {
                    for (uint64_t i = 0; i < n; i++) {
                        if (kv_len - p < 8) return -1;
                        uint64_t slen = 0;
                        for (int b = 0; b < 8; b++) slen |= (uint64_t)kv[p + b] << (8 * b);
                        p += 8;
                        if (slen > kv_len - p) return -1;
                        p += (size_t)slen;
                    }
                } else {
                    size_t esz = 0;
                    switch (et) {
                        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: esz = 1; break;
                        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: esz = 2; break;
                        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: esz = 4; break;
                        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: esz = 8; break;
                        default: return -1;
                    }
                    if (n > (kv_len - p) / esz) return -1;
                    p += (size_t)n * esz;
                }
                vsz = 0;
                break;
            }
            default: return -1;
        }
        if (vsz > kv_len - p) return -1;
        p += vsz;
    }
    return -1;
}

static bool is_imatrix_kv_key(const char *key) {
    return str_starts(key, "quantize.imatrix.");
}

typedef struct {
    size_t start;
    size_t end;
} byte_span;

static size_t tensor_nbytes(ds4q_type type, const int64_t *ne, int n_dims) {
    if (type == DS4Q_TYPE_I32 || type == DS4Q_TYPE_F32) {
        size_t n = 4;
        for (int j = 0; j < n_dims; j++) n *= (size_t)ne[j];
        return n;
    }
    if (type == DS4Q_TYPE_F16 || type == DS4Q_TYPE_BF16) {
        size_t n = 2;
        for (int j = 0; j < n_dims; j++) n *= (size_t)ne[j];
        return n;
    }
    int64_t bs = ds4q_block_size(type);
    size_t ts = 0;
    switch (type) {
        case DS4Q_TYPE_Q8_0:    ts = 34; break;
        case DS4Q_TYPE_Q2_K:    ts = 84; break;
        case DS4Q_TYPE_Q4_K:    ts = 144; break;
        case DS4Q_TYPE_Q8_K:    ts = 292; break;
        case DS4Q_TYPE_IQ2_XXS: ts = 66; break;
        default: die("tensor_nbytes: unsupported type");
    }
    if (bs <= 0 || ne[0] % bs != 0) die("tensor_nbytes: ne[0] not block-aligned");
    size_t row = ts * (size_t)(ne[0] / bs);
    for (int j = 1; j < n_dims; j++) row *= (size_t)ne[j];
    return row;
}

static int tensor_index(const gguf_file *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return (int)i;
    }
    return -1;
}

static gguf_file load_gguf_metadata(const char *path) {
    gguf_file g = {0};
    g.path = xstrdup(path);
    g.n_experts = -1;
    g.n_ff_exp = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || memcmp(magic, "GGUF", 4) != 0) {
        die("bad GGUF magic");
    }
    g.version = read_u32_le_fp(fp, "GGUF version");
    g.n_tensors = read_u64_le_fp(fp, "GGUF tensor count");
    g.n_kv = read_u64_le_fp(fp, "GGUF KV count");
    g.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;

    byte_span *kv_keep = xcalloc((size_t)g.n_kv, sizeof(kv_keep[0]));
    uint64_t n_kv_keep = 0;

    off_t kv_start = ftello(fp);
    if (kv_start < 0) die("GGUF ftell failed");
    for (uint64_t i = 0; i < g.n_kv; i++) {
        off_t rec_start = ftello(fp);
        char *key = read_gguf_string_fp(fp);
        uint32_t type = read_u32_le_fp(fp, "GGUF KV type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = read_u32_le_fp(fp, "GGUF alignment");
            if (a) g.alignment = a;
        } else if (strcmp(key, "deepseek4.expert_count") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                uint32_t n = read_u32_le_fp(fp, "GGUF expert count");
                if (n <= (uint32_t)INT_MAX) g.n_experts = (int)n;
            } else if (type == GGUF_TYPE_UINT64) {
                uint64_t n = read_u64_le_fp(fp, "GGUF expert count");
                if (n <= (uint64_t)INT_MAX) g.n_experts = (int)n;
            } else {
                skip_gguf_value_fp(fp, type);
            }
        } else if (strcmp(key, "deepseek4.expert_feed_forward_length") == 0 ||
                   strcmp(key, "glm-dsa.expert_feed_forward_length") == 0) {
            if (type == GGUF_TYPE_UINT32) {
                uint32_t n = read_u32_le_fp(fp, "GGUF expert FFN length");
                if (n <= (uint32_t)INT_MAX) g.n_ff_exp = (int)n;
            } else {
                skip_gguf_value_fp(fp, type);
            }
        } else {
            skip_gguf_value_fp(fp, type);
        }
        off_t rec_end = ftello(fp);
        if (rec_end < 0 || rec_end < rec_start) die("GGUF ftell failed");
        /* drop stale imatrix provenance keys; re-added when --imatrix is used */
        if (!is_imatrix_kv_key(key)) {
            kv_keep[n_kv_keep++] = (byte_span){ (size_t)(rec_start - kv_start), (size_t)(rec_end - kv_start) };
        }
        free(key);
    }
    off_t tensor_start = ftello(fp);
    if (tensor_start < 0 || tensor_start < kv_start) die("GGUF ftell failed");
    size_t kv_full_len = (size_t)(tensor_start - kv_start);
    uint8_t *kv_full = xmalloc(kv_full_len);
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die("GGUF seek failed");
    if (kv_full_len && fread(kv_full, 1, kv_full_len, fp) != kv_full_len) die("GGUF KV read failed");

    for (uint64_t i = 0; i < n_kv_keep; i++) g.kv_raw_len += kv_keep[i].end - kv_keep[i].start;
    g.kv_raw = xmalloc(g.kv_raw_len ? g.kv_raw_len : 1);
    size_t kv_pos = 0;
    for (uint64_t i = 0; i < n_kv_keep; i++) {
        size_t n = kv_keep[i].end - kv_keep[i].start;
        memcpy(g.kv_raw + kv_pos, kv_full + kv_keep[i].start, n);
        kv_pos += n;
    }
    g.n_kv = n_kv_keep;
    free(kv_full);
    free(kv_keep);
    if (fseeko(fp, tensor_start, SEEK_SET) != 0) die("GGUF seek failed");

    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->name = read_gguf_string_fp(fp);
        t->n_dims = (int)read_u32_le_fp(fp, "GGUF tensor rank");
        if (t->n_dims < 1 || t->n_dims > DS4Q_MAX_DIMS) die("bad GGUF tensor rank");
        for (int j = 0; j < t->n_dims; j++) t->ne[j] = (int64_t)read_u64_le_fp(fp, "GGUF tensor dim");
        t->type = (ds4q_type)read_u32_le_fp(fp, "GGUF tensor type");
        t->old_offset = read_u64_le_fp(fp, "GGUF tensor offset");
        t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
    }
    off_t meta_end = ftello(fp);
    if (meta_end < 0) die("GGUF ftell failed");
    g.data_offset = ds4q_pad((size_t)meta_end, g.alignment);
    fclose(fp);
    return g;
}

static void free_gguf_file(gguf_file *g) {
    free(g->path);
    free(g->kv_raw);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
    memset(g, 0, sizeof(*g));
}

/* =====================================================================
 * DS4-layout dequantization (ported from ds4.c)
 */

#include "iq2xxs_tables.inc"

/* decode: block_iq2_xxs { uint16 d; uint16 qs[QK_K/8]; } with the DS4
 * per-32-group layout: two uint32 aux per group, scale nibbles in aux32[1]. */
static int8_t iq2xxs_signed_grid[256][128][8];
static int iq2xxs_signed_grid_ready = 0;
static pthread_mutex_t iq2xxs_init_guard = PTHREAD_MUTEX_INITIALIZER;

static void iq2xxs_signed_grid_init(void) {
    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = ksigns_iq2xs[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
    iq2xxs_signed_grid_ready = 1;
}

static void iq2xxs_ensure_init(void) {
    if (iq2xxs_signed_grid_ready) return;
    pthread_mutex_lock(&iq2xxs_init_guard);
    if (!iq2xxs_signed_grid_ready) iq2xxs_signed_grid_init();
    pthread_mutex_unlock(&iq2xxs_init_guard);
}

/* dequantize one row of ncols elements from src (block type @type). */
static void dequantize_row(const uint8_t *src, float *dst, int64_t ncols, ds4q_type type) {
    switch (type) {
        case DS4Q_TYPE_F32:
            memcpy(dst, src, (size_t)ncols * 4);
            return;
        case DS4Q_TYPE_F16: {
            const uint16_t *p = (const uint16_t *)src;
            for (int64_t i = 0; i < ncols; i++) dst[i] = ds4q_f16_to_f32(p[i]);
            return;
        }
        case DS4Q_TYPE_BF16: {
            const uint16_t *p = (const uint16_t *)src;
            for (int64_t i = 0; i < ncols; i++) dst[i] = ds4q_bf16_to_f32(p[i]);
            return;
        }
        case DS4Q_TYPE_Q8_0: {
            const int64_t nb = ncols / 32;
            const uint8_t *p = src;
            for (int64_t b = 0; b < nb; b++) {
                const float d = ds4q_f16_to_f32(*(const uint16_t *)p);
                const int8_t *q = (const int8_t *)(p + 2);
                for (int j = 0; j < 32; j++) dst[b * 32 + j] = d * (float)q[j];
                p += 34;
            }
            return;
        }
        case DS4Q_TYPE_Q2_K: {
            /* DS4-custom block: uint8 scales[16], uint8 qs[64], uint16 d, uint16 dmin
             * (standard GGML puts d/dmin first; ds4 encodes scales first) */
            const int64_t nb = ncols / QK_K;
            const uint8_t *p = src;
            for (int64_t b = 0; b < nb; b++) {
                const uint8_t *scales = p + 0;
                const uint8_t *qs = p + 16;
                const float d = ds4q_f16_to_f32(*(const uint16_t *)(p + 80));
                const float dmin = ds4q_f16_to_f32(*(const uint16_t *)(p + 82));
                for (int k = 0; k < QK_K; k++) {
                    const int group = k / 16;
                    const int l = k % 16;
                    const int q_base = 32 * (group / 8) + 16 * (group & 1);
                    const int shift = ((group / 2) & 3) * 2;
                    const int q = (qs[q_base + l] >> shift) & 0x03;
                    const uint8_t sc = scales[group];
                    dst[b * QK_K + k] = d * (float)(sc & 0x0f) * (float)q - dmin * (float)(sc >> 4);
                }
                p += 84;
            }
            return;
        }
        case DS4Q_TYPE_IQ2_XXS: {
            iq2xxs_ensure_init();
            const int64_t nb = ncols / QK_K;
            const uint8_t *p = src;
            for (int64_t b = 0; b < nb; b++) {
                const float d = ds4q_f16_to_f32(*(const uint16_t *)p);
                const uint16_t *q2 = (const uint16_t *)(p + 2);
                uint32_t aux32[2];
                const uint8_t *aux8 = (const uint8_t *)aux32;
                for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
                    memcpy(aux32, q2 + 4 * ib32, 2 * sizeof(uint32_t));
                    const float scale = 0.125f * d * (float)(2u * (aux32[1] >> 28) + 1u);
                    const uint32_t base = (uint32_t)ib32 * 32u;
                    for (int l = 0; l < 4; l++) {
                        const uint32_t sign_idx = (aux32[1] >> (7 * l)) & 127u;
                        const int8_t *grid = iq2xxs_signed_grid[aux8[l]][sign_idx];
                        for (int j = 0; j < 8; j++) {
                            dst[b * QK_K + base + (uint32_t)l * 8u + (uint32_t)j] = scale * (float)grid[j];
                        }
                    }
                }
                p += 66;
            }
            return;
        }
        default:
            fprintf(stderr, "deepseek4-shrink: unsupported source type for dequantization: %s\n",
                    ds4q_type_name(type));
            exit(1);
    }
}

/* =====================================================================
 * imatrix stores: legacy ds4 .dat and GGUF-embedded ("imatrix."tensors)
 */

typedef struct {
    char *name;
    float *values;
    int64_t n_values;
} imatrix_entry;

typedef struct {
    imatrix_entry *entries;
    int n_entries;
    char *file;
    char *dataset;
    int chunks;
    bool strict;
} imatrix_store;

/* Shrink-aware imatrix lookup.  ncols_dst is the emitted row length, ncols_full
 * the source row length (== ncols_dst when only rows are sliced, e.g. gate/up).
 * A full-length vector is accepted and its prefix used when the column dim is
 * shrunk (down expert), matching the semantics of keeping the first rows. */
static const float *imatrix_find(const imatrix_store *im, const char *name,
                                 int64_t ncols_dst, int64_t ncols_full,
                                 int expert_id, int n_experts) {
    if (!im || im->n_entries == 0) return NULL;
    char tmp[4200];
    for (int pass = 0; pass < 4; pass++) {
        const char *candidate = NULL;
        if (pass == 0) {
            snprintf(tmp, sizeof(tmp), "imatrix.%.4060s", name);
            candidate = tmp;
        } else if (pass == 1) {
            candidate = name;
        } else if ((pass == 2 || pass == 3) && expert_id >= 0) {
            snprintf(tmp, sizeof(tmp), "%.4060s.expert%s%d", name, pass == 2 ? "." : "_", expert_id);
            candidate = tmp;
        } else {
            break;
        }
        for (int i = 0; i < im->n_entries; i++) {
            if (strcmp(im->entries[i].name, candidate) != 0) continue;
            const imatrix_entry *e = &im->entries[i];
            if (e->n_values == ncols_dst) return e->values;
            if (e->n_values == ncols_full) return e->values; /* prefix */
            if (expert_id >= 0 && n_experts > 0) {
                if (e->n_values == ncols_dst * (int64_t)n_experts) {
                    return e->values + (size_t)expert_id * (size_t)ncols_dst;
                }
                if (e->n_values == ncols_full * (int64_t)n_experts) {
                    return e->values + (size_t)expert_id * (size_t)ncols_full;
                }
            }
            fprintf(stderr, "deepseek4-shrink: imatrix size mismatch for %s: got %" PRId64
                    " expected %" PRId64 " or %" PRId64 "\n",
                    candidate, e->n_values, ncols_dst, ncols_full);
            exit(1);
        }
    }
    if (im->strict) {
        fprintf(stderr, "deepseek4-shrink: missing imatrix entry for %s\n", name);
        exit(1);
    }
    return NULL;
}

/* legacy ds4 .dat imatrix (same entry stream deepseek4-quantize.c reads) */
static void imatrix_load_dat(imatrix_store *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->file = xstrdup(path);
    im->strict = strict;
    im->chunks = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    int32_t n_entries;
    if (fread(&n_entries, sizeof(n_entries), 1, fp) != 1 || n_entries < 1) die("bad imatrix header");
    im->entries = xcalloc((size_t)n_entries, sizeof(im->entries[0]));
    im->n_entries = n_entries;
    for (int i = 0; i < n_entries; i++) {
        int32_t len;
        if (fread(&len, sizeof(len), 1, fp) != 1) die("bad imatrix name length");
        if (len <= 0 || len > 4096) die("bad imatrix name length");
        char *name = xmalloc((size_t)len + 1);
        if (fread(name, 1, (size_t)len, fp) != (size_t)len) die("short imatrix name read");
        name[len] = '\0';
        int32_t ncall, nval;
        if (fread(&ncall, sizeof(ncall), 1, fp) != 1) die("short imatrix calls");
        if (fread(&nval, sizeof(nval), 1, fp) != 1 || nval < 1) die("bad imatrix value count");
        float *values = xmalloc((size_t)nval * sizeof(float));
        if (fread(values, sizeof(float), (size_t)nval, fp) != (size_t)nval) die("short imatrix value read");
        if (ncall > 0) {
            for (int j = 0; j < nval; j++) values[j] /= (float)ncall;
        }
        for (int j = 0; j < nval; j++) {
            if (!isfinite(values[j])) die("non-finite imatrix value");
        }
        im->entries[i] = (imatrix_entry){ .name = name, .values = values, .n_values = nval };
    }
    /* optional trailing chunk/dataset trailer */
    int c = fgetc(fp);
    if (c != EOF) {
        ungetc(c, fp);
        int32_t chunks;
        if (fread(&chunks, sizeof(chunks), 1, fp) == 1) {
            im->chunks = chunks;
            int32_t dlen;
            if (fread(&dlen, sizeof(dlen), 1, fp) == 1 && dlen > 0 && dlen < (1 << 20)) {
                char *ds = xmalloc((size_t)dlen + 1);
                if (fread(ds, 1, (size_t)dlen, fp) == (size_t)dlen) {
                    ds[dlen] = '\0';
                    im->dataset = ds;
                } else {
                    free(ds);
                }
            }
        }
    }
    fclose(fp);
    fprintf(stderr, "loaded imatrix %s: %d entries%s%s\n",
            path, n_entries, im->dataset ? ", dataset=" : "", im->dataset ? im->dataset : "");
}

/* GGUF imatrix file: a GGUF whose tensors are named imatrix.<tensor name>.
 * (llama.cpp imatrix .dat and DS4 fused -imatrix models use this shape.) */
static void imatrix_load_gguf(imatrix_store *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->file = xstrdup(path);
    im->strict = strict;
    im->chunks = -1;
    gguf_file g = load_gguf_metadata(path);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const tensor_meta *t = &g.tensors[i];
        if (!str_starts(t->name, "imatrix.")) continue;
        if (t->n_dims != 1 || t->type != DS4Q_TYPE_F32) {
            fprintf(stderr, "deepseek4-shrink: unsupported imatrix tensor %s (dims=%d type=%s)\n",
                    t->name, t->n_dims, ds4q_type_name(t->type));
            exit(1);
        }
        float *values = xmalloc(t->size);
        FILE *fp = fopen(g.path, "rb");
        if (!fp) die_errno("open imatrix GGUF", g.path);
        if (fseeko(fp, (off_t)(g.data_offset + t->old_offset), SEEK_SET) != 0) die_errno("seek imatrix GGUF", g.path);
        if (fread(values, 1, t->size, fp) != t->size) die_errno("read imatrix tensor", g.path);
        fclose(fp);
        im->entries = xrealloc(im->entries, (size_t)(im->n_entries + 1) * sizeof(im->entries[0]));
        im->entries[im->n_entries++] = (imatrix_entry){
            .name = xstrdup(t->name + 8), .values = values, .n_values = t->ne[0] };
    }
    free_gguf_file(&g);
    if (im->n_entries == 0) die("no imatrix.* tensors found in imatrix GGUF");
    fprintf(stderr, "loaded imatrix %s: %d GGUF entries\n", path, im->n_entries);
}

static void imatrix_load(imatrix_store *im, const char *path, bool strict) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    char magic[4];
    bool is_gguf = fread(magic, 1, sizeof(magic), fp) == sizeof(magic) && memcmp(magic, "GGUF", 4) == 0;
    fclose(fp);
    if (is_gguf) {
        imatrix_load_gguf(im, path, strict);
    } else {
        imatrix_load_dat(im, path, strict);
    }
}

static void imatrix_free(imatrix_store *im) {
    for (int i = 0; i < im->n_entries; i++) {
        free(im->entries[i].name);
        free(im->entries[i].values);
    }
    free(im->entries);
    free(im->file);
    free(im->dataset);
    memset(im, 0, sizeof(*im));
}

/* =====================================================================
 * Expert tensor slicing + requantization
 */

typedef struct {
    bool is_expert;
    int layer; /* -1 if none */
    bool is_mtp;
    bool is_shared; /* dense shared expert (ffn_*_shexp), rank-2 */
    int part; /* 0 gate, 1 down, 2 up */
} expert_tensor;

static expert_tensor parse_expert_tensor(const char *name) {
    expert_tensor out = { .is_expert = false, .layer = -1, .is_mtp = false, .part = -1 };
    int layer = -1;
    char kind[16];
    int rest = 0;
    if (sscanf(name, "blk.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2
        && rest == (int)strlen(name)) {
        out.layer = layer;
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            out.is_expert = true;
            out.part = strcmp(kind, "gate") == 0 ? 0 : strcmp(kind, "down") == 0 ? 1 : 2;
        }
        return out;
    }
    if (sscanf(name, "blk.%d.ffn_%15[^_]_shexp.weight%n", &layer, kind, &rest) == 2
        && rest == (int)strlen(name)) {
        out.layer = layer;
        out.is_shared = true;
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            out.is_expert = true;
            out.part = strcmp(kind, "gate") == 0 ? 0 : strcmp(kind, "down") == 0 ? 1 : 2;
        }
        return out;
    }
    if (sscanf(name, "mtp.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2
        && rest == (int)strlen(name)) {
        out.layer = layer;
        out.is_mtp = true;
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            out.is_expert = true;
            out.part = strcmp(kind, "gate") == 0 ? 0 : strcmp(kind, "down") == 0 ? 1 : 2;
        }
        return out;
    }
    if (sscanf(name, "mtp.%d.ffn_%15[^_]_shexp.weight%n", &layer, kind, &rest) == 2
        && rest == (int)strlen(name)) {
        out.layer = layer;
        out.is_mtp = true;
        out.is_shared = true;
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            out.is_expert = true;
            out.part = strcmp(kind, "gate") == 0 ? 0 : strcmp(kind, "down") == 0 ? 1 : 2;
        }
        return out;
    }
    return out;
}

typedef struct {
    const gguf_file *in;
    int tensor_idx;
    const tensor_meta *t;
    int64_t cols_full;  /* source row width (ne[0] before column shrink) */
    int64_t cols_dst;   /* emitted row width (ne[0] after plan mutation) */
    int64_t rows_keep;  /* rows emitted per expert (ne[1] after plan mutation) */
    int64_t src_rows;   /* source rows per expert */
    int n_experts;      /* ne[2], or 1 for rank-2 shared experts */
    bool is_shared;     /* rank-2 dense shared expert (no per-expert stride) */
    bool imatrix_strict;
    const imatrix_store *imatrix;
    uint8_t *out;       /* rows_keep * n_experts rows, contiguous per expert */
    int next;
    int done;
    pthread_mutex_t lock;
} shrink_job;

static void shrink_one_expert(shrink_job *j, int xid) {
    const tensor_meta *t = j->t;
    const int64_t cols_full = j->cols_full;
    const int64_t cols_dst = j->cols_dst;
    const size_t row_bytes_full = ds4q_row_size(t->type, cols_full);
    const size_t row_elems = (size_t)cols_dst;

    const float *imat = NULL;
    if (j->imatrix && j->imatrix->n_entries > 0) {
        imat = imatrix_find(j->imatrix, t->name, cols_dst, cols_full, xid, (int)t->ne[2]);
    } else if (ds4q_requires_imatrix(t->type)) {
        fprintf(stderr, "deepseek4-shrink: %s requires an imatrix (IQ2_XXS); pass --imatrix\n", t->name);
        exit(1);
    }

    /* read + dequant the source rows [0, rows_keep) of this expert.  For a
     * column-shrunk tensor (down) only the first cols_dst elements of each
     * source row are decoded; dequantize_row() decodes a prefix by count.
     * Rank-2 shared experts are a single dense tensor: xid is always 0 and
     * there is no per-expert stride. */
    const size_t src_off = j->is_shared ? 0 : (size_t)xid * (size_t)j->src_rows * row_bytes_full;
    float *f32 = xmalloc((size_t)j->rows_keep * row_elems * sizeof(float));
    uint8_t *rowbuf = xmalloc(row_bytes_full);
    FILE *fp = fopen(j->in->path, "rb");
    if (!fp) die_errno("open GGUF", j->in->path);
    for (int64_t r = 0; r < j->rows_keep; r++) {
        const size_t abs_off = j->in->data_offset + (size_t)t->old_offset + src_off + (size_t)r * row_bytes_full;
        if (fseeko(fp, (off_t)abs_off, SEEK_SET) != 0) die_errno("seek GGUF", j->in->path);
        if (fread(rowbuf, 1, row_bytes_full, fp) != row_bytes_full) die_errno("read GGUF", j->in->path);
        dequantize_row(rowbuf, f32 + (size_t)r * row_elems, cols_dst, t->type);
    }
    fclose(fp);
    free(rowbuf);

    if (getenv("DS4_SHRINK_DEBUG") && xid == 0 && j->rows_keep > 0 && strstr(t->name, "down")) {
        uint8_t *db = xmalloc(168);
        FILE *fpd = fopen(j->in->path, "rb");
        if (fpd) {
            size_t off0 = j->in->data_offset + (size_t)t->old_offset + src_off;
            if (fseeko(fpd, (off_t)off0, SEEK_SET) == 0 && fread(db, 1, 168, fpd) == 168) {
                fprintf(stderr, "  [hex0] ");
                for (int z = 0; z < 168; z++) fprintf(stderr, "%02x ", db[z]);
                fprintf(stderr, "\n");
                uint16_t d0 = (uint16_t)(db[0] | (db[1] << 8)), dm0 = (uint16_t)(db[2] | (db[3] << 8));
                uint16_t d1 = (uint16_t)(db[84] | (db[85] << 8)), dm1 = (uint16_t)(db[86] | (db[87] << 8));
                fprintf(stderr, "  [d0] d=%04x dmin=%04x | [d1] d=%04x dmin=%04x\n", d0, dm0, d1, dm1);
            }
            fclose(fpd);
        }
        free(db);
    }
    if (getenv("DS4_SHRINK_DEBUG")) {
        double m = 0.0, mx = 0.0, s = 0.0;
        for (int64_t k = 0; k < (int64_t)j->rows_keep * row_elems; k++) {
            float v = f32[k];
            if (!isfinite(v)) { fprintf(stderr, "  DBG %s xid=%d NONFINITE at %" PRId64 "\n", t->name, xid, k); break; }
            m += fabsf(v); if (fabsf(v) > mx) mx = fabsf(v); s += v;
        }
        fprintf(stderr, "  [dbg] %s xid=%d rows=%" PRId64 " cols=%" PRId64 " mean|.|=%.4f max=%.4f sum=%.2f imat=%s\n",
                t->name, xid, j->rows_keep, cols_dst, m / (double)(j->rows_keep * row_elems),
                mx, s, imat ? "yes" : "no");
    }

    /* requantize kept rows into the output slot */
    const size_t per_expert_out = (size_t)j->rows_keep * ds4q_row_size(t->type, cols_dst);
    uint8_t *slot = j->out + (size_t)xid * per_expert_out;
    ds4q_quantize_chunk(t->type, f32, slot, 0, j->rows_keep, cols_dst, imat);
    free(f32);
}

static void *shrink_worker(void *arg) {
    shrink_job *j = arg;
    for (;;) {
        pthread_mutex_lock(&j->lock);
        int xid = j->next++;
        pthread_mutex_unlock(&j->lock);
        if (xid >= j->n_experts) break;
        shrink_one_expert(j, xid);
        pthread_mutex_lock(&j->lock);
        int done = ++j->done;
        if (done % 12 == 0 || done == j->n_experts) {
            fprintf(stderr, "  %s: %d/%d experts\n", j->t->name, done, j->n_experts);
        }
        pthread_mutex_unlock(&j->lock);
    }
    return NULL;
}

static uint8_t *shrink_expert_tensor(const gguf_file *in, int idx, int64_t divide,
                                     int n_threads, const imatrix_store *imatrix,
                                     bool imatrix_strict) {
    const tensor_meta *t = &in->tensors[idx];
    expert_tensor e = parse_expert_tensor(t->name);
    const bool is_shared = e.is_shared;
    const int ndim_ok = is_shared ? 2 : 3;
    if (t->n_dims != ndim_ok) {
        fprintf(stderr, "deepseek4-shrink: %s has rank %d, expected %d\n",
                t->name, t->n_dims, ndim_ok);
        exit(1);
    }
    if (t->ne[0] % ds4q_block_size(t->type) != 0) {
        fprintf(stderr, "deepseek4-shrink: %s: ne[0]=%" PRId64 " not block-aligned for %s\n",
                t->name, t->ne[0], ds4q_type_name(t->type));
        exit(1);
    }
    if (!is_shared && in->n_experts > 0 && t->ne[2] != in->n_experts) {
        fprintf(stderr, "deepseek4-shrink: %s: ne[2]=%" PRId64 " but expert_count=%d\n",
                t->name, t->ne[2], in->n_experts);
        exit(1);
    }
    /* the caller's plan already divided the sliced dim (rows for gate/up,
     * columns for down); the source tensor still has divide times more
     * elements along that dim per expert.  Shared experts are rank-2 dense
     * tensors with the same part-aware slicing (gate/up row-slice, down
     * column-slice) and are treated as a single logical expert. */
    const bool slice_rows = e.part != 1; /* down (part 1) is [hidden, embd]: slice columns */
    const int64_t rows_keep = t->ne[1];          /* rows emitted per expert */
    const int64_t cols_dst = t->ne[0];           /* emitted row width */
    const int64_t cols_full = slice_rows ? cols_dst : cols_dst * divide;
    const int64_t src_rows = slice_rows ? rows_keep * divide : rows_keep;
    const int64_t n_experts = is_shared ? 1 : t->ne[2];
    const size_t per_expert = (size_t)rows_keep * ds4q_row_size(t->type, cols_dst);
    uint8_t *out = xcalloc((size_t)n_experts, per_expert);
    ds4q_quantize_init(t->type);
    shrink_job job = {
        .in = in, .tensor_idx = idx, .t = t,
        .cols_full = cols_full, .cols_dst = cols_dst,
        .rows_keep = rows_keep, .src_rows = src_rows,
        .n_experts = (int)n_experts, .is_shared = is_shared,
        .imatrix_strict = imatrix_strict,
        .imatrix = imatrix, .out = out, .next = 0, .done = 0,
    };
    pthread_mutex_init(&job.lock, NULL);
    int workers = n_threads > 0 ? n_threads : 1;
    if (workers > (int)n_experts) workers = (int)n_experts;
    pthread_t *threads = xcalloc((size_t)workers, sizeof(threads[0]));
    for (int i = 1; i < workers; i++) pthread_create(&threads[i], NULL, shrink_worker, &job);
    shrink_worker(&job);
    for (int i = 1; i < workers; i++) pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&job.lock);
    free(threads);
    return out;
}

/* =====================================================================
 * Output writer
 */

static void write_u32(FILE *fp, uint32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u32 failed");
}

static void write_u64(FILE *fp, uint64_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u64 failed");
}

static void write_gguf_string(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    write_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write string failed");
}

static void write_padding(FILE *fp, size_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t chunk = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) die("write padding failed");
        n -= chunk;
    }
}

static void write_imatrix_kvs(FILE *fp, const imatrix_store *im) {
    if (!im || im->n_entries == 0) return;
    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_FILE);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, im->file);
    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES);
    write_u32(fp, GGUF_TYPE_UINT64);
    write_u64(fp, (uint64_t)im->n_entries);
    if (im->dataset) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_DATASET);
        write_u32(fp, GGUF_TYPE_STRING);
        write_gguf_string(fp, im->dataset);
    }
}

static uint64_t extra_imatrix_kv_count(const imatrix_store *im) {
    if (!im || im->n_entries == 0) return 0;
    return 2 + (im->dataset ? 1 : 0);
}

static size_t extra_imatrix_kv_size(const imatrix_store *im) {
    if (!im || im->n_entries == 0) return 0;
    size_t n = 0;
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_FILE) + 4 + gguf_string_size(im->file);
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES) + 4 + 8;
    if (im->dataset) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_DATASET) + 4 + gguf_string_size(im->dataset);
    return n;
}

/* =====================================================================
 * CLI + main
 */

static void usage(const char *argv0) {
    printf("usage: %s --in MODEL.gguf --out OUT.gguf [options]\n", argv0);
    printf("\nDS4 GGUF width-shrink: slice routed expert FFN dims to ne[1]/divide rows\n");
    printf("and re-quantize with the same block type (optionally with an imatrix),\n");
    printf("copying all other tensors verbatim.  D144W4 = --divide 4.\n\n");
    printf("options:\n");
    printf("  --in FILE             source DS4 GGUF (required)\n");
    printf("  --out FILE            output GGUF (required unless --dump/--validate)\n");
    printf("  --divide N            hidden dim divisor, default 4\n");
    printf("  --layers CSV          only shrink routed layers (e.g. 0 or 0,1,2; default: all)\n");
    printf("  --emit-selected       write only the tensors being shrunk (proof mode)\n");
    printf("  --no-mtp              skip mtp.N.* expert tensors\n");
    printf("  --no-shexp            skip ffn_*_shexp (shared-expert, always-active)\n");
    printf("                        tensors; routed experts still shrink. Requires an\n");
    printf("                        engine build with an independent shared-expert\n");
    printf("                        width field (DS4_N_FF_SHEXP)\n");
    printf("  --imatrix FILE        imatrix: legacy .dat or GGUF with imatrix.* tensors\n");
    printf("  --imatrix-strict      fail if a shrunk tensor has no matching imatrix\n");
    printf("  --threads N           expert worker count, default 4\n");
    printf("  --validate NAME       dequant-compare NAME between in and out files\n");
    printf("  --dump                print the in-file tensor table and exit\n");
    printf("  --dry-run             print the shrink plan and exit\n");
    printf("  --overwrite           replace --out if it exists\n");
    printf("  --resume              continue writing into an existing partial --out\n");
}

static void dump_tensors(const gguf_file *g, const char *label) {
    printf("== %s ==\nversion=%u tensors=%" PRIu64 " kv=%" PRIu64 " experts=%d align=%zu data_offset=%zu\n",
           label, g->version, g->n_tensors, g->n_kv, g->n_experts, g->alignment, g->data_offset);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const tensor_meta *t = &g->tensors[i];
        printf("%-52s rank=%d ne=[% " PRId64 ",% " PRId64 " %" PRId64 " %" PRId64 "] type=%-9s size=%zu off=%" PRIu64 "\n",
               t->name, t->n_dims,
               t->n_dims > 0 ? t->ne[0] : 0, t->n_dims > 1 ? t->ne[1] : 0,
               t->n_dims > 2 ? t->ne[2] : 0, t->n_dims > 3 ? t->ne[3] : 0,
               ds4q_type_name(t->type), t->size, t->old_offset);
    }
}

static void print_plan(const gguf_file *in, int64_t divide, const bool *layer_sel,
                       bool mtp, bool shexp, const imatrix_store *imatrix) {
    (void)imatrix;
    size_t in_total = 0, out_total = 0;
    int n_shrunk = 0;
    for (uint64_t i = 0; i < in->n_tensors; i++) {
        const tensor_meta *t = &in->tensors[i];
        in_total += t->size;
        expert_tensor e = parse_expert_tensor(t->name);
        bool shrink = e.is_expert && (shexp || !e.is_shared) &&
            ((!e.is_mtp && layer_sel[e.layer]) || (e.is_mtp && mtp && layer_sel[e.layer]));
        if (shrink) {
            /* down (part==1) is [hidden, embd]: slice columns; gate/up slice rows */
            int64_t n0 = t->ne[0], n1 = t->ne[1];
            if (e.part == 1) n0 = t->ne[0] / divide; else n1 = t->ne[1] / divide;
            const size_t n_exps = e.is_shared ? 1 : (size_t)t->ne[2];
            size_t out_sz = ds4q_row_size(t->type, n0) * (size_t)n1 * n_exps;
            out_total += out_sz;
            n_shrunk++;
            printf("shrink: %-52s %s %" PRId64 "x%" PRId64 "x%" PRId64 " -> %" PRId64 "x%" PRId64 "x%" PRId64 " (%zu MiB)\n",
                   t->name, ds4q_type_name(t->type), t->ne[0], t->ne[1], t->ne[2],
                   n0, n1, t->ne[2], out_sz >> 20);
        } else {
            out_total += t->size;
        }
    }
    printf("plan: %d tensors shrunk, in=%zu MiB out=%zu MiB (%.1f%% of source)\n",
           n_shrunk, in_total >> 20, out_total >> 20, 100.0 * (double)out_total / (double)in_total);
}

static void validate_tensor(const gguf_file *in, const gguf_file *out, const char *name) {
    int ii = tensor_index(in, name);
    int oi = tensor_index(out, name);
    if (ii < 0 || oi < 0) die("validate: tensor not present in both files");
    const tensor_meta *t = &in->tensors[ii];
    const tensor_meta *o = &out->tensors[oi];
    if (t->n_dims != o->n_dims) die("validate: rank changed");
    for (int j = 0; j < t->n_dims; j++) {
        if (t->ne[j] != o->ne[j]) {
            printf("validate: %s dims differ: ne[%d] %" PRId64 " vs %" PRId64 "\n", name, j, t->ne[j], o->ne[j]);
        }
    }
    if (t->type != o->type) {
        printf("validate: %s type differs: %s vs %s\n", name, ds4q_type_name(t->type), ds4q_type_name(o->type));
    }

    expert_tensor e = parse_expert_tensor(name);
    if (e.is_expert && (t->ne[1] != o->ne[1] || t->ne[0] != o->ne[0])) {
        /* shrunk tensor: compare dequantized kept rows (common column prefix;
         * down shrinks ne[0], gate/up shrink ne[1]) */
        const int64_t ncols = o->ne[0] < t->ne[0] ? o->ne[0] : t->ne[0];
        const int64_t keep = o->ne[1] < t->ne[1] ? o->ne[1] : t->ne[1];
        /* row bytes are a function of the quant type and row width; the
         * size/dims ratio would divide by ne[2]=0 for rank-2 shared experts */
        const size_t src_row_bytes = ds4q_row_size(t->type, (int64_t)t->ne[0]);
        const size_t out_row_bytes = ds4q_row_size(o->type, (int64_t)o->ne[0]);
        double mean_err = 0, mean_abs_x = 0, max_err = 0, max_abs_x = 0;
        double n = 0;
        uint8_t *rb = xmalloc(src_row_bytes);
        uint8_t *ro = xmalloc(out_row_bytes);
        float *xs = xmalloc((size_t)ncols * sizeof(float));
        float *yo = xmalloc((size_t)ncols * sizeof(float));
        FILE *fin = fopen(in->path, "rb");
        FILE *fout = fopen(out->path, "rb");
        if (!fin || !fout) die("open failure during validate");
        const int n_x = e.is_shared ? 1 : (int)t->ne[2];
        for (int xid = 0; xid < n_x; xid++) {
            const size_t xstride = e.is_shared ? 0 : (size_t)xid * (size_t)t->ne[1] * src_row_bytes;
            for (int64_t r = 0; r < keep; r++) {
                size_t off_in = in->data_offset + (size_t)t->old_offset +
                                xstride + (size_t)r * src_row_bytes;
                size_t off_out = out->data_offset + (size_t)o->old_offset +
                                 (e.is_shared ? 0 : (size_t)xid * (size_t)o->ne[1] * out_row_bytes) +
                                 (size_t)r * out_row_bytes;
                if (fseeko(fin, (off_t)off_in, SEEK_SET) != 0 || fseeko(fout, (off_t)off_out, SEEK_SET) != 0) die("validate seek");
                if (fread(rb, 1, src_row_bytes, fin) != src_row_bytes) die("validate read in");
                if (fread(ro, 1, out_row_bytes, fout) != out_row_bytes) die("validate read out");
                dequantize_row(rb, xs, ncols, t->type);
                dequantize_row(ro, yo, ncols, o->type);
                for (int64_t k = 0; k < ncols; k++) {
                    float ax = fabsf(xs[k]);
                    float err = fabsf(yo[k] - xs[k]);
                    mean_err += err; mean_abs_x += ax;
                    if (err > max_err) max_err = err;
                    if (ax > max_abs_x) max_abs_x = ax;
                    n += 1;
                }
            }
        }
        fclose(fin); fclose(fout);
        free(rb); free(ro); free(xs); free(yo);
        mean_err /= n > 0 ? n : 1;
        mean_abs_x /= n > 0 ? n : 1;
        printf("validate: %s kept-rows round-trip (%" PRId64 " elems)\n", name, (int64_t)n);
        printf("  mean_abs_err=%.6f  mean_abs_x=%.6f  rel=%.4f%%\n", mean_err, mean_abs_x,
               100.0 * mean_err / (mean_abs_x > 0 ? mean_abs_x : 1));
        printf("  max_abs_err=%.4f  max_abs_x=%.4f  rel=%.4f%%\n", max_err, max_abs_x,
               100.0 * max_err / (max_abs_x > 0 ? max_abs_x : 1));
        bool ok = mean_abs_x > 0 && mean_err / mean_abs_x < 0.30;
        printf("validate: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) exit(1);
        return;
    }
    /* not shrunk (or identical dims): byte compare */
    if (t->size != o->size) {
        printf("validate: size differs %zu vs %zu\n", t->size, o->size);
        exit(1);
    }
    size_t n = t->size;
    uint8_t *a = xmalloc(n), *b = xmalloc(n);
    FILE *fin = fopen(in->path, "rb");
    FILE *fout = fopen(out->path, "rb");
    if (!fin || !fout) die("open failure during validate");
    if (fseeko(fin, (off_t)(in->data_offset + t->old_offset), SEEK_SET) != 0 ||
        fseeko(fout, (off_t)(out->data_offset + o->old_offset), SEEK_SET) != 0) die("validate seek");
    if (fread(a, 1, n, fin) != n || fread(b, 1, n, fout) != n) die("validate read");
    fclose(fin); fclose(fout);
    size_t mism = 0;
    for (size_t k = 0; k < n; k++) if (a[k] != b[k]) mism++;
    free(a); free(b);
    printf("validate: %s byte-compare: %s (%zu mismatches of %zu)\n",
           name, mism ? "FAIL" : "PASS", mism, n);
    if (mism) exit(1);
}

static bool parse_layer_list(const char *csv, bool *sel) {
    if (!csv) return false;
    char *tmp = xstrdup(csv);
    for (char *p = tmp; p && *p;) {
        char *comma = strchr(p, ',');
        if (comma) *comma = '\0';
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || *end != '\0' || v < 0 || v >= DS4_SHRINK_MAX_LAYERS) {
            fprintf(stderr, "bad layer id in list: %s\n", p);
            exit(1);
        }
        sel[v] = true;
        p = comma ? comma + 1 : NULL;
    }
    free(tmp);
    return true;
}

int main(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL, *imatrix_path = NULL;
    const char *layers_csv = NULL, *validate_name = NULL;
    int64_t divide = 4;
    int n_threads = 4;
    bool mtp = true, shexp = true, dry_run = false, dump = false, overwrite = false, imatrix_strict = false;
    bool emit_selected = false;
    bool resume = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--in") == 0) in_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0) out_path = argv[++i];
        else if (strcmp(argv[i], "--divide") == 0) divide = atol(argv[++i]);
        else if (strcmp(argv[i], "--layers") == 0) layers_csv = argv[++i];
        else if (strcmp(argv[i], "--no-mtp") == 0) mtp = false;
        else if (strcmp(argv[i], "--no-shexp") == 0) shexp = false;
        else if (strcmp(argv[i], "--emit-selected") == 0) emit_selected = true;
        else if (strcmp(argv[i], "--resume") == 0) resume = true;
        else if (strcmp(argv[i], "--imatrix") == 0) imatrix_path = argv[++i];
        else if (strcmp(argv[i], "--imatrix-strict") == 0) imatrix_strict = true;
        else if (strcmp(argv[i], "--threads") == 0) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--validate") == 0) validate_name = argv[++i];
        else if (strcmp(argv[i], "--dump") == 0) dump = true;
        else if (strcmp(argv[i], "--dry-run") == 0) dry_run = true;
        else if (strcmp(argv[i], "--overwrite") == 0) overwrite = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown option %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!in_path) die("--in is required");
    if (divide < 1) die("--divide must be >= 1");
    if (n_threads < 1) n_threads = 1;

    bool layer_sel[DS4_SHRINK_MAX_LAYERS] = {false};
    bool has_layers = false;
    if (layers_csv) {
        has_layers = true;
        for (int i = 0; i < DS4_SHRINK_MAX_LAYERS; i++) layer_sel[i] = false;
        parse_layer_list(layers_csv, layer_sel);
    }

    gguf_file in = load_gguf_metadata(in_path);
    if (dump) {
        dump_tensors(&in, in_path);
        free_gguf_file(&in);
        return 0;
    }
    if (!has_layers) {
        for (int i = 0; i < DS4_SHRINK_MAX_LAYERS; i++) layer_sel[i] = true;
    }

    imatrix_store imatrix = {0};
    if (imatrix_path) imatrix_load(&imatrix, imatrix_path, imatrix_strict);

    if (validate_name) {
        if (!out_path) die("--validate needs --out");
        gguf_file out = load_gguf_metadata(out_path);
        validate_tensor(&in, &out, validate_name);
        free_gguf_file(&out);
        free_gguf_file(&in);
        imatrix_free(&imatrix);
        return 0;
    }

    if (dry_run) {
        print_plan(&in, divide, layer_sel, mtp, shexp, &imatrix);
        imatrix_free(&imatrix);
        free_gguf_file(&in);
        return 0;
    }
    if (!out_path) die("--out is required unless --dump/--validate");
    if (!overwrite && !resume) {
        FILE *fp = fopen(out_path, "rb");
        if (fp) { fclose(fp); die("output exists; use --overwrite"); }
    }

    /* plan output offsets (ne[1] is mutated to the shrunk row count here) */
    uint64_t off = 0;
    for (uint64_t i = 0; i < in.n_tensors; i++) {
        tensor_meta *t = &in.tensors[i];
        expert_tensor e = parse_expert_tensor(t->name);
        bool shrink = e.is_expert && (shexp || !e.is_shared) &&
                    ((!e.is_mtp && layer_sel[e.layer]) || (e.is_mtp && mtp && layer_sel[e.layer]));
        t->shrink = shrink;
        t->emit = !emit_selected || shrink;
        if (shrink) {
            if (e.part == 1) t->ne[0] = t->ne[0] / divide; /* down: [hidden, embd] */
            else t->ne[1] = t->ne[1] / divide;
        }
        t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
        if (t->emit) {
            t->new_offset = (int64_t)off;
            off += ds4q_pad(t->size, in.alignment);
        }
    }
    /* The routed-expert FFN width is stored in GGUF metadata as
     * deepseek4.expert_feed_forward_length.  A width-shrunk model must carry
     * the shrunk value (source / divide) so ds4 selects the right shape and
     * sizes the routed-FFN kernels; rewrite it in place (same-size u32). */
    {
        const char *ffn_key = "deepseek4.expert_feed_forward_length";
        int64_t patched = kv_patch_u32(in.kv_raw, in.kv_raw_len, ffn_key,
                                       (uint32_t)(in.n_ff_exp > 0 ? in.n_ff_exp / (uint32_t)divide : 0));
        if (patched >= 0) {
            fprintf(stderr, "patched %s -> %" PRId64 "\n", ffn_key, patched);
        } else {
            /* GLM-family models use glm-dsa.expert_feed_forward_length */
            patched = kv_patch_u32(in.kv_raw, in.kv_raw_len, "glm-dsa.expert_feed_forward_length",
                                   (uint32_t)(in.n_ff_exp > 0 ? in.n_ff_exp / (uint32_t)divide : 0));
            if (patched >= 0) fprintf(stderr, "patched glm-dsa.expert_feed_forward_length -> %" PRId64 "\n", patched);
        }
    }

    size_t tensor_info = 0;
    uint64_t n_emit = 0;
    for (uint64_t i = 0; i < in.n_tensors; i++) {
        const tensor_meta *t = &in.tensors[i];
        if (!t->emit) continue;
        n_emit++;
        tensor_info += gguf_string_size(t->name) + 4 + (size_t)t->n_dims * 8 + 4 + 8;
    }
    size_t meta_size = 4 + 4 + 8 + 8 + in.kv_raw_len + extra_imatrix_kv_size(&imatrix) + tensor_info;
    size_t data_offset_out = ds4q_pad(meta_size, in.alignment);

    fprintf(stderr, "plan: %" PRIu64 " tensors (%" PRIu64 " emitted), data %" PRIu64 " bytes (%.1f MiB), alignment %zu\n",
            in.n_tensors, n_emit, off, (double)off / 1048576.0, in.alignment);

    FILE *fp = NULL;
    if (resume) {
        fp = fopen(out_path, "r+b");
        if (!fp) die_errno("open output (resume)", out_path);
        if (fseek(fp, 0, SEEK_END) != 0) die("resume: fseek END failed");
        long sz = ftell(fp);
        if (sz < 0) die("resume: ftell failed");
        if ((size_t)sz < data_offset_out) die("resume: partial file shorter than metadata");
        fprintf(stderr, "resume: existing file %ld bytes, data section at %zu; skipping complete tensors\n",
                sz, data_offset_out);
    } else {
        fp = fopen(out_path, "wb");
        if (!fp) die_errno("open output", out_path);
        if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
        write_u32(fp, in.version);
        write_u64(fp, n_emit);
        write_u64(fp, in.n_kv + extra_imatrix_kv_count(&imatrix));
        if (fwrite(in.kv_raw, 1, in.kv_raw_len, fp) != in.kv_raw_len) die("write GGUF KV failed");
        write_imatrix_kvs(fp, &imatrix);
        for (uint64_t i = 0; i < in.n_tensors; i++) {
            const tensor_meta *t = &in.tensors[i];
            if (!t->emit) continue;
            write_gguf_string(fp, t->name);
            write_u32(fp, (uint32_t)t->n_dims);
            for (int j = 0; j < t->n_dims; j++) write_u64(fp, (uint64_t)t->ne[j]);
            write_u32(fp, (uint32_t)t->type);
            write_u64(fp, (uint64_t)t->new_offset);
        }
        long pos = ftell(fp);
        if (pos < 0) die("ftell failed");
        if ((size_t)pos > data_offset_out) die("GGUF metadata larger than planned");
        write_padding(fp, data_offset_out - (size_t)pos);
    }

    /* data */
    off_t file_len = 0;
    if (resume) {
        if (fseek(fp, 0, SEEK_END) != 0) die("resume: fseek END failed");
        file_len = ftell(fp);
        if (file_len < 0) die("resume: ftell failed");
    }
    for (uint64_t i = 0; i < in.n_tensors; i++) {
        const tensor_meta *t = &in.tensors[i];
        if (!t->emit) continue;
        const size_t abs_off = data_offset_out + (size_t)t->new_offset;
        const size_t padded = ds4q_pad(t->size, in.alignment);
        if (resume && (off_t)(abs_off + padded) <= file_len) {
            fprintf(stderr, "[%3" PRIu64 "/%" PRIu64 "] %s (already written)\n", i + 1, in.n_tensors, t->name);
            continue;
        }
        if (fseeko(fp, (off_t)abs_off, SEEK_SET) != 0) die_errno("seek output", out_path);
        if (t->shrink) {
            uint8_t *data = shrink_expert_tensor(&in, (int)i, divide, n_threads, &imatrix, imatrix_strict);
            size_t want = t->size;
            if (fwrite(data, 1, want, fp) != want) die_errno("write tensor", out_path);
            write_padding(fp, padded - want);
            free(data);
        } else {
            /* verbatim copy */
            FILE *fi = fopen(in.path, "rb");
            if (!fi) die_errno("open source", in.path);
            if (fseeko(fi, (off_t)(in.data_offset + t->old_offset), SEEK_SET) != 0) die_errno("seek source", in.path);
            static uint8_t buf[1 << 20];
            size_t remain = t->size;
            while (remain) {
                size_t chunk = remain < sizeof(buf) ? remain : sizeof(buf);
                if (fread(buf, 1, chunk, fi) != chunk) die_errno("read source", in.path);
                if (fwrite(buf, 1, chunk, fp) != chunk) die_errno("write output", out_path);
                remain -= chunk;
            }
            fclose(fi);
            write_padding(fp, padded - t->size);
        }
        fprintf(stderr, "[%3" PRIu64 "/%" PRIu64 "] %s (%s)\n", i + 1, in.n_tensors,
                t->name, t->shrink ? "shrunk" : "verbatim");
    }
    fclose(fp);
    fprintf(stderr, "wrote %s\n", out_path);

    free_gguf_file(&in);
    imatrix_free(&imatrix);
    return 0;
}