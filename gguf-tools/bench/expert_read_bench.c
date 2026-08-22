/* Standalone O_DIRECT expert-read microbenchmark. No CUDA, no ds4 engine —
 * isolates the pure disk-read cost of one routed-expert tensor slice,
 * mirroring ds4_cuda.cu's own cuda_model_stage_read() alignment logic
 * (round down offset to 4096, round up length, read the aligned window).
 *
 * Usage: expert_read_bench <file> <tensor_byte_offset> <total_tensor_size> <n_experts> <n_reads>
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 4096u

static uint64_t round_down(uint64_t v, uint64_t a) { return v - (v % a); }
static uint64_t round_up(uint64_t v, uint64_t a) { return ((v + a - 1) / a) * a; }

static double wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <file> <tensor_offset> <tensor_size> <n_experts> <n_reads>\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    uint64_t tensor_off = strtoull(argv[2], NULL, 10);
    uint64_t tensor_size = strtoull(argv[3], NULL, 10);
    int n_experts = atoi(argv[4]);
    int n_reads = atoi(argv[5]);
    uint64_t expert_bytes = tensor_size / (uint64_t)n_experts;

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fprintf(stderr, "open O_DIRECT failed: %s (falling back to buffered)\n", strerror(errno));
        fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "open failed: %s\n", strerror(errno)); return 1; }
    }

    void *buf = NULL;
    uint64_t buf_bytes = round_up(expert_bytes + ALIGN, ALIGN) + ALIGN;
    if (posix_memalign(&buf, ALIGN, buf_bytes) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }

    srand(42);
    uint64_t total_bytes = 0;
    double total_time = 0.0;
    double min_lat = 1e9, max_lat = 0.0;
    int fail = 0;

    /* Warm-up read discarded (page cache / driver state settle) */
    {
        uint64_t off0 = tensor_off;
        uint64_t aligned_off = round_down(off0, ALIGN);
        uint64_t read_size = round_up((off0 - aligned_off) + expert_bytes, ALIGN);
        (void)pread(fd, buf, read_size, aligned_off);
    }

    for (int i = 0; i < n_reads; i++) {
        int expert_idx = rand() % n_experts;
        uint64_t off = tensor_off + (uint64_t)expert_idx * expert_bytes;
        uint64_t aligned_off = round_down(off, ALIGN);
        uint64_t delta = off - aligned_off;
        uint64_t read_size = round_up(delta + expert_bytes, ALIGN);

        double t0 = wall_sec();
        ssize_t got = pread(fd, buf, read_size, aligned_off);
        double dt = wall_sec() - t0;

        if (got != (ssize_t)read_size) {
            fprintf(stderr, "pread short/failed at expert %d: got=%zd want=%llu errno=%s\n",
                    expert_idx, got, (unsigned long long)read_size, strerror(errno));
            fail++;
            continue;
        }
        total_bytes += read_size;
        total_time += dt;
        if (dt < min_lat) min_lat = dt;
        if (dt > max_lat) max_lat = dt;
    }

    close(fd);
    free(buf);

    if (fail > 0) fprintf(stderr, "WARNING: %d/%d reads failed\n", fail, n_reads);
    double mb = (double)total_bytes / 1048576.0;
    printf("file=%s expert_bytes=%llu n_reads=%d total_MiB=%.3f total_time_s=%.4f "
           "avg_MBps=%.1f avg_lat_ms=%.4f min_lat_ms=%.4f max_lat_ms=%.4f\n",
           path, (unsigned long long)expert_bytes, n_reads - fail, mb, total_time,
           mb / total_time, (total_time / (n_reads - fail)) * 1000.0,
           min_lat * 1000.0, max_lat * 1000.0);
    return 0;
}
