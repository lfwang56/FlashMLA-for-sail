// Standard CTA sparse attention selection, shared with the decode dispatcher.
#include "ppu/sparse_native_fp8_attention.cuh"

bool try_sparse_fp8_std(const flashmla::sparse_native_fp8::AttentionParams &params) {
    if (params.queries <= 0 || (params.width / 64) <= 0 || (params.width / 64) > 34 ||
        (params.dim != 512 && params.dim != 576))
        return false;
    // Low-query grids retain the producer/consumer Warp-Interleave (WI) path.
    // A full-query CTA uses a larger per-thread register budget at high grids.
    int device = -1;
    if (hggcGetDevice(&device) != hggcSuccess)
        return false;
    struct CachedDevice {
        int id = -1, cus = 0, max_threads = 0, registers = 0, cache_bytes = 0;
        size_t shared = 0;
    };
    static thread_local CachedDevice cached;
    if (cached.id != device) {
        hggcDeviceProp props;
        if (hggcGetDeviceProperties(&props, device) != hggcSuccess)
            return false;
        if (props.major != 8 || props.minor != 9)
            return false;
        cached.id = device;
        cached.cus = props.multiProcessorCount;
        cached.cache_bytes = props.l2CacheSize;
        cached.max_threads = props.maxThreadsPerBlock;
        cached.registers = props.regsPerMultiprocessor;
        cached.shared = props.sharedMemPerBlockOptin;
    }
    constexpr size_t Shared = 3 * 64 * 576 + 64 * 64 + 2176 * 4 + 11 * 64 * 4 + 256;
    // Prefix residency has an initialization cost. The short-list full-grid
    // matrix favors native, so retain it for fewer than eight N64 tiles.
    if ((params.width / 64) < 8 || (long)params.queries * (params.heads / 64) <= cached.cus ||
        cached.max_threads < 256 || cached.registers < 131072 || cached.shared < Shared)
        return false;
    if (params.dim == 512)
        flashmla_fp8_std::dispatch<256, 4>(params);
    else {
        // Output-only prefill favors four-lane gathers. For statistics-producing
        // calls, wide gathers repay their extra row passes only with enough CTAs
        // and a sparse scan of the backing pool; repeated verify queries favor
        // the narrower gather even when their allocation is large.
        constexpr int MinPoolToScanRatio = 20, MinGridWaves = 4;
        const int64_t selected_rows = int64_t(params.queries) * (params.width / 64) * 64;
        // Discount a trailing key tile so page padding cannot flip this crossover.
        const int64_t pool_rows = std::max<int64_t>(
            0, int64_t(params.kvrows) -
                   int64_t(params.queries) * flashmla::sparse_native_fp8::TileKeys);
        const bool streaming =
            pool_rows > selected_rows * MinPoolToScanRatio &&
            int64_t(params.queries) * (params.heads / 64) >= MinGridWaves * cached.cus;
        // Narrow gathers favor cache reuse; wide gathers hide scattered-load
        // latency once the pool outgrows aggregate or per-request cache reuse.
        // Guarded static kernels avoid carrying both gather address sets through
        // the attention loop. Their identical per-request hints are complementary;
        // neither kernel substitutes one query's indices for another's.
        const int requests =
            params.lse_queries > 1 ? params.queries / params.lse_queries : params.queries;
        const int64_t pool_bytes = pool_rows * params.dim;
        constexpr int MinScanCacheFootprints = 3, LargePoolCacheFootprints = 24;
        // With less selected work, the extra row passes of wide gathers cost
        // more than the latency they hide, even for a large backing allocation.
        const bool enough_scan =
            selected_rows * params.dim >= int64_t(cached.cache_bytes) * MinScanCacheFootprints;
        const bool scattered_pool =
            pool_bytes > int64_t(cached.cache_bytes) * LargePoolCacheFootprints ||
            pool_bytes * 2 > int64_t(requests) * cached.cache_bytes * 3;
        // Dense selections repeatedly touch the same small per-request pool;
        // a large batch alone must not promote them to scattered wide gathers.
        constexpr int MinPoolToHistoryRatio = 2;
        const bool sparse_history = pool_rows > int64_t(requests) * (params.width / 64) *
                                                    flashmla::sparse_native_fp8::TileKeys *
                                                    MinPoolToHistoryRatio;
        if ((params.maxima || params.lse) && params.lse_queries > 1 && params.queries >= 2 &&
            enough_scan && scattered_pool && sparse_history) {
            flashmla_fp8_std::dispatch<256, 8, 8, true>(params);
            flashmla_fp8_std::dispatch<256, 8, 16, true>(params);
            return true;
        }
        auto launch = !params.maxima && !params.lse ? flashmla_fp8_std::dispatch<256, 8, 4>
                      : streaming                   ? flashmla_fp8_std::dispatch<256, 8, 16>
                                                    : flashmla_fp8_std::dispatch<256, 8, 8>;
        launch(params);
    }
    return true;
}
