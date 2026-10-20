// The shared expert of a prompt chunk as one dense GEMM (TF_EXL3_SHARED_DENSE): every row of the chunk is a member of
// the shared expert, so the grouped launches decoded its weight tiles once per 64-row member block (32 times at 2048
// rows for gate/up, 64 times for down). Here its tiles are decoded once a call into B fragments (shared_predecode), and
// shared_mma runs the GEMM over row tiles of 128 with each output's arithmetic exactly the grouped kernels': K splits
// into SK x WK chains of per_range k tiles, each chain an mma16816 chain from zero over its k tiles in order (A = the
// member rows from ldmatrix, B = the decoded fragments), the WK chains of a split added in order (sacc = c0, sacc + c1,
// ...), the splits folded from 0 (FOLD: tot = 0 + s0, tot + s1, ...; without FOLD, SK 1, the split's sum as is). So the
// shared slot's Z rows are bit for bit what grouped_mma (gate / up, FOLD) and grouped_rows (down, SK 1) write.
#pragma once

#include "experts_grouped.cuh"

namespace tf_exl3x {

constexpr int SH_ROWS = 128;        // member rows a program (8 warps: 4 row groups x 32 rows, 2 column groups)
constexpr int SH_COLS = 64;         // columns a program (4 n16 tiles)
constexpr int SH_STAGES = 3;        // stages of rows and fragments in flight

// B fragments of one matrix: tile (kt, nt) of the trellis decoded as decode_tile's (b0, b1) for each lane, stored as
// uint4 at [(kt * NTILES + nt) * 32 + lane] (the layout grouped_mma keeps in shared memory)
template <int CB, int K2>
__global__ void __launch_bounds__(256) shared_predecode_kernel(const uint32_t* __restrict__ T, int tiles,
                                                               uint4* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ti = blockIdx.x * 8 + warp;
    if (ti >= tiles) return;
    uint32_t w[Fmt<K2>::LW];
    load_words<K2>(w, T + (size_t)ti * Fmt<K2>::TW + lane, lane);
    const LaneMap<K2> map(lane);
    uint32_t b0[2], b1[2];
    decode_tile<CB, K2>(w, map, lane, b0, b1);
    out[(size_t)ti * 32 + lane] = make_uint4(b0[0], b0[1], b1[0], b1[1]);
}

// grid (N / 64 column blocks, ceil(R / 128) row tiles): a row tile's column blocks run back to back (its rows stay in
// L2), the decoded matrix stays in L2 across row tiles. Member r is Z row r * slots + slot (X row the same).
template <int KB, bool FOLD>
__global__ void __launch_bounds__(256) shared_mma_kernel(const half* __restrict__ X, const uint4* __restrict__ B,
                                                         float* __restrict__ Z, int mat, int K, int N, int P, int R,
                                                         int slots, int slot, int SK, int WK) {
    using AsT = half[SH_STAGES][SH_ROWS][KB * 16 + 8];
    using BsT = uint4[SH_STAGES][KB][4][32];
    extern __shared__ __align__(16) unsigned char sh_dyn[];
    AsT& As = *reinterpret_cast<AsT*>(sh_dyn);
    BsT& Bs = *reinterpret_cast<BsT*>(sh_dyn + sizeof(AsT));

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int rg = warp & 3, cg = warp >> 2;                  // rows 32 rg .. + 31, n16 tiles 2 cg, 2 cg + 1
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    const int nt0 = blockIdx.x * (SH_COLS / 16);
    const int row0 = blockIdx.y * SH_ROWS;
    const int NS = KT / KB;
    const int per_range = KT / (SK * WK);

    // the staged rows: 128 rows x KB * 16 halfs a stage = 128 x 2 KB chunks of 16 bytes, KB a thread;
    // the staged fragments: KB x 4 tiles x 32 lanes uint4 = KB * 128 chunks, KB / 2 a thread
    auto stage_in = [&](int st) {
        const int buf = st % SH_STAGES;
        const int k0 = st * KB * 16;
#pragma unroll
        for (int c = 0; c < KB; ++c) {
            const int idx = tid + c * 256;
            const int row = idx / (2 * KB), part = idx % (2 * KB);
            const int r = row0 + row;
            const bool ok = r < R;
            const half* src = X + ((size_t)(ok ? r : 0) * slots + slot) * K + k0 + part * 8;
            cp_async16(&As[buf][row][part * 8], src, ok);
        }
#pragma unroll
        for (int c = 0; c < KB / 2; ++c) {
            const int idx = tid + c * 256;
            const int kk = idx >> 7, nl = (idx >> 5) & 3, ln = idx & 31;
            cp_async16(&Bs[buf][kk][nl][ln], B + ((size_t)(st * KB + kk) * NTILES + nt0 + nl) * 32 + ln, true);
        }
    };

#pragma unroll
    for (int st = 0; st < SH_STAGES - 1; ++st) {
        if (st < NS) stage_in(st);
        cp_async_commit();
    }
    cp_async_wait<SH_STAGES - 2>();
    __syncthreads();

    float acc[2][4][4], sacc[2][4][4], tot[2][4][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][i][c] = sacc[m][i][c] = tot[m][i][c] = 0.f;
    int left = per_range, wk = 0;
    // ldmatrix: lane l reads row (l & 7) + 8 ((l >> 3) & 1) of a 16-row tile, columns 8 (l >> 4) of a k tile
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;
    const bool live0 = row0 + rg * 32 < R, live1 = row0 + rg * 32 + 16 < R;

    for (int st = 0; st < NS; ++st) {
        const int buf = st % SH_STAGES;
        if (st + SH_STAGES - 1 < NS) stage_in(st + SH_STAGES - 1);
        cp_async_commit();
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            uint32_t a0[4], a1[4];
            ldmatrix_x4(a0, &As[buf][rg * 32 + lrow][kk * 16 + lcol]);
            ldmatrix_x4(a1, &As[buf][rg * 32 + 16 + lrow][kk * 16 + lcol]);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint4 b = Bs[buf][kk][2 * cg + j][lane];
                const uint32_t bl[2] = {b.x, b.y}, bh[2] = {b.z, b.w};
                if (live0) {
                    mma16816(acc[0][2 * j], a0, bl);
                    mma16816(acc[0][2 * j + 1], a0, bh);
                }
                if (live1) {
                    mma16816(acc[1][2 * j], a1, bl);
                    mma16816(acc[1][2 * j + 1], a1, bh);
                }
            }
        }
        left -= KB;
        if (left == 0) {                                      // a chain ends: into its split's sum, in order
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        sacc[m][i][c] = wk == 0 ? acc[m][i][c] : sacc[m][i][c] + acc[m][i][c];
                        acc[m][i][c] = 0.f;
                    }
            left = per_range;
            if (++wk == WK) {                                 // a split ends
                wk = 0;
#pragma unroll
                for (int m = 0; m < 2; ++m)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int c = 0; c < 4; ++c) tot[m][i][c] = FOLD ? tot[m][i][c] + sacc[m][i][c] : sacc[m][i][c];
            }
        }
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    }
    // Z rows g8 and g8 + 8 of each 16-row tile, columns 8 i + 2 t of the warp's 32 (grouped_mma's epilogue)
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int ra = row0 + rg * 32 + m * 16 + g8, rb = ra + 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = nt0 * 16 + cg * 32 + i * 8 + 2 * t;
            if (ra < R) {
                const size_t p = (size_t)ra * slots + slot;
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + p) * N + col) = make_float2(tot[m][i][0], tot[m][i][1]);
            }
            if (rb < R) {
                const size_t p = (size_t)rb * slots + slot;
                *reinterpret_cast<float2*>(Z + ((size_t)mat * P + p) * N + col) = make_float2(tot[m][i][2], tot[m][i][3]);
            }
        }
    }
}

template <int CB>
void shared_predecode_launch(const uint32_t* T, int k2, int K, int N, uint4* out, cudaStream_t stream) {
    const int tiles = (K / 16) * (N / 16);
    const dim3 grid((unsigned)((tiles + 7) / 8));
#define TF_PD(K2_) shared_predecode_kernel<CB, K2_><<<grid, 256, 0, stream>>>(T, tiles, out)
    switch (k2) {
        case 2: TF_PD(2); break;
        case 3: TF_PD(3); break;
        case 4: TF_PD(4); break;
        case 5: TF_PD(5); break;
        case 6: TF_PD(6); break;
        case 7: TF_PD(7); break;
        case 8: TF_PD(8); break;
        case 9: TF_PD(9); break;
        case 10: TF_PD(10); break;
        case 11: TF_PD(11); break;
        case 12: TF_PD(12); break;
        case 13: TF_PD(13); break;
        case 14: TF_PD(14); break;
        case 15: TF_PD(15); break;
        case 16: TF_PD(16); break;
        default: TORCH_CHECK(false, "shared predecode: unsupported K2 ", k2);
    }
#undef TF_PD
}

inline void shared_mma_launch(const half* X, const uint4* B, float* Z, int mat, int K, int N, int P, int R, int slots,
                              int slot, int SK, int WK, bool fold, cudaStream_t stream) {
    const int KT = K / 16;
    TORCH_CHECK(KT % (SK * WK) == 0, "shared mma: K tiles do not split into the chains");
    const int chain = KT / (SK * WK);
    const int kb = chain % 4 == 0 ? 4 : chain % 2 == 0 ? 2 : 0;
    TORCH_CHECK(kb != 0, "shared mma: chains of ", chain, " k tiles take no step");
    TORCH_CHECK(N % SH_COLS == 0, "shared mma: N must be a multiple of ", SH_COLS);
    const dim3 grid((unsigned)(N / SH_COLS), (unsigned)((R + SH_ROWS - 1) / SH_ROWS));
#define TF_SM(KB_, F_)                                                                                          \
    do {                                                                                                        \
        auto fn = shared_mma_kernel<KB_, F_>;                                                                   \
        const size_t dyn = sizeof(half) * SH_STAGES * SH_ROWS * (KB_ * 16 + 8) + sizeof(uint4) * SH_STAGES * KB_ * 4 * 32; \
        C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn));          \
        fn<<<grid, 256, dyn, stream>>>(X, B, Z, mat, K, N, P, R, slots, slot, SK, WK);                          \
    } while (0)
    if (kb == 4) {
        if (fold) TF_SM(4, true); else TF_SM(4, false);
    } else {
        if (fold) TF_SM(2, true); else TF_SM(2, false);
    }
#undef TF_SM
}

}  // namespace tf_exl3x
