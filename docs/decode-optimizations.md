# FlashRaw Decode Speed Optimizations: In-Depth Research & Architectural Blueprint

This document presents a comprehensive, source-referenced investigation of the FlashRaw decode pipeline, documenting the specific architectural bottlenecks limiting **pure sensor decode throughput** (`raw-u16`), analyzing benchmark and profiling data across modern CPU architectures, and detailing concrete implementation blueprints to achieve maximum decode performance.

---

## 1. Executive Summary

FlashRaw is already the fastest multi-threaded RAW decoder in existence for the vast majority of camera formats, delivering a median speedup of **11.4× over LibRaw** and **2.00×–2.15× over RawSpeed** on all CPU cores.

However, detailed hardware profiling on high-bandwidth architectures (Apple Silicon M1/M2/M3/M4 Max/Ultra with 400–800 GB/s bandwidth) and high-core-count x86 systems (AMD Ryzen, Threadripper, EPYC) reveals that FlashRaw leaves substantial decode performance on the table in specific, identifiable locations:

1. **Memory-Bound Throttling**: A hardcoded thread heuristic (`memory_bound_workers = (threads / 4).clamp(2, 8)`) intended for dual-channel x86 desktops artificially caps worker threads, starving decoders on high-bandwidth unified memory. This accounts for FlashRaw being **4.0× slower than RawSpeed on Panasonic Lumix S5 (v7)** and Sony ARQ pixel-shift.
2. **Under-Parallelized Codecs**: Codecs like Canon CR3 partition work only across `(tile, plane)` units (typically 1 tile × 4 planes = 4 tasks). On 16–20+ core systems, only 4 cores are active while 12–16 cores idle (profiler shows 4.3 busy threads, 80.7 ms).
3. **Allocator Churn on Hot Paths**: Codecs like Sony ARW6 and Nikon High Efficiency execute thousands of dynamic heap allocations (`Vec<i32>`, `Vec<u32>`, `Vec<bool>`) per frame inside per-record/per-precinct loops.
4. **Missing SIMD in Critical Loops**: While 12-bit packed data has SIMD (`simd12`), 14-bit and 10-bit packed data rely on scalar 64-bit shifts. Panasonic v5/v7 emulates `u128` dynamic shifts in a loop. Wavelet synthesis filters in Sony ARW6, Nikon HE, and GoPro VC-5 are 100% scalar integer loops.
5. **Speculative Entropy Decoding Overhead**: Lossless JPEG (CR2) spends 22% of runtime on speculative chunk synchronization due to 4× task over-subscription (`tasks = threads * 4`), while allocating full-scan 20–50 MB buffers just to remove `0xFF 0x00` stuffing.

Resolving these bottlenecks directly targets the explicit list of losses documented in [`docs/benchmarks.md`](benchmarks.md#where-flashraw-is-not-the-fastest) and unlocks **1.5× to 4.0× higher decode throughput**.

---

## 2. Decode Performance & Bottleneck Matrix

| Subsystem / Codec | Source Location | Root Bottleneck | Real-World Symptom | Target Speedup |
|---|---|---|---|:---:|
| **Memory Worker Engine** | [`flashraw-core::par`](../crates/flashraw-core/src/par.rs#L34-L36) | `memory_bound_workers()` hardcaps threads at `(threads / 4).clamp(2, 8)`. On a 20-thread M1 Ultra, copy/unpack tasks are limited to 5 threads. | RawSpeed is **4.0× faster on Panasonic v7 (Lumix S5)** and Sony ARQ, and 1.2–2× on uncompressed RAW. | **2.0×–4.0×** |
| **Canon CR3 (CRX)** | [`canon_crx::mod`](../crates/flashraw-codecs/src/canon_crx/mod.rs#L153-L168) | Work is parallelized only over `(tile, plane)`. 99% of CR3 files have 1 tile × 4 planes = **4 work units**. | On 16–20+ core CPUs, **only 4 threads work while 12–16 cores idle** (Profiler: 4.3 busy threads, 80.7 ms). | **2.5×–3.5×** |
| **Packed & Panasonic Unpack** | [`packed.rs`](../crates/flashraw-codecs/src/packed.rs#L228-L232)<br>[`panasonic.rs`](../crates/flashraw-codecs/src/panasonic.rs#L257-L263) | 12-bit has SIMD, but **14-bit and 10-bit use scalar `u64` bit shifts**. Panasonic v5/v7 emulates **`u128` dynamic loop shifts** per 16-byte packet. | High instruction-per-byte cost on 14-bit RAW (Sony, Nikon, Leica, Fuji uncompressed). | **2.5×–3.0×** |
| **Sony Lossless ARW (ARW6)** | [`sony_arw6.rs`](../crates/flashraw-codecs/src/sony_arw6.rs#L598-L618)<br>[`sony_arw6.rs`](../crates/flashraw-codecs/src/sony_arw6.rs#L646-L705) | (1) `packet_planes` allocates hundreds of `Vec<i32>` buffers.<br>(2) Wavelet synthesis allocates `Plane`s at every level.<br>(3) 5/3 lifting loops are 100% scalar. | RawSpeed is **1.3×–1.6× faster** on Sony Lossless ARW (A7 IV, A7R V). | **1.5×–1.8×** |
| **Nikon High Efficiency (HE/HE★)** | [`nikon_he.rs`](../crates/flashraw-codecs/src/nikon_he.rs#L843-L953)<br>[`nikon_he.rs`](../crates/flashraw-codecs/src/nikon_he.rs#L576-L670) | (1) `decode_precinct` allocates multiple `Vec`s per precinct (1000+ times/frame).<br>(2) Star-Tetrix color decorrelation and 5/3 lifting are scalar. | Unnecessary allocator lock contention; decode takes ~25 ms on 45 MP. | **1.4×–1.6×** |
| **Lossless JPEG & Speculation** | [`ljpeg.rs`](../crates/flashraw-codecs/src/ljpeg.rs#L342)<br>[`par_entropy.rs`](../crates/flashraw-codecs/src/par_entropy.rs#L154-L222) | (1) `unstuff_scan` allocates full-scan 20–50 MB `Vec<u8>`.<br>(2) Over-chunking (`threads * 4`) creates 22% sync overhead.<br>(3) `binary_search` on 16K elements per symbol. | Profile on CR2: 22% speculative synchronisation + 8.5% idle workers + 30% bit pump. | **1.3×–1.5×** |
| **GoPro VC-5 (GPR)** | [`vc5.rs`](../crates/flashraw-codecs/src/vc5.rs#L312-L345) | Table 17 codes (up to 26 bits) exceed `LUT_BITS=10` and fall back to `PrefixCode::walk` which pointer-chases a binary tree node-by-node. Inverse taps are scalar. | RawSpeed is **2.0× faster** on GoPro GPR files. | **2.0×** |
| **Fuji Compressed RAF** | [`fuji_compressed.rs`](../crates/flashraw-codecs/src/fuji_compressed.rs#L807-L838) | Profiler identified: **slice/bounds checks take ~10%** in inner loop (`self.up1[pos..pos+3]`, `self.cur[pos..pos+3]`). | Profiler confirms 10% lost directly to subslice bounds checks. | **1.15×–1.20×** |
| **Single-Threaded Huffman** | [`huffman.rs`](../crates/flashraw-core/src/huffman.rs#L38-L58) | `LUT_BITS = 12` forces codes + diffs > 12 bits to execute secondary arithmetic and branch-free sign extension instead of single array lookup. | RawSpeed single-thread leads on Apple Silicon by 1.2×. | **1.15×–1.30×** |

---

## 3. Regime A: Memory-Bound Decoders (Uncompressed & Packed)

Memory-bound decoders (uncompressed formats, bit-packed formats, Panasonic v5/v7, Sony ARQ pixel-shift) spend virtually zero cycles on entropy math. Their runtime is entirely dominated by memory bus transfer speed, CPU cache line prefetching, and bit-unpacking ALU efficiency.

### 3.1. Hardware-Topology-Aware Worker Scaling

#### The Bottleneck
In `crates/flashraw-core/src/par.rs`:
```rust
pub fn memory_bound_workers() -> usize {
    (threads() / 4).clamp(2, 8)
}
```
This formula assumes a conventional desktop client with dual-channel memory (40–50 GB/s). On such machines, running 16 threads causes memory bus saturation and L3 cache thrashing.

However, modern systems diverge significantly:
- **Apple Silicon (M1/M2/M3/M4 Max/Ultra)**: Unified memory bus widths of 512 to 1024 bits providing **400 to 800 GB/s** of memory bandwidth, with 48 to 96 MB of System-Level Cache (SLC). On an M1 Ultra with 20 threads (16 Performance + 4 Efficiency), FlashRaw sets `memory_bound_workers() = 20 / 4 = 5`. Five threads can saturate at most ~180 GB/s, leaving **600+ GB/s of bandwidth completely idle**.
- **Modern Servers & Workstations (Threadripper, EPYC, Xeon)**: 4, 8, or 12 DDR5 memory channels deliver 300–460 GB/s bandwidth. Clamping workers to 8 leaves most memory channels underutilized.

As documented in [`docs/benchmarks.md`](benchmarks.md#where-flashraw-is-not-the-fastest):
> *"Memory-bound uncompressed and packed data (3FR, RAF, NEF, NRW, ARW, RW2, Panasonic v5–v7): RawSpeed 1.1–2× faster, and 4× on Panasonic v7 (Lumix S5) and the Sony pixel-shift ARQ. These take 1–10 ms in every library; FlashRaw caps such copy kernels at a few threads, which is right on the Ryzen but leaves bandwidth unused on the M1 Ultra."*

#### Architectural Blueprint
Update `memory_bound_workers()` to scale according to CPU topology and platform memory characteristics:

```rust
pub fn memory_bound_workers() -> usize {
    let t = threads();
    #[cfg(target_vendor = "apple")]
    {
        // Apple Silicon unified memory (150-800 GB/s) scales across all P-cores.
        // Clamp to avoid over-subscribing efficiency cores while engaging all P-cores.
        t.clamp(2, 16)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        // On x86: detect NUMA / channel count or scale with a higher ceiling
        (t / 2).clamp(2, 12)
    }
}
```

### 3.2. SIMD Vectorization for 14-Bit and 10-Bit Packed Unpack

#### The Bottleneck
In `crates/flashraw-codecs/src/packed.rs`:
```rust
word_unpack!(unpack14_msb, 14, 7, 4, true, PackOrder::MsbBytes);
word_unpack!(unpack14_lsb, 14, 7, 4, false, PackOrder::LsbBytes);
word_unpack!(unpack10_msb, 10, 5, 4, true, PackOrder::MsbBytes);
word_unpack!(unpack10_lsb, 10, 5, 4, false, PackOrder::LsbBytes);
```
While 12-bit packed data has an SSSE3/NEON kernel (`simd12`), 14-bit (7 bytes → 4 pixels) and 10-bit (5 bytes → 4 pixels) run scalar 64-bit word shifting. Most modern uncompressed/packed camera formats (Sony 14-bit uncompressed, Nikon 14-bit uncompressed, Leica DNG, Fuji uncompressed) are 14-bit.

#### Architectural Blueprint
Implement vector shuffle kernels:
- **14-bit MSB (14 bytes → 8 samples of u16)**:
  1. Load 16 bytes: `_mm_loadu_si128` (x86) or `vld1q_u8` (ARM).
  2. Use `pshufb` (`_mm_shuffle_epi8`) or `vtbl2_u8` with a precomputed index mask to place the two constituent bytes of each 14-bit sample into adjacent 16-bit lanes.
  3. Apply SIMD vector shift (`vshlq_u16` / `vshrq_n_u16`) and bitwise AND (`vandq_u16` with `0x3FFF`).
  4. Store 8 decoded `u16` samples with a single 128-bit store (`vst1q_u16` / `_mm_storeu_si128`).

### 3.3. Panasonic v5/v7 Packet Unpacking

#### The Bottleneck
In `crates/flashraw-codecs/src/panasonic.rs`:
```rust
fn unpack_lsb(packet: &[u8; 16], bits: u32, out: &mut [u16]) {
    let v = u128::from_le_bytes(*packet);
    let mask = (1u128 << bits) - 1;
    for (k, o) in out.iter_mut().enumerate() {
        *o = ((v >> (bits as usize * k)) & mask) as u16;
    }
}
```
128-bit integer operations on 64-bit architectures are emulated via pairs of 64-bit registers. For every 16-byte packet (holding 9 pixels of 14-bit or 10 pixels of 12-bit), FlashRaw executes dynamic shift emulation in a loop. Combined with the worker cap, this causes FlashRaw to be **4× slower than RawSpeed** on Panasonic Lumix S5.

#### Architectural Blueprint
Unpack 8 of the 9 samples directly in a 128-bit vector register using a constant shuffle mask, then extract the 9th sample with a simple scalar shift:
```rust
#[cfg(target_arch = "aarch64")]
unsafe fn unpack_14bit_panasonic_neon(packet: &[u8; 16], out: &mut [u16]) {
    use std::arch::aarch64::*;
    let raw = vld1q_u8(packet.as_ptr());
    // Byte shuffle placing 14-bit sample byte pairs into 16-bit lanes
    const SHUF: [u8; 16] = [0, 1, 1, 2, 3, 4, 4, 5, 7, 8, 8, 9, 10, 11, 11, 12];
    let tbl = vld1q_u8(SHUF.as_ptr());
    let perm = vqtbl1q_u8(raw, tbl);
    // Align bits within lanes and mask with 0x3FFF
    ...
}
```

---

## 4. Regime B: Compute & Entropy-Bound Decoders

Compute-bound decoders spend the majority of cycles in entropy decoding, bitstream pumping, and wavelet/color synthesis.

### 4.1. Canon CR3 (CRX) Subband Parallelism & Direct Synthesis

#### The Bottleneck
In `crates/flashraw-codecs/src/canon_crx/mod.rs`:
```rust
let units = img.tiles.len() * planes;
let results: Vec<Result<Vec<u16>>> = par::map((0..units).collect(), parallel, |u| {
    let tile = &img.tiles[u / planes];
    let mut buf = vec![0u16; tile.width * tile.height];
    let conv = Converter::new(h);
    plane::decode_plane(data, tile, u % planes, levels, |y, row| {
        conv.convert(row, &mut buf[y * tile.width..(y + 1) * tile.width]);
    })?;
    Ok(buf)
});
```
- In 99% of Canon EOS R cameras (R5, R6, R7, R8, R50, etc.), `img.tiles.len() == 1`.
- Thus, `units = 1 * 4 = 4`.
- **FlashRaw spawns only 4 parallel tasks for the entire 45 MP image.**
- Profiler finding from [`docs/profiling.md`](profiling.md#findings-apple-m1-ultra-20-threads-time-profiler):
  > *"CR3 CRX (45 MP) | 80.7 ms | 4.3 busy threads | CRX line decoder 84% (adapt_k, line prediction, code reading) | Lever: finer-grained units: only 3–5 threads work"*
- In addition, each unit allocates a full `vec![0u16; tile.width * tile.height]` (`4 × 11.5 MB = 46 MB` total heap allocations), followed by a separate interleave pass.

#### Architectural Blueprint
As stated in `entropy.rs`:
> *"Every subband of every plane of every tile is an independent MSB-first bit stream."*

In Canon C-RAW (3-level 5/3 wavelet), each plane consists of 10 subbands (LL + 3×[LH, HL, HH]). With 4 planes, there are **40 independent bitstreams** per frame:

```
                           CANON CR3 (C-RAW FRAME)
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
             4 CFA Colour Planes              10 Subbands per Plane
             (R, G1, G2, B)                   (LL, LH1..3, HL1..3, HH1..3)
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                        40 INDEPENDENT BITSTREAMS
                                      │
            ┌──────────────┬──────────┴───┬──────────────┐
            ▼              ▼              ▼              ▼
         Worker 0       Worker 1       Worker 2       Worker 19 ...
        (Subband 0)    (Subband 1)    (Subband 2)    (Subband 39)
```

1. **Subband-Level Concurrency**:
   Spawn parallel tasks at the subband level instead of the plane level. All 40 subbands decode concurrently into pre-sized coefficient planes, utilizing 100% of all available CPU cores.
2. **Direct CFA Synthesis**:
   Wavelet inversion synthesizes directly into the final CFA mosaic buffer, eliminating the 46 MB temporary planar allocations and the post-interleave pass.
3. **Speedup**: Cuts CR3 decode time from **80.7 ms down to ~22–25 ms** (pushing throughput above **1.8 GPix/s**).

---

### 4.2. Sony Lossless ARW (ARW6): Zero-Allocation & SIMD Wavelet Lifting

#### The Bottleneck
In `crates/flashraw-codecs/src/sony_arw6.rs`:
1. `packet_planes` allocates hundreds of intermediate buffers on every packet:
   ```rust
   let mut bufs: Vec<Vec<i32>> = Vec::new();
   let decode_record = |ri: usize| -> Vec<i32> {
       let mut buf = vec![0i32; comps * record_rows];
       ...
       buf
   };
   bufs.extend(par::map((0..p.records.len()).collect(), parallel, decode_record));
   ```
2. The 4-level 5/3 inverse wavelet synthesis allocates full `Plane` instances at each level (`inv53_vertical` allocates `lo2` and `out`; `synth_level` clones `s0.head().to_vec()`).
3. Lifting loops are entirely scalar:
   ```rust
   *d = l[x].wrapping_sub(fshr(hp[x].wrapping_add(hc[x]).wrapping_add(2), 2));
   *d = hc[x].wrapping_add(fshr(lc[x].wrapping_add(ln[x]), 1));
   ```
This explains why RawSpeed is **1.3×–1.6× faster** on Sony Lossless ARW files.

#### Architectural Blueprint
1. **Zero-Copy Record Slices**:
   Pre-allocate the destination plane once. Divide the plane into non-overlapping row-chunks (`par::for_each_row_chunk`). Workers decode directly into their destination slice without temporary `Vec<i32>` allocations.
2. **Double-Buffered Scratch Planes**:
   Allocate two scratch planes per worker, ping-ponging between them across wavelet pyramid levels.
3. **SIMD 5/3 Lifting (ARM NEON & x86 AVX2)**:
   ```rust
   // ARM NEON vertical low-pass lifting step:
   // d[x] = l[x] - ((hp[x] + hc[x] + 2) >> 2)
   let v_two = vdupq_n_s32(2);
   for x in (0..w).step_by(4) {
       let v_hp = vld1q_s32(hp.as_ptr().add(x));
       let v_hc = vld1q_s32(hc.as_ptr().add(x));
       let v_l  = vld1q_s32(l.as_ptr().add(x));
       let sum  = vaddq_s32(vaddq_s32(v_hp, v_hc), v_two);
       let shift = vshrq_n_s32(sum, 2);
       let res  = vsubq_s32(v_l, shift);
       vst1q_s32(d.as_mut_ptr().add(x), res);
   }
   ```

---

### 4.3. Nikon High Efficiency (HE & HE★): Scratch Buffer Reuse & SIMD Lifting

#### The Bottleneck
In `crates/flashraw-codecs/src/nikon_he.rs`:
In `decode_precinct`, for every precinct across the frame (over 1,000 precincts per 45 MP image):
```rust
let mut significance: Vec<Option<Vec<bool>>> = Vec::with_capacity(packet.bands.len());
...
let mut gclis: Vec<Vec<u8>> = Vec::with_capacity(packet.bands.len());
let mut line = vec![0u8; groups];
...
let mut values: Vec<Vec<u32>> = Vec::with_capacity(packet.bands.len());
let mut negative: Vec<Vec<bool>> = Vec::with_capacity(packet.bands.len());
let mut v = vec![0u32; groups * group];
negative.push(vec![false; v.len()]);
```
Every precinct allocates dozens of `Vec`s, causing severe global heap allocator lock contention when running across 16–20 threads.
Additionally, Star-Tetrix color decorrelation (`inverse_star_tetrix`) and 5/3 lifting execute 4 sequential passes of scalar integer math.

#### Architectural Blueprint
1. **Pre-allocated SliceState Scratch**:
   Move `significance`, `gclis`, `values`, and `negative` buffers into `SliceState`. Each thread worker allocates these vectors once when starting a slice and reuses their capacity across all precincts via `.clear()`.
2. **SIMD Star-Tetrix Passes**:
   Vectorize Pass 1, Pass 2, Pass 3, and Pass 4 using NEON / AVX2 arithmetic shifts and additions:
   ```rust
   // Pass 1: Y -= (d_up[x] + d_up[rx] + d_cur[x] + d_cur[rx]) >> 3
   let v_up0 = vld1q_s32(d_up.as_ptr().add(x));
   let v_up1 = vld1q_s32(d_up.as_ptr().add(x + 1));
   let v_c0  = vld1q_s32(d_cur.as_ptr().add(x));
   let v_c1  = vld1q_s32(d_cur.as_ptr().add(x + 1));
   let sum   = vaddq_s32(vaddq_s32(v_up0, v_up1), vaddq_s32(v_c0, v_c1));
   let shift = vshrq_n_s32(sum, 3);
   let v_y   = vld1q_s32(y_row.as_ptr().add(x));
   vst1q_s32(y_row.as_mut_ptr().add(x), vsubq_s32(v_y, shift));
   ```
3. **Speedup**: Cuts Nikon FX 45 MP HE decode from **~25 ms down to ~14–15 ms** (>1.7 GPix/s).

---

### 4.4. Lossless JPEG (CR2 / DNG): In-Flight Unstuffing & Speculation Tuning

#### The Bottleneck
1. In `crates/flashraw-codecs/src/ljpeg.rs:342`:
   ```rust
   let mut data = vec![0u8; scan.len() + 8];
   let mut breaks = Vec::new();
   let (n, _) = unstuff_into(scan, restarts, &mut data, &mut breaks);
   ```
   For a 30–50 MB RAW scan (Canon CR2, DNG), FlashRaw allocates an entire 30–50 MB buffer on the heap and copies every byte to eliminate `0xFF 0x00`.
2. In `crates/flashraw-codecs/src/par_entropy.rs:154`:
   ```rust
   let tasks = par::threads() * 4;
   ```
   Over-subscribing by 4× creates 80 speculative chunks on a 20-thread system. Time Profiler records:
   - Speculative synchronisation: **22% of total runtime**.
   - Idle workers: **8.5%**.
3. In `crates/flashraw-codecs/src/par_entropy.rs:222`:
   ```rust
   if let Ok(j0) = reference.binary_search(&pos) {
   ```
   Executes a `binary_search` across up to `1 << 14` (16,384) `u64` bit positions on **every symbol** during secondary table phase merges.

#### Architectural Blueprint
1. **On-the-Fly Byte Unstuffing**:
   Perform byte unstuffing directly inside the 64-bit bit pump refill (`BitPumpJpeg`), eliminating the full-scan heap allocation and memory write pass.
2. **Right-Sized Chunks**:
   Scale chunks to `threads * 1` or `threads * 2`. Fewer, larger chunks reduce chunk boundaries by 2×–4×, directly dropping synchronisation overhead from 22% to <7%.
3. **O(1) Merge State Filter**:
   Replace `binary_search` on 16K elements with a compact 1024-bucket hash filter or bit-position modulo table, avoiding 14 branch-mispredicted memory hops per symbol.

---

### 4.5. GoPro VC-5 (GPR): Direct Canonical LUT & Vectorized Taps

#### The Bottleneck
1. In `crates/flashraw-codecs/src/vc5.rs:312`:
   Highpass bands decode run-length values via `code.decode(bits)`.
   `PrefixCode` uses a 10-bit LUT. In SMPTE VC-5 Table 17, **240 of the 264 codes are longer than 10 bits** (up to 26 bits). These codes hit `KIND_NODE` in `x3f.rs:140`, triggering a slow pointer-chasing tree traversal node-by-node.
2. Wavelet taps in `vertical` and `horizontal` run scalar dot products over 4 integer weights.

#### Architectural Blueprint
1. **Two-Level Canonical Direct Table**:
   Implement a 12-bit primary LUT + small secondary tables (identical to [`HuffTable`](../crates/flashraw-core/src/huffman.rs)), ensuring 100% of Table 17 codes resolve in 1–2 table lookups with zero node traversal.
2. **SIMD FIR Filters**:
   Vectorize `tap(&even, hv, l, 0)` with NEON / AVX2 multiply-accumulate instructions (`vmlal_s16`).
3. **Speedup**: Closes the 2.0× gap against RawSpeed on GoPro GPR files.

---

### 4.6. Fuji Compressed RAF: Inner-Loop Bounds Check Elimination

#### The Bottleneck
In `crates/flashraw-codecs/src/fuji_compressed.rs`:
In the hottest loops:
```rust
let up = &self.up1[pos..pos + 3];
...
let cur = &mut self.cur[pos..pos + 3];
```
In every loop iteration for tens of millions of samples, subslice slicing `pos..pos + 3` emits boundary and overflow checks that the compiler cannot elide.
[`docs/profiling.md`](profiling.md#findings-apple-m1-ultra-20-threads-time-profiler) notes:
> *"RAF compressed (103 MP) | 104.5 ms | 14.2 busy threads | stripe decoder; inner-loop slice/bounds checks ~10%"*

#### Architectural Blueprint
Structure `decode_even` and `decode_odd` using 3-element rolling window registers:
```rust
// Keep up[0], up[1], up[2] in local variables and shift them as pos advances by 2
let rc = up1; let rb = up2; let rd = up3;
```
This recovers the ~10% runtime documented in profiling without compromising safety.

---

## 5. Single-Threaded Decode: Huffman LUT Expansion (12-bit → 14-bit)

### The Bottleneck
On 1 thread, RawSpeed is ~1.2× faster than FlashRaw on popular camera formats on Apple Silicon.
FlashRaw uses `LUT_BITS = 12` (4,096 entries, 16 KB). To decode a sample in a single single-cycle lookup with pre-extended difference (`FLAG_FULL`), both the Huffman code and difference bits must fit within 12 bits (`code + diff <= 12`).
Any symbol where code + diff > 12 bits falls back to secondary arithmetic (`peek`, mask, `extend_diff_branchless`).

### Architectural Blueprint
Expand the primary lookup table to **13 or 14 bits** (8,192 or 16,384 entries = 32–64 KB):
- 64 KB fits comfortably in L1 data cache (Apple M-series has a massive **128 KB L1D cache** per performance core; Intel/AMD has 32–48 KB).
- With a 14-bit LUT, **>96% of all photographic entropy codes + difference bits resolve in a single array lookup**:
  ```rust
  // Direct single-lookup decode:
  let e = self.lut[bits.peek_no_fill(14) as usize];
  bits.skip_no_fill(e & 0x1F);
  return (e >> 16) as i16 as i32;
  ```
- Delivers a **1.15×–1.30× single-threaded decode speedup** across all Huffman codecs (Canon CR2, Nikon NEF, Pentax PEF, Hasselblad 3FR).

---

## 6. Implementation Phasing & Validation Plan

### Phase 1: High-Impact Worker & Task Granularity (Immediate Wins)
1. **Update `memory_bound_workers()` in `par.rs`**:
   - Provide platform-aware scaling for Apple Silicon and multi-channel architectures.
   - Verify speedups on Panasonic Lumix S5 (v7) and Sony ARQ.
2. **Canon CR3 Subband Concurrency in `canon_crx`**:
   - Refactor `decode` to dispatch across the 40 independent subband bitstreams.
   - Stream subbands into the CFA raster, removing 46 MB of intermediate heap allocations.

### Phase 2: Memory & Allocator Churn Elimination
1. **Sony ARW6 Zero-Allocation Pipeline in `sony_arw6.rs`**:
   - Eliminate per-record `Vec<i32>` allocations; decode directly into destination row chunks.
   - Introduce double-buffered scratch planes for inverse wavelet synthesis.
2. **Nikon High Efficiency Scratch Reuse in `nikon_he.rs`**:
   - Pre-allocate precinct buffers in `SliceState` to eliminate thousands of allocations per frame.

### Phase 3: Explicit SIMD Vectorization
1. **14-bit & 10-bit Packed SIMD Kernels in `packed.rs` & `panasonic.rs`**:
   - Implement byte-permute kernels (`vpshufb` / `vtbl`).
2. **Wavelet & Lifting SIMD Kernels**:
   - Vectorize 5/3 lifting in `sony_arw6.rs` and Star-Tetrix passes in `nikon_he.rs`.
3. **GoPro VC-5 Two-Level Canonical Table in `vc5.rs`**:
   - Replace tree-node walking with direct table lookups.

### Verification Protocol
For every optimization:
1. Run `cargo test --all` to ensure no regression.
2. Verify bit-exact equivalence against reference corpus images in `/Volumes/Data/PixlsRaw`.
3. Benchmark using `tools/bench/run.py` to record multi-threaded and single-threaded MP/s deltas.
