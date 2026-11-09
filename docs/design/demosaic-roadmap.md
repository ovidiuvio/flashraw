# Demosaic audit, results and open work

What was measured on the demosaic algorithms (Linux x86-64, 8 cores / 16 threads), what was built from it, and
what is left. Nothing here changes a default: `Demosaic::Bilinear` stays the default; `Auto`, `Ppg` and `Rcd`
are opt-in.

## Algorithm selection

`Demosaic::Auto` picks the algorithm per sensor when the file is decoded: edge-aware on Bayer and X-Trans, bilinear on
any other pattern. The choice depends only on the sensor type, so it is the same on every machine. It is edge-aware
because that has the smallest error against ground truth on noisy tiles of both sensor types and on clean X-Trans
tiles (see [benchmarks](../benchmarks.md) and the table in [bayer-demosaics](bayer-demosaics.md)); on clean Bayer
tiles `Rcd` is level with it. `Ppg` and `Rcd` fail with `Error::InvalidArgument` on a sensor that is not Bayer when an
RGB output is requested; `Ahd` on another sensor, and edge-aware on other patterns, keep their documented fallbacks so
that existing callers keep working.

## Is edge-aware "less detailed" than AHD?

No. The fine-detail index (luma Laplacian of the central 1024x1024 crop) is higher for AHD than for edge-aware on real
Bayer files (0.0219 against 0.0179 on one 20 MP file, bilinear 0.0166), but that index also counts residual noise and
zipper energy. Against ground truth the order is the other way: on the 98 tiles of the new set edge-aware has the better
PSNR and edge PSNR than AHD at every noise level (clean 35.08 / 29.67 against 34.75 / 29.42 dB), and AHD has the lower
colour error on clean and moderate tiles. On a clean file the noise smoothing is off, so it does not explain the gap
either (output with and without it is identical on that file).

## Where edge-aware spends its time

One thread, 18 MP Bayer, 624 ms in all; per stage (timers inside the demosaic, one run):

| Stage | ms | Share |
|---|---:|---:|
| 7x7 directional filters (4 directions, 2 colours) | 205 | 33 % |
| colour conversion and 8-bit encoding (every mode pays this) | about 130 | 21 % |
| chroma pass 1 (gated mean of colour differences) | 112 | 18 % |
| chroma pass 2 | 55 | 9 % |
| direction weights | 52 | 8 % |
| first green estimate `G0` | 37 | 6 % |
| white-balanced plane build | 33 | 5 % |

Chroma pass 1 computed the same gate weight twice (red and blue); it is computed once now with identical output.
Scaling stops at about 8 threads, the physical core count of the machine; allocation, noise estimation (about 1 ms)
and column tiling were tried and did not explain it.

## Built from it

* `Demosaic::Auto`, `Demosaic::Ppg`, `Demosaic::Rcd` ([bayer-demosaics](bayer-demosaics.md)). On the held-out tiles of the
  published benchmark `Ppg` is above LibRaw PPG and `Rcd` above LibRaw DCB at every noise level ([benchmarks](../benchmarks.md)).
* `Demosaic::Ahd` blends its two candidates by their homogeneity instead of a hard choice: +0.25 dB clean and +0.7 to
  +0.8 dB under noise, and the lowest real-file speckle index of every demosaic measured ([ahd-demosaic](ahd-demosaic.md)).
* Edge-aware X-Trans filters trained with a 5x5 support (below).
* Speed, all with unchanged output unless noted: `Ppg` and `Rcd` in branch-free row loops with a 4-sample halo (25 % and
  14 % faster; the border ring changes) and 256-column chunks (cache-resident with many threads: 18 % and 25 % faster on
  16 threads of the x86-64 machine), the black-level normalisation vectorised (it was about a fifth of the
  post-process time of every demosaic), the 16-bit encode loop, `clamp` replaced in the AHD green loop (it kept the
  loop scalar; 10 % faster), and the bilinear path of X-Trans and other patterns (11 % faster). Figures from an Apple
  M1, one thread.
* `tools/bench/demosaic/own.py`: a ground-truth benchmark that needs numpy only (the older scripts need LibRaw, OpenCV
  and colour-demosaicing bindings), and `edgeaware_train.py gt` accepts X-Trans files (`.raf`). `truth.py --only`
  merges into the earlier results, as its help said.

## Tried and not kept

* **Homogeneity choice among the four directional candidates on X-Trans** (candidate images from the edge-aware
  filters, AHD-style thresholds and counts over the 3x3 window; hard choice among candidates within 90 % of the best
  count, or the gradient weights times the relative count to a power): on 98 tiles, clean X-Trans, 34.99 / 29.66 dB
  for the hard choice, 35.28 / 29.93 at best for the combination, against 35.30 / 29.93 for edge-aware. No gain.
* **A second pass that recomputes the direction weights from the first pass's green**: 35.29 / 29.91, level with
  edge-aware. The trained filters are tuned to the first-pass weights, so the combination stage is not where the
  remaining error is.
* **Residual interpolation for Bayer**: below `Rcd` everywhere; see [bayer-demosaics](bayer-demosaics.md).

## Filter support: 5x5 against 7x7

`edgeaware_train.py train --support 2` trains 5x5 filters inside the 7x7 table layout (outer taps exactly zero, which
the CPU kernels skip). Trained on the tiles of the published benchmark with the test split left out and scored with
`truth.py` on the 11 held-out photographic tiles (PSNR / edge PSNR, dB):

| held-out tiles | clean | moderate | high |
|---|---|---|---|
| Bayer 7x7 (shipped) | 42.04 / 39.43 | 37.67 / 36.16 | 34.40 / 33.40 |
| Bayer 5x5 | 41.99 / 39.32 | 37.67 / 36.15 | 34.41 / 33.41 |
| X-Trans 7x7 | 41.89 / 39.14 | 37.64 / 36.03 | 34.44 / 33.38 |
| X-Trans 5x5 (shipped) | 41.92 / 39.20 | 37.63 / 36.03 | 34.43 / 33.38 |

X-Trans now ships the 5x5 filters (retrained on all tiles; a 7x7 retrain on the same tiles reproduces the previous
table byte for byte, so the pipeline is reproducible). Bayer keeps 7x7: the 5x5 fit loses 0.11 dB of edge PSNR on
clean tiles. The time saved is small, 5 to 7 % of the X-Trans develop on one thread (a measurement on a busy machine), because the halo stays at 8 (Bayer
still needs it) and the filters are only part of the work. On a different, 98-tile set (half-size renders of 95
real-world files) the two supports were also level on Bayer.

## Open

* Training data: full-resolution linear targets with simulated blur, clipping and read noise; split by source image
  and camera. The 2x2-binned renders are slightly soft. Not started.
* The shared output stage: the 16-bit sRGB encode is about a third of a bilinear develop to `rgb16` (a gather and an
  interpolation per sample); `finish_rgb` works on interleaved triples and does not vectorise.
* Edge-aware on aarch64 is relatively slower than on x86-64 with AVX2 (6.6 times bilinear on an Apple M1, 4.0 times on
  a Ryzen 7 9700X, one thread): the kernels are written for 16-lane accumulators and NEON has 4 lanes, with no wider
  second compile.
* A per-pattern halo (7 for X-Trans with 5x5 filters) would save a little more on X-Trans.
* New X-Trans algorithms beyond edge-aware: the homogeneity choice and the second pass did not help; better training
  data is the next candidate.
