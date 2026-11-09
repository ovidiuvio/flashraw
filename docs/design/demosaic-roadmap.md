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

* `Demosaic::Auto`, `Demosaic::Ppg`, `Demosaic::Rcd` ([bayer-demosaics](bayer-demosaics.md)): on 98 held-out tiles `Ppg`
  beats AHD on every measure at every noise level, and `Rcd` matches edge-aware on clean tiles.
* `tools/bench/demosaic/own.py`: a ground-truth benchmark that needs numpy only (the older scripts need LibRaw, OpenCV
  and colour-demosaicing bindings), and `edgeaware_train.py gt` accepts X-Trans files (`.raf`).

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
the CPU kernels skip). Bayer, trained on the same 76 training tiles (every fifth source file left out) and scored on
the 22 test tiles, sRGB PSNR / edge PSNR / colour error:

| Bayer, test split | clean | moderate | high |
|---|---|---|---|
| shipped tables (7x7) | 35.31 / 30.20 / 0.0468 | 33.38 / 29.10 / 0.0487 | 31.32 / 27.79 / 0.0516 |
| retrained 7x7 | 35.50 / 30.46 / 0.0457 | 33.45 / 29.24 / 0.0479 | 31.37 / 27.90 / 0.0509 |
| retrained 5x5 | 35.52 / 30.49 / 0.0457 | 33.47 / 29.26 / 0.0478 | 31.38 / 27.92 / 0.0507 |

On real files (12 cameras) the colour speckle index is 2.07 for the shipped tables and 2.06 for both retrained sets;
the fine-detail index is 6 % lower for the retrained sets. So 25 taps do as well as 49 on this data, which would roughly
halve the cost of the filter stage. The shipped tables were **not** replaced: the retrained ones come from a different
tile set than the one the published benchmark was produced with, and that benchmark cannot be repeated without the
CC0 corpus and the reference converters, so the numbers in [benchmarks](../benchmarks.md) would stop describing the
shipped tables. The next step is to retrain on the corpus tiles with `--support 2`, rerun `truth.py`, and ship the
tables and the new numbers together.

## Open

* Retrain with 5x5 support on the corpus tiles, rerun the published benchmarks, then reduce the filter support in the
  CPU and GPU kernels (the halo shrinks from 8 to 7).
* Vectorise `Ppg` and `Rcd` like the AHD and edge-aware kernels (they are plain scalar loops).
* Training data: full-resolution linear targets with simulated blur, clipping and read noise; split by source image
  and camera. The 2x2-binned renders are slightly soft.
* Measure the output tail (colour conversion and encoding, about 130 ms) on a quiet machine and vectorise it.
* Speed tables for `Ppg` and `Rcd` against the reference converters, and the published real-file and ground-truth
  tables for them, need the reference bindings (rawpy); this machine does not have them.
* New X-Trans algorithms beyond edge-aware: the two cheap ideas above did not help, so the next candidates are
  better trained filters (above), not a different combination rule.
