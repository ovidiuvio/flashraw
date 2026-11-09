# Demosaic audit and experiment plan

What was measured on the current algorithms (Linux x86-64, 8 cores / 16 threads, 18 MP Bayer and 40 MP X-Trans
files; numbers from [benchmarks](../benchmarks.md)), and which experiments follow from it. Nothing here changes a
default: `Demosaic::Bilinear` stays the default and `Demosaic::Auto` is opt-in.

## Algorithm selection

`Demosaic::Auto` picks the algorithm per sensor when the file is decoded: edge-aware on Bayer and X-Trans,
bilinear on any other pattern. The choice depends only on the sensor type, so it is the same on every machine.
It resolves to the edge-aware demosaic because that has the smallest error against ground truth on Bayer
(clean, moderate and high noise) and on X-Trans (moderate and high noise; on clean X-Trans tiles Markesteijn-class
methods are within 0.12 dB on edges). The unsupported combinations stay as documented: `Ahd` on a non-Bayer
sensor uses the edge-aware demosaic, edge-aware on other patterns uses bilinear; both are documented fallbacks,
not errors, to keep existing callers working.

## Is edge-aware "less detailed" than AHD?

The fine-detail index (luma Laplacian of the central 1024x1024 crop) is higher for AHD than for edge-aware on
real Bayer files (0.0219 against 0.0179 on one 20 MP compact-camera file, bilinear 0.0166). That index also counts
residual noise and zipper energy, so it cannot say which is sharper. Against ground truth the order is the other
way: on clean photographic tiles edge-aware has the better edge PSNR (39.43 dB against 38.87 dB for AHD) and the
better overall PSNR on noisy tiles (+2.7 dB at moderate noise). On a clean file the noise smoothing is off
(`NoiseReduction::Auto` measures the noise first), so the smoothing does not explain the detail gap there either:
output with and without it is identical on that file. The finding: the AHD detail excess is mostly texture the
ground truth does not contain; edge-aware is not losing real detail. No change was made.

## Where edge-aware spends its time

Single thread, 18 MP Bayer: bilinear 171 ms, AHD 392 ms, edge-aware 601 ms, all including the colour conversion and
8-bit encoding that every mode shares. Edge-aware computes a first green estimate, four directional 7x7 filters per
missing colour, the direction weights, two gated chroma passes and the optional smoothing. The kernels are already
register-tiled with exact-zero taps removed; what remains is dense 7x7 arithmetic. Scaling stops at about 8 threads,
which is the physical core count of the test machine; allocation, noise estimation (about 1 ms) and column tiling
were tried and did not explain it. Not yet profiled per stage: how the 601 ms splits between the filters, the chroma
passes and the shared tail.

## Prioritised experiments

1. **Edge-aware detail and cost.** Train 5x5 and 5x7 variants of the directional filters and measure the PSNR cost
   against the speed gain (the 7x7 support is where the time goes). Keep only if edge PSNR drops by less than 0.1 dB.
2. **AHD correctness checks.** Compare against synthetic edges of each orientation and colour; the speckle index
   already matches the published method (1.53 against 1.57), so the work is in the confidence-weighted blend in
   ambiguous areas (soft selection where the homogeneity counts are within one).
3. **X-Trans candidates.** Build several directional candidates for the 6x6 pattern and pick by homogeneity, as AHD
   does for Bayer, and compare with the current edge-aware filters on the same ground-truth tiles.
4. **Training data.** Full-resolution linear-light targets with simulated shot and read noise, split by source image.
   The current targets are 2x2-binned renders, which are slightly soft.
5. **Output tail.** Measure the share of the colour conversion and sRGB encoding, which every mode pays, and
   vectorise it if it is a large part of the bilinear 171 ms.
6. **PPG, RCD.** Only if they beat bilinear on speed/quality or AHD on speed without losing speckle; PPG would sit
   between bilinear (171 ms) and AHD (392 ms) on one thread, which is a narrow gap.
