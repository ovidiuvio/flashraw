# Bayer demosaics: `Ppg` and `Rcd`

Two more opt-in demosaics for Bayer sensors, next to [`Ahd`](ahd-demosaic.md) and
[`EdgeAware`](edge-aware-demosaic.md). Both reconstruct green first and then red and blue from colour
differences. They are CPU only (the GPU backend decodes on the CPU for them), Bayer only (any other
sensor is `Error::InvalidArgument` for an RGB output; raw outputs ignore the demosaic), and
bit-identical for any thread count, band or crop: a pixel depends on the mosaic within 8 samples, and
pixels closer to the frame border use bilinear.

| | CLI | Python | JavaScript | C |
|---|---|---|---|---|
| Ppg | `--demosaic ppg` | `demosaic="ppg"` | `demosaic: 'ppg'` | `FLASHRAW_DEMOSAIC_PPG` |
| Rcd | `--demosaic rcd` | `demosaic="rcd"` | `demosaic: 'rcd'` | `FLASHRAW_DEMOSAIC_RCD` |

## Ppg (patterned pixel grouping)

1. **Green at a red or blue sample.** Row and column gradients `|G_w - G_e| + |2c - c_w2 - c_e2|` (and the same
   down the column); green is interpolated along the direction of the smaller gradient, with a
   second-difference correction from the sample's own colour, limited to the range of the two greens. Equal
   gradients average the two.
2. **Red at a blue sample (blue at a red one).** The mean of the colour differences `R - G` (`B - G`) along the
   diagonal of smaller gradient, added to the green.
3. **Red and blue at a green sample.** The green plus the mean colour difference of the two neighbours in the
   row (for the colour measured there) or in the column.

## Rcd (ratio corrected, experimental)

Written from the multiplicative colour model rather than from any existing implementation: where the image
scales by a factor from one pixel to the next, the green at a sample is the neighbouring green times the ratio
of the local intensities. The intensity is a low pass of the mosaic (`[1 2 1; 2 4 2; 1 2 1] / 16`), taken at the
sample and two away, so the estimate from the neighbour `n` is `g_n * 2 L_c / (eps + L_c + L_2n)`.

1. Four estimates (north, south, west, east), combined by inverse squared gradient weights into a vertical and a
   horizontal green, blended by how much the image varies across each direction, and limited to the range of the
   four neighbouring greens.
2. Red and blue as in Ppg, with soft diagonal weights `1 / (eps + d)^2` instead of the choice of one diagonal.

It may change or be removed between minor versions.

## What was measured

Ground truth: 98 low-noise 384x384 tiles (half-size renders of 95 files from 95 camera models of the real-world
set used for the other tests; some of those files may also be in the corpus the shipped edge-aware tables were
trained on, which would flatter edge-aware slightly), re-mosaicked as RGGB, noise added to the mosaic as in the other
benchmarks, `tools/bench/demosaic/own.py`. sRGB PSNR over all pixels / over the 15 % highest-gradient pixels
(dB) / colour error on edges (lower is better), means.

| Bayer | clean | moderate noise | high noise |
|---|---:|---:|---:|
| bilinear | 34.78 / 29.07 / 0.0412 | 32.92 / 28.18 / 0.0445 | 30.89 / 27.02 / 0.0495 |
| edge-aware | 35.08 / 29.67 / 0.0395 | **33.56 / 28.84 / 0.0414** | **31.77 / 27.73 / 0.0442** |
| AHD | 34.75 / 29.42 / 0.0378 | 32.19 / 28.05 / 0.0402 | 29.68 / 26.46 / 0.0440 |
| **Ppg** | 34.91 / 29.63 / 0.0369 | 32.34 / 28.24 / 0.0397 | 29.81 / 26.62 / 0.0440 |
| **Rcd** | **35.08 / 29.79 / 0.0368** | 32.57 / 28.43 / 0.0397 | 30.07 / 26.82 / 0.0439 |

* Ppg is better than AHD on every measure at every noise level of this set (0.16 dB more PSNR and 0.2 dB more
  edge PSNR on clean tiles, a lower colour error), and it is a much simpler method.
* Rcd matches edge-aware's PSNR on clean tiles, has the better edge PSNR and the lower colour error there, and
  is a pure interpolation. Edge-aware is ahead on noisy tiles because it smooths them (`NoiseReduction`), which
  the others do not.
* None of them replaces edge-aware as the choice for the smallest error on noisy files.

Cost, user CPU time per iteration on one thread (`flashraw benchmark`, 18 MP file, minimum of three runs). The
machine was heavily loaded by other jobs at the time, which inflates every figure about twofold, so read the
ratios, not the milliseconds: bilinear 367, **Ppg 647**, **Rcd 706**, AHD 836, edge-aware 1222. Ppg and Rcd use
plain scalar loops; they have not been vectorised like the AHD and edge-aware kernels, so there is room left.

Experiments that were run and not kept:

* **Residual interpolation** (a guided-filter prediction of red and blue from green over 5x5 windows of the
  samples, with the prediction error interpolated bilinearly and added back; green from the same ratio-corrected
  estimate): below Rcd on every measure at every noise level, for every regularisation
  tried (`1e-4` to `1e-1`). Not shipped.
