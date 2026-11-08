# Edge-aware demosaic (`Demosaic::EdgeAware`)

`Demosaic::Bilinear` (the default) is the fastest full-resolution demosaic: on Bayer sensors it averages the
nearest samples of each colour, and on X-Trans and other patterns it averages the same-colour samples of the 3x3
window. That is cheap and has no ringing, but it zippers along edges and, on X-Trans (red and blue are only one
pixel in five), leaves isolated green, magenta and blue dots where bright and dark detail meet
([issue #3](https://github.com/ybouane/flashraw/issues/3)). `Demosaic::EdgeAware` is the opt-in alternative for
Bayer and X-Trans sensors. Other patterns (four-colour, Foveon, monochrome, staggered) use bilinear.

| | CLI | Python | JavaScript | C |
|---|---|---|---|---|
| name | `--demosaic edge-aware` | `demosaic="edge-aware"` | `demosaic: 'edge-aware'` | `FLASHRAW_DEMOSAIC_EDGE_AWARE` |

It is CPU only. `Backend::Auto` and `Backend::Gpu` decode on the CPU when it is selected (as they do for output
profiles with lookup tables); GPU-resident outputs reject it with `NotAvailable`. The result may change between
minor versions as the filters are retrained, so pin it per version if you hash outputs.

## Method

Per pixel, on the white-balanced mosaic (samples clipped to the white level, as the output is):

1. **Green estimate `G0`.** One non-directional 7x7 filter per CFA position gives a first green plane.
2. **Direction weights.** The squared gradients of `sqrt(G0)` along four directions (horizontal, vertical and the
   two diagonals), each summed over 3x3, give `E_d`. The weights are `w_d = 1 / (E_d + 1e-3)^2`, normalised to sum
   to one: the less the image varies along a direction, the more that direction is trusted. Soft weights, not a
   hard choice, so there are no switching artefacts.
3. **Interpolation.** Each missing colour is `sum_d w_d * (F_d * mosaic)`, where `F_d` is a 7x7 filter chosen by
   CFA position (4 for Bayer, 36 for X-Trans), colour and direction. The measured colour is kept as is. Every
   filter sums to one, so a flat field stays flat.
4. **False-colour suppression.** Isolated coloured dots go; green (luminance) detail and the measured samples are
   untouched. Bayer, two passes: each missing red or blue is the green plus the weighted mean of the differences
   `R - G` (or `B - G`) measured at the red (blue) sites of the 3x3 window, a site weighing `1 / (1 + e^2)` with
   `e = (G_site - G) / s` and `s = 0.2 G + 0.015` (on the white-balanced, normalised mosaic), so that colour does
   not leak across a luminance edge; then the same over the differences of every pixel of the 3x3 window (measured
   and pass-1 values) with `s = 0.1 G + 0.007`. X-Trans: `R - G` and `B - G` are replaced by their 3x3 median (the
   usual colour-difference median of classical demosaics). Both cost some colour resolution at fine saturated
   edges. Choosing the Bayer stage: a 3x3 median on the full differences left the real-file speckle at 2.99 on the
   index below; a single gated pass of measured differences brought it to 2.08 but gave back 0.1 to 0.7 dB of
   photo edge PSNR (it discards the filters' own estimate); the second pass over all pixels gets 1.98 and brings
   the PSNR back, 0.8 dB above the median on synthetic edges.

5. **Noise smoothing (Bayer).** On noisy frames every channel of a pixel moves by `beta` towards the mean of its 3x3
   window, a neighbour weighing `1 / (1 + e^2)` with `e = (G_neighbour - G) / (0.2 G + 0.015)`. `beta` comes from a
   noise estimate over 48 rows spread over the whole frame (so it does not depend on the crop): the lower quintile of
   `|a - b| / sqrt((a + b) / 2)` over same-colour green pairs two samples apart gives a noise coefficient `nu`, and
   `beta = 0.5 * clamp((nu - 2e-4) / 6e-4, 0, 1)`: 0 on clean frames, 0.5 at the noise of the 'moderate' training level
   and above. The filters keep the measured sample, so on a noisy mosaic they lose to the 3x3 average; the smoothing
   recovers that (on the tiles: +0.6 dB at moderate and +1.1 dB at high noise over bilinear, nothing on clean
   frames). X-Trans has no smoothing step yet.

Support is 8 pixels around each output pixel on Bayer (3 for the filter, 2 for the weights' gradients, 2 for the chroma passes, 1 for the smoothing; 6 on X-Trans, whose median is one pass).
The develop runs in bands of 32 output rows plus that halo, so memory stays bounded for any size (a few MB per
thread), and every pixel is computed by the same operations in the same order whatever the band, thread count,
crop or backend setting: results are bit-identical (tests: `edgeaware::tests`, `tests/edgeaware.rs`). Pixels
closer than 8 samples to the frame border use bilinear.

A 4-wide `f32` accumulate over contiguous per-phase planes keeps the cost to a few times bilinear:

| Post-process (RGBA8, this Linux x86-64 machine, 16 cores) | bilinear | edge-aware |
|---|---|---|
| 39.8 MP X-Trans, 1 thread | 0.94 s | 3.5 s |
| 39.8 MP X-Trans, 16 threads | 0.11 s | 0.59 s |
| 18 MP Bayer, 1 thread | 0.12 s | 1.2 s |
| 18 MP Bayer, 16 threads | 0.02 s | 0.23 s |

## Where the filters come from

The filters are the least-squares solution (ridge regularised, constrained to sum to one) that reconstructs ground
truth images from their re-mosaicked version, for fixed direction weights, so the whole design stays one linear
problem solved with `numpy.linalg`. They are trained by [`tools/demosaic/edgeaware_train.py`](../../tools/demosaic/edgeaware_train.py)
and stored as little-endian `f32` tables in `crates/flashraw/src/edgeaware/{bayer,xtrans}.bin` (10 KB and 92 KB).

No third-party code, tables or coefficients are used. The ideas are standard: gradient-directed interpolation,
colour-difference interpolation and median false-colour suppression, and least-squares filter design. The training
data is:

* tiles (384x384) of half-size renders of CC0 sample files (raw.pixls.us, `tools/corpus/fetch.py`): 2x2 binning
  measures all three colours at every pixel, so these are true RGB images. Only low-noise tiles are kept, one
  of the most detailed and one of median detail per file;
* synthetic images generated by the script (rotated rectangles, discs, thin lines, gradients at four times
  supersampling) for hard edges that natural tiles rarely have;
* each image at three simulated noise levels (variance `a * value + b`), so that the filters also smooth chroma
  noise.

To regenerate: `edgeaware_train.py gt --corpus corpus --flashraw target/release/flashraw --out GT` extracts the
tiles, `edgeaware_train.py train --gt GT --out crates/flashraw/src/edgeaware` writes the tables, and
`edgeaware_train.py eval --gt GT` trains on 80 % of the files and prints PSNR on the rest.

## Measured

Held-out tiles, sRGB-encoded PSNR in dB (all pixels / the 15 % highest-gradient pixels), against the 3x3
average (bilinear for Bayer is slightly different, see the table note):

| | clean mosaic | noisy mosaic |
|---|---|---|
| Bayer: 3x3 average | 37.05 / 31.34 | 34.22 / 29.53 |
| Bayer: edge-aware | 37.55 / 32.20 | 33.98 / 29.81 |
| X-Trans: 3x3 average | 36.23 / 30.28 | 33.63 / 28.74 |
| X-Trans: edge-aware | 37.20 / 31.72 | 33.67 / 29.39 |

PSNR against clean references understates the main effect, which is the removal of isolated coloured dots and
zipper patterns (step 4 trades a little fidelity for that: without it Bayer clean would be 37.7 over all pixels and
X-Trans 37.95 / 32.3). On the file of issue #3 (Fujifilm X-T50, DSCF0120.RAF) the chroma high-frequency index of the
issue (mean Laplacian of R-G and B-G over the Laplacian of luma, of the sRGB-encoded 16-bit output) goes from
4.29 with bilinear to 2.25.

## Limits

* CPU only; several times slower than bilinear (table above).
* Hard saturated colour edges (pure red against green) lose a little sharpness in the colour channels because of
  the chroma stage; bilinear does not.
* The weights assume a roughly neutral scene after white balance (the training data is white balanced); strongly
  coloured light sources still decode correctly but the direction estimate is less reliable.
* Highlight handling is not part of it: clipped samples are clamped to white as in every other mode.
