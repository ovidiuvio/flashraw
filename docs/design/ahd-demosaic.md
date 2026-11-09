# Homogeneity-directed demosaic (`Demosaic::Ahd`)

`Demosaic::Ahd` is the opt-in alternative to [`Demosaic::EdgeAware`](edge-aware-demosaic.md) for Bayer sensors
that follows a published, widely used method: adaptive homogeneity-directed demosaicing (Hirakawa and Parks, 2005).
It is written from the paper's description; the module is `crates/flashraw/src/ahd.rs`.

| | CLI | Python | JavaScript | C |
|---|---|---|---|---|
| name | `--demosaic ahd` | `demosaic="ahd"` | `demosaic: 'ahd'` | `FLASHRAW_DEMOSAIC_AHD` |

Sensors that are not Bayer (X-Trans, four-colour, Foveon, ...) use the edge-aware demosaic where it exists and bilinear
otherwise. It runs on the CPU only: the GPU backend decodes on the CPU for it.

## Method

Per pixel, on the white-balanced mosaic (clipped at the white level):

1. **Two candidates.** Green is interpolated along rows (H) and along columns (V): the mean of the two nearest greens
   plus a quarter of the second difference of the colour at the pixel, limited to the range of those two greens.
   Red and blue follow from colour differences: a missing colour is the pixel's green plus the mean of the
   differences `R - G` (or `B - G`) measured in the 3x3 window, each candidate using its own green plane.
2. **Perceptual coordinates.** Each candidate maps to a lightness `L = (R' + 2 G' + B') / 4` and chroma
   `a = R' - G'`, `b = B' - G'` with `X' = sqrt(max(X, 0))`. This is a cheap stand-in for CIELab: on the benchmark
   tiles and real crops it gave the same scores to the second decimal, and costs a square root instead of a
   cube root with a table.
3. **Homogeneity.** The thresholds at a pixel are the smaller, over H (neighbours along the row) and V (neighbours along
   the column), of the largest lightness step and of the largest squared chroma step to the two neighbours. A
   candidate's homogeneity there is the number of the 8 neighbours within both thresholds, summed over the 3x3 window.
4. **Choice.** The candidate with the larger sum wins; a tie averages the two.

Support is 6 samples on every side, so every pixel is computed by the same operations whatever the band (64 rows),
column chunk (192 columns), thread count or crop; pixels closer than 6 samples to the frame border use bilinear.
The plane loops are compiled a second time for AVX2 (no FMA) and dispatched at run time; results are bit-identical.
The homogeneity counts run over contiguous planes (a first version over interleaved colour triples was 2.5 times
slower).

## Against the other demosaics

Bayer, ground-truth tiles (PSNR over all pixels / over the highest-gradient pixels, dB, photographic tiles), the
no-reference speckle index of real files and the cost, from `docs/benchmarks.md`:

| | clean | moderate noise | speckle index (13 files) |
|---|---:|---:|---:|
| `Demosaic::Ahd` | 41.51 / 38.87 | 34.97 / 33.99 | 1.62 |
| `Demosaic::EdgeAware` | 42.04 / 39.43 | 37.67 / 36.16 | 2.09 |

`EdgeAware` is closer to the truth on clean and (much more) on noisy tiles, because its trained filters keep the
measured sample and it smooths noise; `Ahd` is cleaner on real files by the no-reference index, which counts the
sensor noise it keeps as detail. Use `Ahd` to match the look of the method it implements, `EdgeAware` for the
smallest error.
