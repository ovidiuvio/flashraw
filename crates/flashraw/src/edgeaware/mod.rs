//! Edge-aware demosaic ([`crate::Demosaic::EdgeAware`]) for Bayer and X-Trans sensors.
//!
//! For every pixel:
//!
//! 1. a first estimate of the green plane `G0` comes from one non-directional 7x7 filter per
//!    CFA position,
//! 2. the squared gradients of `sqrt(G0)` along four directions (horizontal, vertical and
//!    both diagonals), summed over 3x3, give four weights `w_d = 1 / (E_d + EPS)^2`
//!    (normalised to sum to one): the lower the variation along a direction, the more that
//!    direction is trusted,
//! 3. each missing colour is `sum_d w_d * (filter_d * mosaic)`, with one 7x7 filter per CFA
//!    position, colour and direction. The measured colour is kept as is,
//!
//! The filters are least-squares fits to re-mosaicked training images and sum to one, so a
//! flat field stays flat; they are trained by `tools/demosaic/edgeaware_train.py` (see
//! `docs/design/edge-aware-demosaic.md` for the method and its provenance) and stored in
//! `bayer.bin` / `xtrans.bin`.
//!
//! 4. the colour differences R-G and B-G are cleaned, so that isolated coloured dots (the
//!    main artefact of sparse red and blue sites) disappear and green (luminance) detail is
//!    untouched. X-Trans: their 3x3 median. Bayer, two passes: each missing red or blue is
//!    the green plus the weighted mean of the differences measured at the red or blue sites
//!    of the 3x3 window, a site counting `1 / (1 + ((G_site - G) / s)^2)` with
//!    `s = 0.2 G + 0.015`, so that colour does not leak across a luminance edge; then the
//!    same over the differences of all pixels of the 3x3 window with `s = 0.1 G + 0.007`.
//! 5. On noisy frames: every channel of the pixel moves by `beta` towards the
//!    mean of its 3x3 window, weighted as in step 4 with `s = 0.2 G + 0.015`. `beta` is 0 for
//!    clean frames and up to 0.5 for noisy ones, from a noise estimate over 48 rows of the
//!    whole frame ([`Plan::set_noise_from`]): the lower quintile of
//!    `|a - b| / sqrt((a + b) / 2)` over same-colour green pairs two samples apart.
//!
//! The mosaic is white balanced before the filters see it, so the colour channels have
//! comparable levels (as in the training data).
//!
//! Every output pixel depends only on mosaic samples within [`HALO`] pixels, and is computed
//! by the same operations in the same order whatever the band, thread or block it falls in:
//! results are bit-identical for any strip height, thread count or crop.

use flashraw_core::CfaPattern;
use std::f32::consts::FRAC_1_SQRT_2;
use std::sync::OnceLock;

const K: usize = 7;
const R: usize = K / 2;
const TAPS: usize = K * K;
const NDIR: usize = 4;
const EPS: f32 = 1e-3;
/// Mosaic samples needed around a pixel by the filters: the 7x7 filter reaches 3, and the
/// weights need `G0` two pixels further (gradient, then 3x3 sum) which itself reaches 3.
const CORE_HALO: usize = R + 2;
/// Mosaic samples needed around a pixel: the filters, plus three on Bayer (two chroma passes
/// and the noise smoothing; X-Trans needs one, for its median).
pub(crate) const HALO: usize = CORE_HALO + 3;
/// Output rows per band: bounds the working memory (a few MB at 8000 columns).
pub(crate) const BAND_ROWS: usize = 32;

struct Tables {
    period: usize,
    /// Colour (0 R, 1 G, 2 B) of each position of the canonical pattern.
    canon: Vec<u8>,
    /// `[period*period][49]`: the `G0` filter.
    g0: Vec<f32>,
    /// `[period*period][3][NDIR][49]`.
    dir: Vec<f32>,
}

fn load(bytes: &[u8], period: usize, canon: &[u8]) -> Tables {
    let n = period * period;
    let f: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    assert_eq!(f.len(), n * TAPS + n * 3 * NDIR * TAPS, "edge-aware table size");
    Tables { period, canon: canon.to_vec(), g0: f[..n * TAPS].to_vec(), dir: f[n * TAPS..].to_vec() }
}

fn bayer() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| load(include_bytes!("bayer.bin"), 2, &[0, 1, 1, 2]))
}

fn xtrans() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        // GGRGGB / GGBGGR / BRGRBG / GGBGGR / GGRGGB / RBGBRG
        let rows = ["GGRGGB", "GGBGGR", "BRGRBG", "GGBGGR", "GGRGGB", "RBGBRG"];
        let canon: Vec<u8> = rows
            .iter()
            .flat_map(|r| {
                r.bytes().map(|c| match c {
                    b'R' => 0,
                    b'G' => 1,
                    _ => 2,
                })
            })
            .collect();
        load(include_bytes!("xtrans.bin"), 6, &canon)
    })
}

/// The edge-aware demosaic of one frame: which table applies and where the CFA sits in it.
#[derive(Clone)]
pub(crate) struct Plan {
    t: &'static Tables,
    /// Canonical column / row of frame column / row 0.
    cx: usize,
    cy: usize,
    /// Weight of the gated smoothing of the output (Bayer), from the noise of the frame.
    beta: f32,
    /// `shifts[q][tx]` = (phase of the tap's column, offset of its index in that phase's plane).
    shifts: Vec<[(usize, isize); K]>,
}

impl Plan {
    /// The plan for a frame whose sample `(0, 0)` is at sensor position `origin`, or `None`
    /// when the pattern is neither a Bayer nor an X-Trans arrangement (a shifted copy of the
    /// canonical one is fine). `chan` maps `CfaColor` values to channels 0..3.
    pub(crate) fn new(cfa: &CfaPattern, chan: &[u8; 8], origin: (usize, usize)) -> Option<Plan> {
        for t in [bayer(), xtrans()] {
            let p = t.period;
            if cfa.width() != p || cfa.height() != p {
                continue;
            }
            for sy in 0..p {
                for sx in 0..p {
                    let ok = (0..p).all(|y| {
                        (0..p).all(|x| chan[cfa.color_at(x, y) as usize] == t.canon[((y + sy) % p) * p + (x + sx) % p])
                    });
                    if ok {
                        let (cx, cy) = ((sx + origin.0) % p, (sy + origin.1) % p);
                        let off = |q: usize| (q + p - cx) % p; // frame column offset of phase q
                        let shifts = (0..p)
                            .map(|q| {
                                std::array::from_fn(|tx| {
                                    let dx = tx as isize - R as isize;
                                    let q2 = (q as isize + dx).rem_euclid(p as isize) as usize;
                                    (q2, (off(q) as isize + dx - off(q2) as isize) / p as isize)
                                })
                            })
                            .collect();
                        return Some(Plan { t, cx, cy, beta: 0.0, shifts });
                    }
                }
            }
        }
        None
    }

    /// Develop rows `y0..y1` (frame coordinates) and columns `xa..xb` into `out`
    /// (`(y1 - y0) * (xb - xa)` camera RGB pixels, white balanced by `wb`, indexed by
    /// channel). Needs `HALO <= y0`, `y1 + HALO <= height`, `HALO <= xa`, `xb + HALO <= w`.
    /// `fetch(y, row)` fills `row` (`w` samples) with the normalised mosaic row `y`.
    pub(crate) fn develop(
        &self,
        w: usize,
        rows: (usize, usize),
        cols: (usize, usize),
        wb: &[f32; 4],
        fetch: &mut dyn FnMut(usize, &mut [f32]),
        out: &mut [[f32; 3]],
    ) {
        // The same source compiled for AVX2 (without FMA: every lane performs the same
        // multiplies and adds in the same order, so results are bit-identical to the baseline
        // build on any CPU) gives 8-wide vectors in the filter loops.
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked above).
            return unsafe { self.develop_avx2(w, rows, cols, wb, fetch, out) };
        }
        self.develop_impl(w, rows, cols, wb, fetch, out)
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn develop_avx2(
        &self,
        w: usize,
        rows: (usize, usize),
        cols: (usize, usize),
        wb: &[f32; 4],
        fetch: &mut dyn FnMut(usize, &mut [f32]),
        out: &mut [[f32; 3]],
    ) {
        self.develop_impl(w, rows, cols, wb, fetch, out)
    }

    #[inline(always)]
    fn develop_impl(
        &self,
        w: usize,
        (y0, y1): (usize, usize),
        (xa, xb): (usize, usize),
        wb: &[f32; 4],
        fetch: &mut dyn FnMut(usize, &mut [f32]),
        out: &mut [[f32; 3]],
    ) {
        let (nr, nc) = (y1 - y0, xb - xa);
        debug_assert!(y0 >= HALO && xa >= HALO && xb + HALO <= w && out.len() == nr * nc);
        if self.t.period == 2 {
            return self.develop_bayer(w, (y0, y1), (xa, xb), wb, fetch, out);
        }
        // One pixel more on every side for the 3x3 medians, and `h` more for the smoothing.
        let h = usize::from(self.beta > 0.0);
        let m = 1 + h;
        let (er, ec) = (nr + 2 * m, nc + 2 * m);
        let mut rgb: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; er * ec]);
        self.demosaic(w, (y0 - m, y1 + m), (xa - m, xb + m), wb, fetch, &mut rgb);
        let t = self.t;
        let p = t.period;
        let [r, g, bl] = &rgb;
        // Colour differences, then their 3x3 medians (rows of `e2c` values at a time).
        let rg: Vec<f32> = r.iter().zip(g).map(|(a, b)| a - b).collect();
        let bg: Vec<f32> = bl.iter().zip(g).map(|(a, b)| a - b).collect();
        let (e2r, e2c) = (nr + 2 * h, nc + 2 * h);
        let mut mr = vec![0f32; e2c];
        let mut mb = vec![0f32; e2c];
        let mut v2 = vec![[0f32; 3]; e2r * e2c];
        for i in 0..e2r {
            let cr = (y0 + i + self.cy + p - h) % p;
            median_row(&rg, ec, i, &mut mr);
            median_row(&bg, ec, i, &mut mb);
            for j in 0..e2c {
                let c = (i + 1) * ec + j + 1;
                let own = t.canon[cr * p + (xa + j + self.cx + p - h) % p];
                v2[i * e2c + j] =
                    [if own == 0 { r[c] } else { g[c] + mr[j] }, g[c], if own == 2 { bl[c] } else { g[c] + mb[j] }];
            }
        }
        if h == 0 {
            out.copy_from_slice(&v2);
        } else {
            smooth(&v2, e2c, self.beta, nr, nc, out);
        }
    }

    /// Estimate the noise of a frame of `w` x `h` samples and set the weight of the
    /// smoothing pass (module docs, step 5) from it. Deterministic and independent of any
    /// crop: 48 rows spread over the whole frame, so call it with the full frame's `fetch`.
    pub(crate) fn set_noise_from(&mut self, w: usize, h: usize, g_gain: f32, fetch: &mut dyn FnMut(usize, &mut [f32])) {
        self.beta = 0.0;
        if h < 16 || w < 16 {
            return;
        }
        const ROWS: usize = 48;
        let mut row = vec![0f32; w];
        let mut v: Vec<f32> = Vec::with_capacity(ROWS * w / 2);
        for n in 0..ROWS {
            let y = 4 + n * (h - 9) / (ROWS - 1);
            fetch(y, &mut row);
            let p = self.t.period;
            let green = |x: usize| self.t.canon[((y + self.cy) % p) * p + (x + self.cx) % p] == 1;
            for x in 0..w - 2 {
                if !green(x) {
                    continue;
                }
                // Same-colour green pairs one or two samples apart (two apart on Bayer).
                for d in 1..=2 {
                    if !green(x + d) {
                        continue;
                    }
                    let (a, b) = (row[x] * g_gain, row[x + d] * g_gain);
                    let mid = 0.5 * (a + b);
                    if mid > 0.03 && mid < 0.6 {
                        v.push((a - b).abs() / mid.sqrt());
                    }
                }
            }
        }
        if v.len() < 1000 {
            return;
        }
        // The lower quintile of |a - b| / sqrt(mean) comes from the flattest areas, where the
        // difference of two samples of variance nu * mean has a half-normal distribution.
        let k = v.len() / 5;
        let q = *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
        let sigma = q / (std::f32::consts::SQRT_2 * 0.2533);
        let nu = sigma * sigma;
        self.beta = 0.5 * ((nu - 2e-4) / 6e-4).clamp(0.0, 1.0);
    }

    /// Bayer: the filters, the two-pass luminance-gated chroma stage and, on a noisy frame, a
    /// gated smoothing (module docs, steps 4 and 5). Same contract as [`Plan::develop`].
    #[inline(always)]
    fn develop_bayer(
        &self,
        w: usize,
        (y0, y1): (usize, usize),
        (xa, xb): (usize, usize),
        wb: &[f32; 4],
        fetch: &mut dyn FnMut(usize, &mut [f32]),
        out: &mut [[f32; 3]],
    ) {
        let t = self.t;
        let (nr, nc) = (y1 - y0, xb - xa);
        // `h` extra pixels around the core for the smoothing pass; two for the chroma passes.
        let h = usize::from(self.beta > 0.0);
        let m = 2 + h;
        let (er, ec) = (nr + 2 * m, nc + 2 * m);
        let mut rgb: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; er * ec]);
        self.demosaic(w, (y0 - m, y1 + m), (xa - m, xb + m), wb, fetch, &mut rgb);
        let [r, g, bl] = &rgb;
        // Channel at plane position (iy, ix): the plane starts `m` pixels before (y0, xa).
        let (py, px) = ((y0 + self.cy + m) % 2, (xa + self.cx + m) % 2);
        let colour = |iy: usize, ix: usize| t.canon[((iy + py) % 2) * 2 + (ix + px) % 2] as usize;
        // Pass 1, over the core plus `1 + h` pixels: green plus the gated mean of the measured
        // differences of the 3x3 window. `d1[ch / 2]` holds the colour difference (R-G, B-G).
        let (e1r, e1c) = (nr + 2 + 2 * h, nc + 2 + 2 * h);
        let mut d1 = [vec![0f32; e1r * e1c], vec![0f32; e1r * e1c]];
        for i in 0..e1r {
            for j in 0..e1c {
                let c = (i + 1) * ec + j + 1;
                let own = colour(i + 1, j + 1);
                let gc = g[c];
                let inv = 1.0 / (0.2 * gc.max(0.0) + 0.015);
                for ch in [0usize, 2] {
                    let plane = if ch == 0 { r } else { bl };
                    d1[ch / 2][i * e1c + j] = if own == ch {
                        plane[c] - gc
                    } else {
                        let (mut sw, mut sd, mut su, mut n) = (0f32, 0f32, 0f32, 0f32);
                        for dy in 0..3 {
                            for dx in 0..3 {
                                if colour(i + dy, j + dx) != ch {
                                    continue;
                                }
                                let k = (i + dy) * ec + j + dx;
                                let d = plane[k] - g[k];
                                let e = (g[k] - gc) * inv;
                                let wt = 1.0 / (1.0 + e * e);
                                sw += wt;
                                sd += wt * d;
                                su += d;
                                n += 1.0;
                            }
                        }
                        // Every 3x3 window holds at least two sites of each colour, and every
                        // weight is at least 1 / (1 + e^2) > 0; the mean is a safeguard.
                        if sw > 1e-12 { sd / sw } else { su / n }
                    };
                }
            }
        }
        // Pass 2, over the core plus `h` pixels: the same gate over the differences of every
        // pixel (measured and pass-1 values), for the two colours the pixel does not measure.
        let (e2r, e2c) = (nr + 2 * h, nc + 2 * h);
        let mut v2 = vec![[0f32; 3]; e2r * e2c];
        for i in 0..e2r {
            for j in 0..e2c {
                let c = (i + 2) * ec + j + 2;
                let own = colour(i + 2, j + 2);
                let gc = g[c];
                let inv = 1.0 / (0.1 * gc.max(0.0) + 0.007);
                let mut v = [r[c], gc, bl[c]];
                for ch in [0usize, 2] {
                    if own == ch {
                        continue;
                    }
                    let (mut sw, mut sd) = (0f32, 0f32);
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let e = (g[(i + dy + 1) * ec + j + dx + 1] - gc) * inv;
                            let wt = 1.0 / (1.0 + e * e);
                            sw += wt;
                            sd += wt * d1[ch / 2][(i + dy) * e1c + j + dx];
                        }
                    }
                    v[ch] = gc + sd / sw;
                }
                v2[i * e2c + j] = v;
            }
        }
        if h == 0 {
            out.copy_from_slice(&v2);
            return;
        }
        smooth(&v2, e2c, self.beta, nr, nc, out);
    }

    /// The filters over `y0..y1` x `xa..xb`, into three planes of `(y1-y0) * (xb-xa)`; needs
    /// `CORE_HALO` samples around.
    #[inline(always)]
    fn demosaic(
        &self,
        w: usize,
        (y0, y1): (usize, usize),
        (xa, xb): (usize, usize),
        wb: &[f32; 4],
        fetch: &mut dyn FnMut(usize, &mut [f32]),
        out: &mut [Vec<f32>; 3],
    ) {
        let t = self.t;
        let p = t.period;
        let (nr, nc) = (y1 - y0, xb - xa);
        debug_assert!(y0 >= CORE_HALO && xa >= CORE_HALO && xb + CORE_HALO <= w && out[0].len() == nr * nc);
        let by0 = y0 - CORE_HALO;
        let nrb = nr + 2 * CORE_HALO;
        let pw = w.div_ceil(p) + 1;
        let off = |q: usize| (q + p - self.cx) % p;

        // 1. White balanced mosaic, split by column phase so every filter tap is a
        //    contiguous run.
        let mut planes = vec![0f32; p * nrb * pw];
        let mut row = vec![0f32; w];
        for lr in 0..nrb {
            fetch(by0 + lr, &mut row);
            let cr = (by0 + lr + self.cy) % p;
            let mut q = self.cx;
            for (x, &v) in row.iter().enumerate() {
                let scale = wb[t.canon[cr * p + q] as usize];
                // Clipped samples end up at the white level in every channel (the output
                // clamps there too).
                planes[(q * nrb + lr) * pw + x / p] = (v * scale).min(1.0);
                q += 1;
                if q == p {
                    q = 0;
                }
            }
        }

        // Columns of phase q within [lo, hi): indices j with x = p*j + off(q).
        let jrange = |q: usize, lo: usize, hi: usize| {
            let o = off(q);
            let j0 = lo.saturating_sub(o).div_ceil(p);
            let j1 = hi.saturating_sub(o).div_ceil(p);
            (j0, j1.max(j0))
        };
        // Start of the run read by tap column `tx` for outputs `j0..` of phase `q` at local row `r`.
        let src = |r: usize, q: usize, tx: usize, j0: usize| {
            let (q2, js) = self.shifts[q][tx];
            (r + q2 * nrb) * pw + (j0 as isize + js) as usize
        };

        // 2. sqrt(G0) on rows y0-2..y1+2, columns xa-2..xb+2.
        let (gr, gc) = (nr + 4, nc + 4);
        let mut g = vec![0f32; gr * gc];
        let mut acc = vec![0f32; gc.div_ceil(p) + 2];
        for i in 0..gr {
            let y = y0 + i - 2;
            let cr = (y + self.cy) % p;
            let lr = y - by0;
            for q in 0..p {
                let (j0, j1) = jrange(q, xa - 2, xb + 2);
                if j1 == j0 {
                    continue;
                }
                let n = j1 - j0;
                let taps = &t.g0[(cr * p + q) * TAPS..][..TAPS];
                acc[..n].fill(0.0);
                for ty in 0..K {
                    for tx in 0..K {
                        let c = taps[ty * K + tx];
                        let s = src(lr + ty - R, q, tx, j0);
                        for (a, v) in acc[..n].iter_mut().zip(&planes[s..s + n]) {
                            *a += c * *v;
                        }
                    }
                }
                for (k, a) in acc[..n].iter().enumerate() {
                    let x = p * (j0 + k) + off(q);
                    g[i * gc + x - (xa - 2)] = a.max(0.0).sqrt();
                }
            }
        }

        // 3. Direction weights from the squared gradients of sqrt(G0): per direction the
        //    sum over 3x3 of the squared derivative, then w = 1/(E+eps)^2 normalised.
        let (dr, dc) = (nr + 2, nc + 2);
        let mut dsq: [Vec<f32>; NDIR] = std::array::from_fn(|_| vec![0f32; dr * dc]);
        for a in 0..dr {
            let rm = &g[a * gc..a * gc + gc];
            let r0 = &g[(a + 1) * gc..(a + 2) * gc];
            let rp = &g[(a + 2) * gc..(a + 3) * gc];
            for b in 0..dc {
                let dh = r0[b + 2] - r0[b];
                let dv = rp[b + 1] - rm[b + 1];
                let d1 = (rp[b + 2] - rm[b]) * FRAC_1_SQRT_2;
                let d2 = (rp[b] - rm[b + 2]) * FRAC_1_SQRT_2;
                dsq[0][a * dc + b] = dh * dh;
                dsq[1][a * dc + b] = dv * dv;
                dsq[2][a * dc + b] = d1 * d1;
                dsq[3][a * dc + b] = d2 * d2;
            }
        }
        let mut hsum: [Vec<f32>; NDIR] = std::array::from_fn(|_| vec![0f32; dr * nc]);
        for d in 0..NDIR {
            for a in 0..dr {
                let s = &dsq[d][a * dc..(a + 1) * dc];
                let o = &mut hsum[d][a * nc..(a + 1) * nc];
                for j in 0..nc {
                    o[j] = s[j] + s[j + 1] + s[j + 2];
                }
            }
        }
        let mut wts: [Vec<f32>; NDIR] = std::array::from_fn(|_| vec![0f32; nr * nc]);
        for i in 0..nr {
            let rows: [[&[f32]; 3]; NDIR] =
                std::array::from_fn(|d| std::array::from_fn(|a| &hsum[d][(i + a) * nc..(i + a + 1) * nc]));
            for j in 0..nc {
                let mut wd = [0f32; NDIR];
                for d in 0..NDIR {
                    let u = rows[d][0][j] + rows[d][1][j] + rows[d][2][j] + EPS;
                    wd[d] = 1.0 / (u * u);
                }
                let inv = 1.0 / (wd[0] + wd[1] + wd[2] + wd[3]);
                for d in 0..NDIR {
                    wts[d][i * nc + j] = wd[d] * inv;
                }
            }
        }

        // 4. The filters. Chunks of columns keep the accumulators in L1 while every tap's
        //    samples are used by all the filters of the pixel.
        const CH: usize = 128;
        let mut accs = vec![[0f32; CH]; 2 * NDIR];
        for i in 0..nr {
            let y = y0 + i;
            let lr = y - by0;
            let cr = (y + self.cy) % p;
            for q in 0..p {
                let (j0, j1) = jrange(q, xa, xb);
                if j1 == j0 {
                    continue;
                }
                let own = t.canon[cr * p + q] as usize;
                let miss: [usize; 2] = match own {
                    0 => [1, 2],
                    1 => [0, 2],
                    _ => [0, 1],
                };
                let base = (cr * p + q) * 3;
                let mut jc = j0;
                while jc < j1 {
                    let jn = (jc + CH).min(j1);
                    let n = jn - jc;
                    for a in accs.iter_mut() {
                        a[..n].fill(0.0);
                    }
                    for ty in 0..K {
                        for tx in 0..K {
                            let s = src(lr + ty - R, q, tx, jc);
                            let v = &planes[s..s + n];
                            for (m, &c) in miss.iter().enumerate() {
                                for d in 0..NDIR {
                                    let coef = t.dir[((base + c) * NDIR + d) * TAPS + ty * K + tx];
                                    for (a, x) in accs[m * NDIR + d][..n].iter_mut().zip(v) {
                                        *a += coef * *x;
                                    }
                                }
                            }
                        }
                    }
                    for k in 0..n {
                        let j = jc + k;
                        let x = p * j + off(q) - xa;
                        let o = i * nc + x;
                        let wd = [wts[0][o], wts[1][o], wts[2][o], wts[3][o]];
                        for (m, &c) in miss.iter().enumerate() {
                            let a = &accs[m * NDIR..];
                            out[c][o] = wd[0] * a[0][k] + wd[1] * a[1][k] + wd[2] * a[2][k] + wd[3] * a[3][k];
                        }
                        out[own][o] = planes[(lr + q * nrb) * pw + j];
                    }
                    jc = jn;
                }
            }
        }
    }
}

/// Medians of the 3x3 neighbourhoods of row `i + 1` of `a` (`ec` wide), for the columns
/// `1..ec - 1`, into `out`.
/// Smoothing (module docs, step 5): each pixel of the `nr` x `nc` core of `v` (rows of `ec`
/// pixels, one pixel of margin on every side) moves by `beta` towards the mean of its 3x3
/// window, a neighbour weighing `1 / (1 + e^2)` with `e = (G_neighbour - G) / (0.2 G + 0.015)`.
#[inline(always)]
fn smooth(v: &[[f32; 3]], ec: usize, beta: f32, nr: usize, nc: usize, out: &mut [[f32; 3]]) {
    for i in 0..nr {
        for j in 0..nc {
            let p = v[(i + 1) * ec + j + 1];
            let inv = 1.0 / (0.2 * p[1].max(0.0) + 0.015);
            let (mut sw, mut s) = (0f32, [0f32; 3]);
            for dy in 0..3 {
                for dx in 0..3 {
                    let q = v[(i + dy) * ec + j + dx];
                    let e = (q[1] - p[1]) * inv;
                    let wt = 1.0 / (1.0 + e * e);
                    sw += wt;
                    for c in 0..3 {
                        s[c] += wt * q[c];
                    }
                }
            }
            out[i * nc + j] = [0, 1, 2].map(|c| p[c] + beta * (s[c] / sw - p[c]));
        }
    }
}

fn median_row(a: &[f32], ec: usize, i: usize, out: &mut [f32]) {
    let n = out.len();
    let r0 = &a[i * ec..(i + 1) * ec];
    let r1 = &a[(i + 1) * ec..(i + 2) * ec];
    let r2 = &a[(i + 2) * ec..(i + 3) * ec];
    for j in 0..n {
        out[j] = median9([r0[j], r0[j + 1], r0[j + 2], r1[j], r1[j + 1], r1[j + 2], r2[j], r2[j + 1], r2[j + 2]]);
    }
}

/// Median of nine values (a fixed comparison network, branch free).
#[inline(always)]
fn median9(mut p: [f32; 9]) -> f32 {
    #[inline(always)]
    fn sort(p: &mut [f32; 9], a: usize, b: usize) {
        let (lo, hi) = (p[a].min(p[b]), p[a].max(p[b]));
        p[a] = lo;
        p[b] = hi;
    }
    sort(&mut p, 1, 2);
    sort(&mut p, 4, 5);
    sort(&mut p, 7, 8);
    sort(&mut p, 0, 1);
    sort(&mut p, 3, 4);
    sort(&mut p, 6, 7);
    sort(&mut p, 1, 2);
    sort(&mut p, 4, 5);
    sort(&mut p, 7, 8);
    sort(&mut p, 0, 3);
    sort(&mut p, 5, 8);
    sort(&mut p, 4, 7);
    sort(&mut p, 3, 6);
    sort(&mut p, 1, 4);
    sort(&mut p, 2, 5);
    sort(&mut p, 4, 7);
    sort(&mut p, 4, 2);
    sort(&mut p, 6, 4);
    sort(&mut p, 4, 2);
    p[4]
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashraw_core::CfaPattern;

    fn chan() -> [u8; 8] {
        [0, 1, 2, 3, 3, 3, 3, 3]
    }

    fn run(plan: &Plan, w: usize, h: usize, mosaic: &[f32], y: (usize, usize), x: (usize, usize)) -> Vec<[f32; 3]> {
        let mut out = vec![[0f32; 3]; (y.1 - y.0) * (x.1 - x.0)];
        let mut fetch = |yy: usize, row: &mut [f32]| row.copy_from_slice(&mosaic[yy * w..(yy + 1) * w]);
        let _ = h;
        plan.develop(w, y, x, &[1.0; 4], &mut fetch, &mut out);
        out
    }

    fn patterns() -> Vec<(&'static str, CfaPattern)> {
        vec![
            ("RGGB", CfaPattern::RGGB),
            ("GBRG", CfaPattern::GBRG),
            ("BGGR", CfaPattern::BGGR),
            ("xtrans", CfaPattern::parse("GGRGGB/GGBGGR/BRGRBG/GGBGGR/GGRGGB/RBGBRG").unwrap()),
            ("xtrans shifted", CfaPattern::parse("GGRGGB/GGBGGR/BRGRBG/GGBGGR/GGRGGB/RBGBRG").unwrap().shifted(2, 1)),
        ]
    }

    #[test]
    fn flat_field_stays_flat() {
        let (w, h) = (64, 48);
        for (name, cfa) in patterns() {
            let plan = Plan::new(&cfa, &chan(), (0, 0)).unwrap_or_else(|| panic!("{name}: no plan"));
            let m = vec![0.4f32; w * h];
            let out = run(&plan, w, h, &m, (HALO, h - HALO), (HALO, w - HALO));
            for px in out {
                for v in px {
                    assert!((v - 0.4).abs() < 1e-4, "{name}: {v}");
                }
            }
        }
    }

    #[test]
    fn keeps_measured_samples_and_interpolates_a_ramp() {
        let (w, h) = (64, 48);
        for (name, cfa) in patterns() {
            let plan = Plan::new(&cfa, &chan(), (0, 0)).unwrap();
            // Smooth grey ramp: every colour has the same value at a given position.
            let m: Vec<f32> = (0..w * h).map(|i| 0.2 + 0.004 * (i % w) as f32 + 0.002 * (i / w) as f32).collect();
            let out = run(&plan, w, h, &m, (HALO, h - HALO), (HALO, w - HALO));
            for (n, px) in out.iter().enumerate() {
                let (x, y) = (HALO + n % (w - 2 * HALO), HALO + n / (w - 2 * HALO));
                let want = m[y * w + x];
                let own = chan()[cfa.color_at(x, y) as usize] as usize;
                assert_eq!(px[own], want, "{name}: measured sample at ({x},{y})");
                for v in px {
                    assert!((v - want).abs() < 0.02, "{name}: ({x},{y}) {v} vs {want}");
                }
            }
        }
    }

    #[test]
    fn smoothing_keeps_bands_and_crops_identical_and_flat_fields_flat() {
        let (w, h) = (90, 60);
        let mut s = 777u32;
        let m: Vec<f32> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        for (name, cfa) in patterns() {
            let mut plan = Plan::new(&cfa, &chan(), (0, 0)).unwrap();
            plan.beta = 0.4;
            let all = run(&plan, w, h, &m, (HALO, h - HALO), (HALO, w - HALO));
            let nc = w - 2 * HALO;
            let part = run(&plan, w, h, &m, (9, 20), (13, 41));
            for (i, y) in (9..20).enumerate() {
                for (j, x) in (13..41).enumerate() {
                    assert_eq!(part[i * 28 + j], all[(y - HALO) * nc + x - HALO], "{name}: ({x},{y})");
                }
            }
            let flat = vec![0.3f32; w * h];
            for px in run(&plan, w, h, &flat, (HALO, h - HALO), (HALO, w - HALO)) {
                for c in px {
                    assert!((c - 0.3).abs() < 1e-5, "{name}: {px:?}");
                }
            }
            // Smoothing changes the result of noise, not of nothing.
            plan.beta = 0.0;
            assert_ne!(run(&plan, w, h, &m, (HALO, h - HALO), (HALO, w - HALO)), all, "{name}");
        }
    }

    /// The AVX2 build of the develop must give exactly the baseline's bits.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn avx2_develop_is_bit_identical_to_the_baseline() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let (w, h) = (90, 60);
        let mut s = 4242u32;
        let m: Vec<f32> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        for (name, cfa) in patterns() {
            let mut plan = Plan::new(&cfa, &chan(), (0, 0)).unwrap();
            for beta in [0.0, 0.4] {
                plan.beta = beta;
                let (y, x) = ((HALO, h - HALO), (HALO, w - HALO));
                let mut a = vec![[0f32; 3]; (y.1 - y.0) * (x.1 - x.0)];
                let mut b = a.clone();
                let mut fetch = |yy: usize, row: &mut [f32]| row.copy_from_slice(&m[yy * w..(yy + 1) * w]);
                plan.develop_impl(w, y, x, &[1.1, 1.0, 0.9, 1.0], &mut fetch, &mut a);
                // SAFETY: AVX2 was detected above.
                unsafe { plan.develop_avx2(w, y, x, &[1.1, 1.0, 0.9, 1.0], &mut fetch, &mut b) };
                assert!(
                    a.iter().zip(&b).all(|(p, q)| p.iter().zip(q).all(|(u, v)| u.to_bits() == v.to_bits())),
                    "{name} {beta}"
                );
            }
        }
    }

    #[test]
    fn noise_estimate_separates_clean_from_noisy_frames() {
        let (w, h) = (256, 256);
        let (_, cfa) = patterns().remove(0);
        let mut s = 99u32;
        let mut gauss = || {
            let mut u = || {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 + 0.5) / (1u32 << 24) as f32
            };
            (-2.0 * u().ln()).sqrt() * (std::f32::consts::TAU * u()).cos()
        };
        let beta_for = |sigma_rel: f32, gauss: &mut dyn FnMut() -> f32| {
            // A smooth ramp with noise of variance sigma_rel^2 * value.
            let m: Vec<f32> = (0..w * h)
                .map(|i| {
                    let v = 0.1 + 0.3 * ((i % w) as f32 / w as f32);
                    v + gauss() * (sigma_rel * sigma_rel * v).sqrt()
                })
                .collect();
            let mut plan = Plan::new(&cfa, &chan(), (0, 0)).unwrap();
            let mut fetch = |yy: usize, row: &mut [f32]| row.copy_from_slice(&m[yy * w..(yy + 1) * w]);
            plan.set_noise_from(w, h, 1.0, &mut fetch);
            plan.beta
        };
        assert_eq!(beta_for(0.005, &mut gauss), 0.0);
        let noisy = beta_for(0.03, &mut gauss);
        assert!(noisy > 0.45 && noisy <= 0.5, "{noisy}");
        let mid = beta_for(0.017, &mut gauss);
        assert!(mid > 0.0 && mid < 0.5, "{mid}");
    }

    #[test]
    fn bands_and_crops_agree_with_the_whole_frame() {
        let (w, h) = (70, 53);
        let mut s = 12345u32;
        let m: Vec<f32> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        for (name, cfa) in patterns() {
            let plan = Plan::new(&cfa, &chan(), (0, 0)).unwrap();
            let all = run(&plan, w, h, &m, (HALO, h - HALO), (HALO, w - HALO));
            let nc = w - 2 * HALO;
            // Rows 9..20, columns 13..41.
            let part = run(&plan, w, h, &m, (9, 20), (13, 41));
            for (i, y) in (9..20).enumerate() {
                for (j, x) in (13..41).enumerate() {
                    assert_eq!(part[i * 28 + j], all[(y - HALO) * nc + x - HALO], "{name}: ({x},{y})");
                }
            }
        }
    }

    /// Sharp coloured shapes (edges at all angles, thin lines), mosaicked and developed:
    /// the edge-aware demosaic must be clearly closer to the truth than the 3x3 average.
    #[test]
    fn beats_the_3x3_average_on_sharp_edges() {
        let (w, h) = (96, 96);
        let truth: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let disc = ((x - 40.0).powi(2) + (y - 50.0).powi(2)).sqrt() < 26.0;
                let diag = (x * 0.8 + y * 0.6) as i32 % 18 < 6;
                let line = (y - 20.0).abs() < 0.8 || (x - 70.0).abs() < 0.8;
                match (disc, diag, line) {
                    (_, _, true) => [0.9, 0.1, 0.1],
                    (true, true, _) => [0.1, 0.8, 0.3],
                    (true, false, _) => [0.2, 0.3, 0.9],
                    (false, true, _) => [0.8, 0.7, 0.2],
                    _ => [0.3, 0.3, 0.3],
                }
            })
            .collect();
        for (name, cfa) in patterns() {
            let plan = Plan::new(&cfa, &chan(), (0, 0)).unwrap();
            let mosaic: Vec<f32> =
                (0..w * h).map(|i| truth[i][chan()[cfa.color_at(i % w, i / w) as usize] as usize]).collect();
            let out = run(&plan, w, h, &mosaic, (HALO, h - HALO), (HALO, w - HALO));
            let nc = w - 2 * HALO;
            // Reference: average of the same-colour samples in the 3x3 window.
            let bil = |x: usize, y: usize, c: usize| {
                let (mut s, mut n) = (0.0, 0.0);
                for (dy, dx) in (0..3).flat_map(|a| (0..3).map(move |b| (a, b))) {
                    let (xx, yy) = (x + dx - 1, y + dy - 1);
                    if chan()[cfa.color_at(xx, yy) as usize] as usize == c {
                        s += mosaic[yy * w + xx];
                        n += 1.0;
                    }
                }
                if n > 0.0 { s / n } else { f32::NAN }
            };
            let (mut e_edge, mut e_bil, mut count) = (0.0f64, 0.0f64, 0.0f64);
            for y in HALO..h - HALO {
                for x in HALO..w - HALO {
                    let px = out[(y - HALO) * nc + x - HALO];
                    for c in 0..3 {
                        let b = bil(x, y, c);
                        if b.is_nan() {
                            continue;
                        }
                        e_edge += ((px[c] - truth[y * w + x][c]) as f64).powi(2);
                        e_bil += ((b - truth[y * w + x][c]) as f64).powi(2);
                        count += 1.0;
                    }
                }
            }
            let (m_edge, m_bil) = (e_edge / count, e_bil / count);
            eprintln!("{name}: edge-aware MSE {m_edge:.5} vs 3x3 average {m_bil:.5}");
            assert!(m_edge < m_bil, "{name}: edge-aware MSE {m_edge:.5} vs 3x3 average {m_bil:.5}");
        }
    }

    #[test]
    fn median9_is_the_median() {
        let mut s = 99u32;
        for _ in 0..2000 {
            let v: [f32; 9] = std::array::from_fn(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 16) % 7) as f32
            });
            let mut sorted = v;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(median9(v), sorted[4], "{v:?}");
        }
    }

    #[test]
    fn unknown_patterns_have_no_plan() {
        // Greens in a column instead of on a diagonal: not a shifted Bayer pattern.
        let rgbg = CfaPattern::parse("RGBG").unwrap();
        assert!(Plan::new(&rgbg, &chan(), (0, 0)).is_none());
    }
}
