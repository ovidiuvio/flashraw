//! Alternative Bayer demosaics ([`crate::Demosaic::Ppg`], [`crate::Demosaic::Rcd`]). Both reconstruct green first and then
//! red and blue from colour differences; they differ in how green is estimated and how the
//! diagonal differences are weighted. Each pixel's result depends only on the samples within [`HALO`], so
//! bands, crops and thread counts give identical output. Pixels closer than [`HALO`] to the frame
//! border use bilinear.
//!
//! * **Ppg** (patterned pixel grouping): green at a red or blue sample is interpolated along the
//!   row or the column, whichever has the smaller gradient, with a second-difference correction
//!   limited to the range of the two greens. Red at a blue sample (and the reverse) is the green
//!   plus the mean colour difference along the diagonal of smaller gradient; red and blue at a
//!   green sample are the green plus the mean colour difference of the two neighbours in the row
//!   or in the column.
//! * **Rcd** (ratio corrected, written from the multiplicative colour model: where the image
//!   scales by a factor from one pixel to the next, the green at a pixel is the neighbouring
//!   green times the ratio of the local intensities, estimated with a low-pass of the mosaic):
//!   four directional estimates `g_n * 2 L_c / (L_c + L_2n)` combined with inverse-gradient
//!   weights, then blended between the row and column results by how much the image varies across
//!   them. Red and blue as in Ppg with soft diagonal weights.
//!
//! Residual interpolation (a guided-filter prediction of red and blue from green, corrected by the
//! interpolated residuals at the samples) was implemented and benchmarked on the same tiles; it
//! scored below the colour-difference method of [`Method::Rcd`] on every measure at every noise
//! level, so it was not kept.

use flashraw_core::CfaPattern;

/// Mosaic samples needed around a pixel (even, so the colour of a pixel follows from the parity of
/// its frame coordinates).
pub(crate) const HALO: usize = 8;
/// Output rows per band.
pub(crate) const BAND_ROWS: usize = 64;

/// The algorithm of a plan.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Method {
    Ppg,
    Rcd,
}

/// One frame's plan: the colour of each of the four positions of the 2x2 pattern, indexed by
/// `(y & 1) * 2 + (x & 1)` of the frame.
#[derive(Clone)]
pub(crate) struct Plan {
    col: [u8; 4],
    method: Method,
}

impl Plan {
    /// The plan for a frame whose sample `(0, 0)` is at sensor position `origin`, or `None` when
    /// the pattern is not a Bayer arrangement.
    pub(crate) fn new(method: Method, cfa: &CfaPattern, chan: &[u8; 8], origin: (usize, usize)) -> Option<Plan> {
        crate::ahd::Plan::new(cfa, chan, origin).map(|p| Plan { col: p.colours(), method })
    }

    /// Develop rows `y0..y1` and columns `xa..xb` (frame coordinates) into `out`
    /// (`(y1 - y0) * (xb - xa)` RGB pixels, white balanced by `wb`). Needs `HALO <= y0`,
    /// `y1 + HALO <= height`, `HALO <= xa`, `xb + HALO <= w`. `fetch(y, row)` fills `row` with
    /// the normalised mosaic row `y`.
    pub(crate) fn develop(
        &self,
        w: usize,
        rows: (usize, usize),
        cols: (usize, usize),
        wb: &[f32; 4],
        fetch: &mut dyn FnMut(usize, &mut [f32]),
        out: &mut [[f32; 3]],
    ) {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked above). Compiled without FMA, so the
            // results equal the baseline build's bit for bit.
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
        let (er, bw) = (nr + 2 * HALO, nc + 2 * HALO);
        let mut m = vec![0f32; er * bw];
        let mut row = vec![0f32; w];
        for i in 0..er {
            let y = y0 + i - HALO;
            fetch(y, &mut row);
            for j in 0..bw {
                let x = xa + j - HALO;
                m[i * bw + j] = (row[x] * wb[self.col[(y & 1) * 2 + (x & 1)] as usize]).min(1.0);
            }
        }
        let band = Band { m: &m, er, bw, col: &self.col, py: y0 & 1, px: xa & 1 };
        let g = band.green(self.method);
        band.diagonal(self.method, &g, out, nr, nc);
    }
}

/// A band of the balanced mosaic: `er` rows of `bw` samples, starting `HALO` before the first
/// output row and column.
struct Band<'a> {
    m: &'a [f32],
    er: usize,
    bw: usize,
    col: &'a [u8; 4],
    /// Parity of the frame row / column of band row / column `HALO` (HALO is even).
    py: usize,
    px: usize,
}

impl Band<'_> {
    /// Colour of the sample at band row `i`, column `j`.
    #[inline(always)]
    fn colour(&self, i: usize, j: usize) -> u8 {
        self.col[((i + self.py) & 1) * 2 + ((j + self.px) & 1)]
    }

    /// First column `>= 2` of row `i` whose sample is not green (they alternate with greens).
    #[inline(always)]
    fn first_non_green(&self, i: usize) -> usize {
        if self.colour(i, 2) != 1 { 2 } else { 3 }
    }

    /// The green plane: measured at green samples, estimated elsewhere, valid for rows and
    /// columns `3..` from the edges (the support of the estimates is 3 samples).
    fn green(&self, method: Method) -> Vec<f32> {
        let (er, bw, m) = (self.er, self.bw, self.m);
        let mut g = m.to_vec();
        match method {
            Method::Ppg => {
                for i in 2..er - 2 {
                    let mut j = self.first_non_green(i);
                    while j < bw - 2 {
                        let k = i * bw + j;
                        let c = m[k];
                        let (gn, gs, gw, ge) = (m[k - bw], m[k + bw], m[k - 1], m[k + 1]);
                        let (n2, s2, w2, e2) = (m[k - 2 * bw], m[k + 2 * bw], m[k - 2], m[k + 2]);
                        let dv = (gn - gs).abs() + (2.0 * c - n2 - s2).abs();
                        let dh = (gw - ge).abs() + (2.0 * c - w2 - e2).abs();
                        let ev = ((gn + gs) * 0.5 + (2.0 * c - n2 - s2) * 0.25).clamp(gn.min(gs), gn.max(gs));
                        let eh = ((gw + ge) * 0.5 + (2.0 * c - w2 - e2) * 0.25).clamp(gw.min(ge), gw.max(ge));
                        g[k] = if dv < dh {
                            ev
                        } else if dh < dv {
                            eh
                        } else {
                            (ev + eh) * 0.5
                        };
                        j += 2;
                    }
                }
            }
            Method::Rcd => self.green_ratio(&mut g),
        }
        g
    }

    /// Green by ratio-corrected directional estimates (module docs).
    fn green_ratio(&self, g: &mut [f32]) {
        const EPS: f32 = 1e-4;
        let (er, bw, m) = (self.er, self.bw, self.m);
        // Low pass of the mosaic, `[1 2 1; 2 4 2; 1 2 1] / 16`, for rows and columns 1.. from the edges.
        let mut l = vec![0f32; er * bw];
        for i in 1..er - 1 {
            for j in 1..bw - 1 {
                let k = i * bw + j;
                l[k] = (4.0 * m[k]
                    + 2.0 * (m[k - 1] + m[k + 1] + m[k - bw] + m[k + bw])
                    + (m[k - bw - 1] + m[k - bw + 1] + m[k + bw - 1] + m[k + bw + 1]))
                    * (1.0 / 16.0);
            }
        }
        for i in 3..er - 3 {
            let mut j = if self.colour(i, 3) != 1 { 3 } else { 4 };
            while j < bw - 3 {
                let k = i * bw + j;
                let (lc, ln, ls, lw, le) = (l[k], l[k - bw], l[k + bw], l[k - 1], l[k + 1]);
                let (ln2, ls2, lw2, le2) = (l[k - 2 * bw], l[k + 2 * bw], l[k - 2], l[k + 2]);
                // The ratio-corrected estimate from each neighbouring green: the intensity at the
                // neighbour is taken as the mean of the low pass at the centre and two away.
                let est = |gv: f32, l2: f32| gv * 2.0 * lc / (EPS + lc + l2);
                let (en, es, ew, ee) =
                    (est(m[k - bw], ln2), est(m[k + bw], ls2), est(m[k - 1], lw2), est(m[k + 1], le2));
                let wgt = |a: f32, b: f32| {
                    let d = EPS + (a - b).abs();
                    1.0 / (d * d)
                };
                // Weights from the low pass: the closer a green's intensity is to the centre's,
                // the more it counts.
                let (wn, ws, ww, we) = (wgt(ln, lc), wgt(ls, lc), wgt(lw, lc), wgt(le, lc));
                let gv = (wn * en + ws * es) / (wn + ws);
                let gh = (ww * ew + we * ee) / (ww + we);
                // Trust the direction across which the image varies less.
                let ev = (ln - ls).abs() + (l[k - 2 * bw] - lc).abs() + (l[k + 2 * bw] - lc).abs();
                let eh = (lw - le).abs() + (l[k - 2] - lc).abs() + (l[k + 2] - lc).abs();
                let a = (eh * eh + EPS) / (ev * ev + eh * eh + 2.0 * EPS);
                let est_g = a * gv + (1.0 - a) * gh;
                let (lo, hi) = (
                    m[k - bw].min(m[k + bw]).min(m[k - 1]).min(m[k + 1]),
                    m[k - bw].max(m[k + bw]).max(m[k - 1]).max(m[k + 1]),
                );
                g[k] = est_g.clamp(lo, hi);
                j += 2;
            }
        }
    }

    /// Red and blue from colour differences, written to `out` for the `nr` x `nc` core.
    fn diagonal(&self, method: Method, g: &[f32], out: &mut [[f32; 3]], nr: usize, nc: usize) {
        const EPS: f32 = 1e-4;
        let (bw, m) = (self.bw, self.m);
        for oi in 0..nr {
            let i = oi + HALO;
            for oj in 0..nc {
                let j = oj + HALO;
                let k = i * bw + j;
                let c = self.colour(i, j);
                let e = |q: usize| m[q] - g[q];
                let rgb = if c == 1 {
                    // Green sample: the colour of its row neighbours from the row, the other
                    // from the column.
                    let h = 0.5 * (e(k - 1) + e(k + 1));
                    let v = 0.5 * (e(k - bw) + e(k + bw));
                    if self.colour(i, j + 1) == 0 { [g[k] + h, g[k], g[k] + v] } else { [g[k] + v, g[k], g[k] + h] }
                } else {
                    let (ne, nw, se, sw) = (k - bw + 1, k - bw - 1, k + bw + 1, k + bw - 1);
                    let d1 = (m[ne] - m[sw]).abs() + (2.0 * g[k] - g[ne] - g[sw]).abs();
                    let d2 = (m[nw] - m[se]).abs() + (2.0 * g[k] - g[nw] - g[se]).abs();
                    let other = if method == Method::Rcd {
                        let (w1, w2) = (1.0 / ((EPS + d1) * (EPS + d1)), 1.0 / ((EPS + d2) * (EPS + d2)));
                        g[k] + (w1 * 0.5 * (e(ne) + e(sw)) + w2 * 0.5 * (e(nw) + e(se))) / (w1 + w2)
                    } else if d1 < d2 {
                        g[k] + 0.5 * (e(ne) + e(sw))
                    } else if d2 < d1 {
                        g[k] + 0.5 * (e(nw) + e(se))
                    } else {
                        g[k] + 0.25 * (e(ne) + e(sw) + e(nw) + e(se))
                    };
                    if c == 0 { [m[k], g[k], other] } else { [other, g[k], m[k]] }
                };
                out[oi * nc + oj] = rgb;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flashraw_core::CfaPattern;

    fn chan() -> [u8; 8] {
        let mut c = [255u8; 8];
        c[..3].copy_from_slice(&[0, 1, 2]);
        c
    }

    fn scene(w: usize, h: usize) -> (Vec<f32>, Vec<[f32; 3]>) {
        let mut rgb = vec![[0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / w as f32;
                let edge = if x > w / 2 { 0.4 } else { 0.0 };
                rgb[y * w + x] = [0.2 + 0.3 * t + edge, 0.3 + 0.2 * (y as f32 / h as f32) + edge * 0.5, 0.6 - 0.3 * t];
            }
        }
        let m = (0..w * h)
            .map(|k| {
                let (x, y) = (k % w, k / w);
                rgb[k][[0usize, 1, 1, 2][(y & 1) * 2 + (x & 1)]]
            })
            .collect();
        (m, rgb)
    }

    fn run(plan: &Plan, w: usize, m: &[f32], rows: (usize, usize), cols: (usize, usize)) -> Vec<[f32; 3]> {
        let mut out = vec![[0f32; 3]; (rows.1 - rows.0) * (cols.1 - cols.0)];
        let mut fetch = |y: usize, r: &mut [f32]| r.copy_from_slice(&m[y * w..(y + 1) * w]);
        plan.develop(w, rows, cols, &[1.0; 4], &mut fetch, &mut out);
        out
    }

    const METHODS: [Method; 2] = [Method::Ppg, Method::Rcd];

    #[test]
    fn smooth_fields_are_reproduced_and_measured_samples_kept() {
        let (w, h) = (90, 80);
        let (m, rgb) = scene(w, h);
        for method in METHODS {
            let plan = Plan::new(method, &CfaPattern::RGGB, &chan(), (0, 0)).unwrap();
            let got = run(&plan, w, &m, (HALO, h - HALO), (HALO, w - HALO));
            let nc = w - 2 * HALO;
            let mut worst = 0f32;
            for y in HALO..h - HALO {
                for x in HALO..w / 2 - 6 {
                    let p = got[(y - HALO) * nc + x - HALO];
                    for c in 0..3 {
                        worst = worst.max((p[c] - rgb[y * w + x][c]).abs());
                    }
                }
            }
            assert!(worst < 0.03, "{method:?} {worst}");
            for y in HALO..h - HALO {
                for x in HALO..w - HALO {
                    let c = [0usize, 1, 1, 2][(y & 1) * 2 + (x & 1)];
                    assert_eq!(got[(y - HALO) * nc + x - HALO][c], m[y * w + x], "{method:?} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn bands_and_crops_agree_with_the_whole_frame() {
        let (w, h) = (130, 70);
        let mut s = 11u32;
        let m: Vec<f32> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        for method in METHODS {
            for origin in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let plan = Plan::new(method, &CfaPattern::RGGB, &chan(), origin).unwrap();
                let all = run(&plan, w, &m, (HALO, h - HALO), (HALO, w - HALO));
                let nc = w - 2 * HALO;
                let part = run(&plan, w, &m, (11, 29), (13, 77));
                for (i, y) in (11..29).enumerate() {
                    for (j, x) in (13..77).enumerate() {
                        assert_eq!(
                            part[i * 64 + j],
                            all[(y - HALO) * nc + x - HALO],
                            "{method:?} {origin:?} ({x},{y})"
                        );
                    }
                }
            }
        }
    }
}
