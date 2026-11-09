//! Homogeneity-directed demosaic ([`crate::Demosaic::Ahd`]) for Bayer sensors.
//!
//! The method of Hirakawa and Parks (adaptive homogeneity-directed demosaicing, 2005), written
//! from the paper's description:
//!
//! 1. **Two candidates.** Green is interpolated along rows (candidate H) and along columns
//!    (candidate V) with the second-difference corrected average of the two nearest greens,
//!    limited to the range of those greens. Red and blue come from colour differences: the
//!    missing colour of a pixel is its green plus the mean of the differences `R - G` (or
//!    `B - G`) measured in the 3x3 window, where each candidate uses its own green plane.
//! 2. **Perceptual space.** Each candidate is mapped to a lightness `L` and two chroma
//!    coordinates `a`, `b` (square-root encoded red, green and blue: `L` is their weighted mean,
//!    `a = R' - G'`, `b = B' - G'`, a cheap stand-in for CIELab that gives the same results on
//!    the benchmarks).
//! 3. **Homogeneity.** For a pixel, the thresholds are the smaller of the largest lightness
//!    (chroma) steps to its two neighbours along the row in candidate H and along the column in
//!    candidate V. A candidate's homogeneity there is the number of the 8 neighbours whose
//!    lightness and chroma are both within the thresholds of the pixel; it is summed over the
//!    3x3 window.
//! 4. **Choice.** The candidate with the higher summed homogeneity wins; equal sums average
//!    the two.
//!
//! The support of a pixel is 6 samples (green 2, colour differences 1, steps 1, neighbours 1,
//! window 1), so results do not depend on the band, the column chunk, the thread count or the
//! crop. Pixels closer than 6 samples to the frame border use bilinear.

use flashraw_core::CfaPattern;

/// Mosaic samples needed around a pixel.
pub(crate) const HALO: usize = 6;
/// Output rows per band: bounds the working memory.
pub(crate) const BAND_ROWS: usize = 64;
/// Output columns developed together inside a band.
const CHUNK: usize = 192;

/// The homogeneity-directed demosaic of one frame: the colour (0 R, 1 G, 2 B) of each of the
/// four positions of the 2x2 pattern, indexed by `(y & 1) * 2 + (x & 1)` of the frame.
#[derive(Clone)]
pub(crate) struct Plan {
    col: [u8; 4],
}

impl Plan {
    /// The plan for a frame whose sample `(0, 0)` is at sensor position `origin`, or `None`
    /// when the pattern is not a Bayer arrangement. `chan` maps `CfaColor` values to channels.
    pub(crate) fn new(cfa: &CfaPattern, chan: &[u8; 8], origin: (usize, usize)) -> Option<Plan> {
        if cfa.width() != 2 || cfa.height() != 2 {
            return None;
        }
        let mut col = [0u8; 4];
        for (i, c) in col.iter_mut().enumerate() {
            *c = chan[cfa.color_at(origin.0 + (i & 1), origin.1 + (i >> 1)) as usize];
        }
        let greens = (col[0] == 1 && col[3] == 1) || (col[1] == 1 && col[2] == 1);
        let others = |a: u8, b: u8| (a == 0 && b == 2) || (a == 2 && b == 0);
        let ok = greens && if col[0] == 1 { others(col[1], col[2]) } else { others(col[0], col[3]) };
        ok.then_some(Plan { col })
    }

    /// Develop rows `y0..y1` (frame coordinates) and columns `xa..xb` into `out`
    /// (`(y1 - y0) * (xb - xa)` camera RGB pixels, white balanced by `wb`, indexed by channel).
    /// Needs `HALO <= y0`, `y1 + HALO <= height`, `HALO <= xa`, `xb + HALO <= w`.
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
        // The same source compiled for AVX2 (no FMA, so every lane performs the same operations
        // in the same order: bit-identical to the baseline build) vectorises the plane loops.
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
        let er = nr + 2 * HALO;
        let bw = nc + 2 * HALO;
        // The balanced mosaic of the band, clipped at the white level like the output.
        let mut m = vec![0f32; er * bw];
        let mut row = vec![0f32; w];
        for i in 0..er {
            let y = y0 + i - HALO;
            fetch(y, &mut row);
            for j in 0..bw {
                let x = xa + j - HALO;
                let c = self.col[(y & 1) * 2 + (x & 1)] as usize;
                m[i * bw + j] = (row[x] * wb[c]).min(1.0);
            }
        }
        let mut work = Work::new(er, CHUNK + 2 * HALO);
        let mut c0 = 0;
        while c0 < nc {
            let c1 = (c0 + CHUNK).min(nc);
            self.chunk(&m, bw, (y0, xa, c0, c1), er, &mut work, out, nc);
            c0 = c1;
        }
    }

    /// One column chunk `c0..c1` (band columns) of the band.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn chunk(
        &self,
        m: &[f32],
        bw: usize,
        (y0, xa, c0, c1): (usize, usize, usize, usize),
        er: usize,
        wk: &mut Work,
        out: &mut [[f32; 3]],
        nc: usize,
    ) {
        let ec = c1 - c0 + 2 * HALO;
        // Local copy of the chunk's mosaic: local column j is band column c0 + j.
        for i in 0..er {
            wk.m[i * ec..(i + 1) * ec].copy_from_slice(&m[i * bw + c0..i * bw + c0 + ec]);
        }
        // Colour of local (i, j): frame row y0 - HALO + i, column xa - HALO + c0 + j.
        let (py, px) = ((y0 + HALO) & 1, (xa + HALO + c0) & 1); // parity of y0-HALO, xa-HALO+c0 (HALO is even)
        let colour = |i: usize, j: usize| self.col[((i + py) & 1) * 2 + ((j + px) & 1)] as usize;
        let at = |i: usize, j: usize| i * ec + j;

        // 1. Green along rows (H) and columns (V).
        for i in 2..er - 2 {
            for j in 2..ec - 2 {
                let k = at(i, j);
                let c = wk.m[k];
                if colour(i, j) == 1 {
                    wk.g[0][k] = c;
                    wk.g[1][k] = c;
                    continue;
                }
                let (a, b) = (wk.m[k - 1], wk.m[k + 1]);
                let h = (a + b) * 0.5 + (2.0 * c - wk.m[k - 2] - wk.m[k + 2]) * 0.25;
                wk.g[0][k] = h.clamp(a.min(b), a.max(b));
                let (a, b) = (wk.m[k - ec], wk.m[k + ec]);
                let v = (a + b) * 0.5 + (2.0 * c - wk.m[k - 2 * ec] - wk.m[k + 2 * ec]) * 0.25;
                wk.g[1][k] = v.clamp(a.min(b), a.max(b));
            }
        }

        // 2. Red and blue by colour differences, then the perceptual coordinates.
        for d in 0..2 {
            for i in 3..er - 3 {
                for j in 3..ec - 3 {
                    let k = at(i, j);
                    let g = wk.g[d][k];
                    let diff = |kk: usize| wk.m[kk] - wk.g[d][kk];
                    let rgb = match colour(i, j) {
                        0 => {
                            let b =
                                g + 0.25 * (diff(k - ec - 1) + diff(k - ec + 1) + diff(k + ec - 1) + diff(k + ec + 1));
                            [wk.m[k], g, b]
                        }
                        2 => {
                            let r =
                                g + 0.25 * (diff(k - ec - 1) + diff(k - ec + 1) + diff(k + ec - 1) + diff(k + ec + 1));
                            [r, g, wk.m[k]]
                        }
                        _ => {
                            // Green: the pixel's row neighbours are of one colour, its column
                            // neighbours of the other.
                            let along_row = 0.5 * (diff(k - 1) + diff(k + 1));
                            let along_col = 0.5 * (diff(k - ec) + diff(k + ec));
                            if colour(i, j + 1) == 0 {
                                [g + along_row, g, g + along_col]
                            } else {
                                [g + along_col, g, g + along_row]
                            }
                        }
                    };
                    wk.rgb[d][k] = rgb;
                    let s = rgb.map(|v| v.max(0.0).sqrt());
                    wk.lab[d][0][k] = (s[0] + 2.0 * s[1] + s[2]) * 0.25;
                    wk.lab[d][1][k] = s[0] - s[1];
                    wk.lab[d][2][k] = s[2] - s[1];
                }
            }
        }

        // 3. Thresholds: the smaller of the largest steps along the row in H and along the
        //    column in V (chroma as a squared distance).
        let step = |lab: &[Vec<f32>; 3], k: usize, o: usize| {
            let (l, a, b) = (lab[0][k], lab[1][k], lab[2][k]);
            let dl = (lab[0][k - o] - l).abs().max((lab[0][k + o] - l).abs());
            let sq = |kk: usize| (lab[1][kk] - a) * (lab[1][kk] - a) + (lab[2][kk] - b) * (lab[2][kk] - b);
            (dl, sq(k - o).max(sq(k + o)))
        };
        for i in 4..er - 4 {
            for j in 4..ec - 4 {
                let k = at(i, j);
                let (lh, ch) = step(&wk.lab[0], k, 1);
                let (lv, cv) = step(&wk.lab[1], k, ec);
                wk.eps[0][k] = lh.min(lv);
                wk.eps[1][k] = ch.min(cv);
            }
        }

        // 4. Homogeneity: neighbours within the thresholds, counted per row of the 3x3 window.
        for d in 0..2 {
            let [pl, pa, pb] = &wk.lab[d];
            for i in 5..er - 5 {
                let n = ec - 10;
                let base = i * ec + 5;
                let (el, ecr) = (&wk.eps[0][base..base + n], &wk.eps[1][base..base + n]);
                let (cl, ca, cb) = (&pl[base..base + n], &pa[base..base + n], &pb[base..base + n]);
                let cnt = &mut wk.hom[d][base..base + n];
                cnt.fill(0.0);
                for (r, dxs) in [(-1isize, [-1isize, 0, 1]), (0, [-1, 1, 0]), (1, [-1, 0, 1])] {
                    let rb = (base as isize + r * ec as isize) as usize;
                    for (t, dx) in dxs.iter().enumerate() {
                        if r == 0 && t == 2 {
                            continue; // the pixel itself
                        }
                        let o = (rb as isize + dx) as usize;
                        let (ql, qa, qb) = (&pl[o..o + n], &pa[o..o + n], &pb[o..o + n]);
                        for x in 0..n {
                            let dc = (qa[x] - ca[x]) * (qa[x] - ca[x]) + (qb[x] - cb[x]) * (qb[x] - cb[x]);
                            cnt[x] += f32::from(u8::from((ql[x] - cl[x]).abs() <= el[x]) & u8::from(dc <= ecr[x]));
                        }
                    }
                }
            }
        }

        // 5. The candidate with the larger summed homogeneity.
        for i in HALO..er - HALO {
            for j in HALO..ec - HALO {
                let k = at(i, j);
                let sum = |h: &[f32]| {
                    let mut s = 0f32;
                    for r in [k - ec, k, k + ec] {
                        s += h[r - 1] + h[r] + h[r + 1];
                    }
                    s
                };
                let (sh, sv) = (sum(&wk.hom[0]), sum(&wk.hom[1]));
                let wh = if sh > sv {
                    1.0
                } else if sv > sh {
                    0.0
                } else {
                    0.5
                };
                let (a, b) = (wk.rgb[0][k], wk.rgb[1][k]);
                out[(i - HALO) * nc + c0 + j - HALO] = [0, 1, 2].map(|c| wh * a[c] + (1.0 - wh) * b[c]);
            }
        }
    }
}

/// Working planes of one chunk (`er` rows of up to `CHUNK + 2 * HALO` columns).
struct Work {
    m: Vec<f32>,
    g: [Vec<f32>; 2],
    rgb: [Vec<[f32; 3]>; 2],
    lab: [[Vec<f32>; 3]; 2],
    eps: [Vec<f32>; 2],
    hom: [Vec<f32>; 2],
}

impl Work {
    fn new(er: usize, ec: usize) -> Work {
        let n = er * ec;
        Work {
            m: vec![0.0; n],
            g: [vec![0.0; n], vec![0.0; n]],
            rgb: [vec![[0.0; 3]; n], vec![[0.0; 3]; n]],
            lab: std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; n])),
            eps: [vec![0.0; n], vec![0.0; n]],
            hom: [vec![0.0; n], vec![0.0; n]],
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

    /// A smooth coloured scene with a hard diagonal-free edge, mosaicked as RGGB.
    fn scene(w: usize, h: usize) -> (Vec<f32>, Vec<[f32; 3]>) {
        let mut rgb = vec![[0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / w as f32;
                let edge = if x > w / 2 { 0.5 } else { 0.0 };
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

    #[test]
    fn rejects_other_patterns_and_accepts_shifted_bayer() {
        assert!(Plan::new(&CfaPattern::RGGB, &chan(), (0, 0)).is_some());
        assert!(Plan::new(&CfaPattern::RGGB, &chan(), (1, 1)).is_some());
        assert!(Plan::new(&CfaPattern::RGGB, &chan(), (1, 0)).is_some());
    }

    #[test]
    fn flat_and_smooth_fields_are_reproduced() {
        let (w, h) = (80, 70);
        let (m, rgb) = scene(w, h);
        let plan = Plan::new(&CfaPattern::RGGB, &chan(), (0, 0)).unwrap();
        let got = run(&plan, w, &m, (HALO, h - HALO), (HALO, w - HALO));
        // Away from the edge the smooth scene is recovered closely.
        let mut worst = 0f32;
        for y in HALO..h - HALO {
            for x in HALO..w / 2 - 4 {
                let g = got[(y - HALO) * (w - 2 * HALO) + x - HALO];
                for c in 0..3 {
                    worst = worst.max((g[c] - rgb[y * w + x][c]).abs());
                }
            }
        }
        assert!(worst < 0.02, "{worst}");
    }

    #[test]
    fn bands_chunks_and_crops_agree_with_the_whole_frame() {
        let (w, h) = (430, 61);
        let mut s = 7u32;
        let m: Vec<f32> = (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        for origin in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let plan = Plan::new(&CfaPattern::RGGB, &chan(), origin).unwrap();
            let all = run(&plan, w, &m, (HALO, h - HALO), (HALO, w - HALO));
            let nc = w - 2 * HALO;
            // Rows 9..20 and columns 13..300 cross chunk borders (192 columns).
            let part = run(&plan, w, &m, (9, 20), (13, 300));
            for (i, y) in (9..20).enumerate() {
                for (j, x) in (13..300).enumerate() {
                    assert_eq!(part[i * 287 + j], all[(y - HALO) * nc + x - HALO], "{origin:?} ({x},{y})");
                }
            }
        }
    }
}
