//! Fujifilm compressed RAF codec (lossless and lossy, X-Trans and Bayer).
//!
//! ## Stream layout
//! The payload starts with a 16-byte big-endian header (signature `IS`), followed by
//! one big-endian `u32` byte size per vertical *block* (padded to 16 bytes), then —
//! for lossy files only — one quantisation base per block and 6-row *line*
//! (`blocks × align16(lines)` bytes), and finally the entropy-coded blocks back to back.
//!
//! The image is split into vertical stripes of `block_size` (768) columns. Every
//! stripe is an independent bit stream, which is what makes this codec embarrassingly
//! parallel: [`FujiCompressed::decode`] decodes all stripes concurrently straight
//! into the output buffer.
//!
//! ## Coding
//! A stripe is coded line by line, a line being six sensor rows. Samples of each
//! colour are gathered into per-colour line buffers (3 red, 6 green, 3 blue per line,
//! plus the two previous lines of each colour as context). Each sample is predicted
//! from its neighbours (a median-edge-like rule that differs for even and odd
//! positions), and the residual is coded with an adaptive Golomb-Rice-like code: a
//! unary prefix (zero run) followed by `k` raw bits, `k` derived from running
//! statistics selected by a quantised local gradient. Lossy files additionally
//! quantise residuals with a per-line step `2 * q_base + 1`. X-Trans streams leave
//! some positions to pure interpolation (no bits are spent on them).
//!
//! Decoding is bit-exact, including tolerance of a one-byte over-read at the end of
//! a stripe.

use flashraw_core::bits::{BitPumpMsb, BitRead};
use flashraw_core::par::{self, GridWriter};
use flashraw_core::{Error, Rect, Result};

/// Size of the stream header in bytes.
pub const HEADER_SIZE: usize = 16;

/// Parsed stream header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FujiHeader {
    pub lossless: bool,
    /// X-Trans (raw type 16) or Bayer (raw type 0).
    pub xtrans: bool,
    pub bits: u32,
    pub height: u32,
    pub rounded_width: u32,
    pub width: u32,
    pub block_size: u32,
    pub blocks: u32,
    /// Number of 6-row lines.
    pub lines: u32,
}

impl FujiHeader {
    /// Parse and validate the 16-byte header.
    pub fn parse(d: &[u8]) -> Result<Self> {
        let h = d.get(..HEADER_SIZE).ok_or_else(|| Error::corrupt("truncated Fuji compressed header"))?;
        let be16 = |i: usize| u16::from_be_bytes([h[i], h[i + 1]]) as u32;
        let signature = be16(0);
        let lossless = h[2];
        let raw_type = h[3];
        let bits = h[4] as u32;
        let height = be16(5);
        let rounded_width = be16(7);
        let width = be16(9);
        let block_size = be16(11);
        let blocks = h[13] as u32;
        let lines = be16(14);
        let ok = signature == 0x4953
            && lossless <= 1
            && (6..=0x4002).contains(&height)
            && height % 6 == 0
            && block_size == 0x300
            && (0x300..=0x4200).contains(&width)
            && width % 24 == 0
            && rounded_width <= 0x4200
            && rounded_width >= block_size
            && rounded_width % block_size == 0
            && rounded_width - width < block_size
            && blocks > 0
            && blocks <= 0x10
            && blocks == rounded_width / block_size
            && lines > 0
            && lines <= 0xAAB
            && lines == height / 6
            && matches!(bits, 12 | 14 | 16)
            && matches!(raw_type, 0 | 16);
        if !ok {
            return Err(Error::corrupt("invalid Fuji compressed header"));
        }
        Ok(FujiHeader {
            lossless: lossless == 1,
            xtrans: raw_type == 16,
            bits,
            height,
            rounded_width,
            width,
            block_size,
            blocks,
            lines,
        })
    }

    /// Samples per colour line buffer.
    fn line_width(&self) -> usize {
        if self.xtrans { self.block_size as usize * 2 / 3 } else { self.block_size as usize / 2 }
    }

    /// Width in pixels of block `i` (the last block may be narrower).
    pub fn block_width(&self, i: u32) -> u32 {
        if i + 1 == self.blocks { self.width - self.block_size * i } else { self.block_size }
    }
}

/// Colour (0 = red, 1 = green, 2 = blue) of every position of a 6×6 tile anchored at
/// the buffer origin. Bayer patterns are repeated to fill the tile.
pub type ColorTile = [[u8; 6]; 6];

/// A parsed compressed payload, ready to decode.
pub struct FujiCompressed<'a> {
    pub header: FujiHeader,
    /// Entropy-coded data of each block.
    blocks: Vec<&'a [u8]>,
    /// Lossy only: per block, the quantisation base of each line.
    q_bases: Vec<&'a [u8]>,
}

impl<'a> FujiCompressed<'a> {
    /// `data` starts at the stream header.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let header = FujiHeader::parse(data)?;
        let body = &data[HEADER_SIZE..];
        let nb = header.blocks as usize;
        let sizes_len = 4 * nb;
        let sizes = body.get(..sizes_len).ok_or_else(|| Error::corrupt("truncated Fuji block size table"))?;
        let mut pos = (sizes_len + 15) & !15;
        let mut q_bases = Vec::new();
        if !header.lossless {
            let step = (header.lines as usize + 15) & !15;
            for b in 0..nb {
                let start = pos + b * step;
                let q = body.get(start..start + header.lines as usize);
                // A short table means missing data.
                q_bases.push(q.ok_or_else(|| Error::corrupt("truncated Fuji lossy quantisation table"))?);
            }
            pos += nb * step;
        }
        let mut blocks = Vec::with_capacity(nb);
        for b in 0..nb {
            let size =
                u32::from_be_bytes([sizes[4 * b], sizes[4 * b + 1], sizes[4 * b + 2], sizes[4 * b + 3]]) as usize;
            // A block past the end of the file is empty: its decode fails (and is
            // reported as truncation by the container), while opening still succeeds.
            let start = pos.min(body.len());
            let end = pos.saturating_add(size).min(body.len());
            blocks.push(&body[start..end]);
            pos = pos.saturating_add(size);
        }
        Ok(FujiCompressed { header, blocks, q_bases })
    }

    /// Byte ranges (relative to the header start) of each block's entropy data.
    pub fn block_ranges(&self, base: &'a [u8]) -> Vec<(usize, usize)> {
        self.blocks.iter().map(|b| (b.as_ptr() as usize - base.as_ptr() as usize, b.len())).collect()
    }

    /// Decode into `out`, a row-major `width × height` buffer that must match the
    /// stream's image size. Stripes decode in parallel when `parallel`.
    pub fn decode(&self, out: &mut [u16], width: u32, height: u32, colors: &ColorTile, parallel: bool) -> Result<()> {
        let h = &self.header;
        if width != h.width || height != h.height {
            return Err(Error::InvalidArgument("Fuji compressed output size does not match the stream".into()));
        }
        self.decode_region(out, Rect::new(0, 0, width, height), colors, parallel)
    }

    /// Decode only `region` (image coordinates) into `out` (`region.width × region.height`).
    /// Only the stripes intersecting the region are decoded, and each stops after the
    /// last 6-row line the region needs.
    pub fn decode_region(&self, out: &mut [u16], region: Rect, colors: &ColorTile, parallel: bool) -> Result<()> {
        let h = &self.header;
        let (w, ht) = (region.width as usize, region.height as usize);
        if out.len() < w * ht || region.right() > h.width || region.bottom() > h.height {
            return Err(Error::InvalidArgument("Fuji compressed region outside the image or buffer too small".into()));
        }
        if region.is_empty() {
            return Ok(());
        }
        let common = Common::new(h, &self.q_bases)?;
        let map = OutputMap::new(h, colors);
        let grid = GridWriter::new(
            out,
            region.width,
            region.height,
            1,
            (region.x, region.y),
            h.block_size,
            h.height,
            h.width,
            h.height,
        )
        .ok_or_else(|| Error::InvalidArgument("invalid Fuji output grid".into()))?;
        let needed_lines = (region.bottom() as usize).div_ceil(6).min(h.lines as usize);
        let errors = std::sync::Mutex::new(None);
        par::for_each_index(self.blocks.len(), parallel, |b| {
            // Stripes outside the region are not claimed (claim returns None).
            let Some(mut tile) = grid.claim(b as u32, 0) else { return };
            let (_, th) = tile.size();
            let (sx, sy) = (tile.skip_x as usize, tile.skip_y as usize);
            let q = self.q_bases.get(b).copied().unwrap_or(&[]);
            let emit = |line: usize, lines: &[u16]| {
                for r in 0..6 {
                    let y = line * 6 + r;
                    if y < sy || y - sy >= th {
                        continue;
                    }
                    map.emit_row(r, sx, tile.row(y - sy), lines);
                }
            };
            let res = if h.lossless {
                decode_block::<false>(&common, self.blocks[b], q, needed_lines, emit)
            } else {
                decode_block::<true>(&common, self.blocks[b], q, needed_lines, emit)
            };
            if let Err(e) = res {
                errors.lock().unwrap().get_or_insert(e);
            }
        });
        match errors.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

// Line buffer indices.
const R0: usize = 0;
const R2: usize = 2;
const R3: usize = 3;
const R4: usize = 4;
const G0: usize = 5;
const G2: usize = 7;
const G3: usize = 8;
const G4: usize = 9;
const G5: usize = 10;
const G6: usize = 11;
const G7: usize = 12;
const B0: usize = 13;
const B2: usize = 15;
const B3: usize = 16;
const B4: usize = 17;
const LINES: usize = 18;

/// Maps each of the 6 output rows × 6 horizontal CFA phases to its base index in the
/// per-colour line buffers. Advancing by 6 output columns advances each line index by
/// `step` (`4` for X-Trans, `3` for Bayer).
struct OutputMap {
    base: [[usize; 6]; 6],
    step: usize,
    bayer_2x2: bool,
}

impl OutputMap {
    fn new(h: &FujiHeader, colors: &ColorTile) -> Self {
        let stride = h.line_width() + 2;
        let step = if h.xtrans { 4 } else { 3 };
        let base = std::array::from_fn(|r| {
            std::array::from_fn(|m| {
                let c = colors[r][m];
                let line = match c {
                    0 => R2 + r / 2,
                    2 => B2 + r / 2,
                    _ => G2 + r,
                };
                let i = if h.xtrans { [0, 1, 1, 2, 3, 3][m] } else { m >> 1 };
                line * stride + 1 + i
            })
        });
        let bayer_2x2 =
            !h.xtrans && colors.iter().all(|r| r[0] == r[2] && r[0] == r[4] && r[1] == r[3] && r[1] == r[5]);
        OutputMap { base, step, bayer_2x2 }
    }

    #[inline]
    fn emit_row(&self, r: usize, sx: usize, row: &mut [u16], lines: &[u16]) {
        if self.bayer_2x2 && (sx & 1) == 0 && (row.len() & 1) == 0 {
            let pairs = row.len() / 2;
            let half_sx = sx / 2;
            let s0 = &lines[self.base[r][0] + half_sx..][..pairs];
            let s1 = &lines[self.base[r][1] + half_sx..][..pairs];
            for (dst, (&v0, &v1)) in row.chunks_exact_mut(2).zip(s0.iter().zip(s1.iter())) {
                dst[0] = v0;
                dst[1] = v1;
            }
            return;
        }
        if self.step == 4 && sx % 6 == 0 {
            let chunks = row.len() / 6;
            let k0 = sx / 6;
            if chunks > 0 {
                let [b0, b1, b2, b3, b4, b5] = self.base[r];
                let start = k0 * 4;
                let need = (chunks - 1) * 4 + 1;
                let (s0, s1, s2, s3, s4, s5) = (
                    &lines[b0 + start..][..need],
                    &lines[b1 + start..][..need],
                    &lines[b2 + start..][..need],
                    &lines[b3 + start..][..need],
                    &lines[b4 + start..][..need],
                    &lines[b5 + start..][..need],
                );
                for (dk, dst) in row.chunks_exact_mut(6).enumerate() {
                    let off = dk * 4;
                    dst[0] = s0[off];
                    dst[1] = s1[off];
                    dst[2] = s2[off];
                    dst[3] = s3[off];
                    dst[4] = s4[off];
                    dst[5] = s5[off];
                }
            }
            let done = chunks * 6;
            for (i, dst) in row[done..].iter_mut().enumerate() {
                let p = sx + done + i;
                *dst = lines[self.base[r][p % 6] + (p / 6) * 4];
            }
            return;
        }
        let base = &self.base[r];
        let step = self.step;
        for (i, dst) in row.iter_mut().enumerate() {
            let p = sx + i;
            *dst = lines[base[p % 6] + (p / 6) * step];
        }
    }
}

/// Size of a level table: a power of two above every difference of 16-bit samples
/// (`-65535..=65535`), so a difference wraps to a unique index.
const LUT_LEN: usize = 1 << 17;
type Lut = Box<[i8; LUT_LEN]>;

/// A quantisation table: maps a neighbour difference to a level in `-4..=4`.
#[derive(Clone)]
struct QTable {
    lut: Lut,
    q_base: i32,
    max_grad: i32,
    grad_mult: i32,
    total_values: i32,
    raw_bits: u32,
    /// Longest unary prefix of a Rice code; longer prefixes escape to a raw value.
    max_zeros: i32,
}

fn log2ceil(v: i32) -> u32 {
    if v <= 1 { 0 } else { 32 - ((v - 1) as u32).leading_zeros() }
}

impl QTable {
    /// Level lookup for differences in `-max..=max` with thresholds `qp[0..4]`, indexed
    /// by the difference modulo the table size. Filled by contiguous slice ranges (`memset`)
    /// instead of a 131K-iteration per-element branch loop.
    fn lut(qp: [i32; 5]) -> Lut {
        let mut t: Lut = vec![0i8; LUT_LEN].into_boxed_slice().try_into().unwrap();
        let [q0, q1, q2, q3, max] = qp.map(|x| x as usize);
        if q0 + 1 < q1 {
            t[q0 + 1..q1].fill(1);
            t[LUT_LEN - (q1 - 1)..LUT_LEN - q0].fill(-1);
        }
        if q1 < q2 {
            t[q1..q2].fill(2);
            t[LUT_LEN - q2..LUT_LEN - (q1 - 1)].fill(-2);
        }
        if q2 < q3 {
            t[q2..q3].fill(3);
            t[LUT_LEN - q3..LUT_LEN - (q2 - 1)].fill(-3);
        }
        if q3 <= max {
            t[q3..=max].fill(4);
            t[LUT_LEN - max..=LUT_LEN - q3].fill(-4);
        }
        t
    }

    fn new(lut: Lut, q_base: i32, max_grad: i32, grad_mult: i32, total_values: i32, max_value: i32) -> Self {
        let raw_bits = log2ceil(total_values);
        let max_bits = 4 * log2ceil(max_value + 1) as i32;
        QTable { lut, q_base, max_grad, grad_mult, total_values, raw_bits, max_zeros: max_bits - raw_bits as i32 - 1 }
    }

    /// The main table for a line quantisation base.
    fn main(q_base: i32, max_value: i32) -> QTable {
        let max_val = max_value + 1;
        let mut qp = [q_base, 3 * q_base + 0x12, 5 * q_base + 0x43, 7 * q_base + 0x114, max_value];
        if qp[1] >= max_val || qp[1] < q_base + 1 {
            qp[1] = q_base + 1;
        }
        if qp[2] < qp[1] || qp[2] >= max_val {
            qp[2] = qp[1];
        }
        if qp[3] < qp[2] || qp[3] >= max_val {
            qp[3] = qp[2];
        }
        let total_values = (qp[4] + 2 * q_base) / (2 * q_base + 1) + 1;
        Self::new(Self::lut(qp), q_base, 0, 9, total_values, max_value)
    }

    #[inline(always)]
    fn level(&self, v: i32) -> i32 {
        // The mask keeps the index in range without a bounds check.
        self.lut[v as usize & (LUT_LEN - 1)] as i32
    }

    /// The three fixed tables used by lossy streams for flat areas.
    fn lossy_extra(max_value: i32) -> [QTable; 3] {
        let m = max_value;
        let pick = |limit: i32, fallback: i32| if m >= limit { limit } else { fallback };
        let mk = |q_base: i32, max_grad: i32, total_values: i32, t: [i32; 3]| {
            let q1 = pick(t[0], q_base + 1);
            let q2 = pick(t[1], q1);
            let q3 = pick(t[2], q2);
            Self::new(Self::lut([q_base, q1, q2, q3, m]), q_base, max_grad, 3, total_values, m)
        };
        [
            mk(0, 5, m + 1, [0x12, 0x43, 0x114]),
            mk(1, 6, (m + 2) / 3 + 1, [0x15, 0x48, 0x11B]),
            mk(2, 7, (m + 4) / 5 + 1, [0x18, 0x4D, 0x122]),
        ]
    }
}

/// Parameters shared by all blocks.
struct Common {
    xtrans: bool,
    line_width: usize,
    max_value: i32,
    /// Main tables indexed by q_base (lossless: only index 0).
    main: Vec<Option<QTable>>,
    extra: Option<[QTable; 3]>,
}

impl Common {
    fn new(h: &FujiHeader, q_bases: &[&[u8]]) -> Result<Self> {
        let max_value = (1i32 << h.bits) - 1;
        let mut main: Vec<Option<QTable>> = vec![None; 256];
        let extra = if h.lossless {
            main[0] = Some(QTable::main(0, max_value));
            None
        } else {
            // Precompute the (few) distinct line quantisers once for all blocks.
            let mut seen = [false; 256];
            for q in q_bases.iter().flat_map(|q| q.iter()) {
                seen[*q as usize] = true;
            }
            for (q, _) in seen.iter().enumerate().filter(|(_, s)| **s) {
                main[q] = Some(QTable::main(q as i32, max_value));
            }
            Some(QTable::lossy_extra(max_value))
        };
        Ok(Common { xtrans: h.xtrans, line_width: h.line_width(), max_value, main, extra })
    }
}

/// Adaptive statistics for one gradient context: (accumulated magnitude, count).
/// The Rice parameter `bit_diff(sum, n)` is kept alongside, computed when the
/// statistics change rather than on every use.
/// Packed in 8 bytes so one load fetches all three.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct Stat {
    sum: i32,
    n: u8,
    k: u8,
}

#[inline(always)]
fn new_stat(sum: i32, n: i32) -> Stat {
    // `n` stays within 1..=MIN_VALUE and `k` within 0..=15.
    let k = bit_diff(sum, n);
    Stat { sum, n: n as u8, k: k as u8 }
}

/// Per-context statistics: 41 main gradient contexts and, for lossy streams, 5
/// contexts for each of the three flat-area tables.
#[derive(Clone, Copy)]
struct Grads {
    main: [Stat; 41],
    lossy: [[Stat; 5]; 3],
}

const MIN_VALUE: i32 = 0x40;

/// Smallest `k` (capped at 15) such that `n << k >= sum`: the Rice parameter.
#[inline(always)]
fn bit_diff(sum: i32, n: i32) -> u32 {
    // Branch-free: start from the bit-length difference and correct by one.
    let (s, n) = (sum.max(1) as u32, n.max(1) as u32);
    let k = n.leading_zeros().saturating_sub(s.leading_zeros());
    let k = k + (((n as u64) << k) < s as u64) as u32;
    k.min(15)
}

/// Decode all lines of one stripe, handing each finished line's buffers to `emit`.
fn decode_block<const LOSSY: bool>(
    c: &Common,
    data: &[u8],
    q_bases: &[u8],
    lines: usize,
    mut emit: impl FnMut(usize, &[u16]),
) -> Result<()> {
    let mut d = BlockDecoder::<LOSSY>::new(c, data, q_bases)?;
    for line in 0..lines {
        d.decode_line(line)?;
        emit(line, &d.lines);
        d.advance_line();
    }
    Ok(())
}

struct BlockDecoder<'c, 'a, const LOSSY: bool> {
    c: &'c Common,
    bits: BitPumpMsb<'a>,
    data_len: usize,
    /// 18 colour lines of `line_width + 2` samples (one guard sample on each side).
    lines: Vec<u16>,
    stride: usize,
    /// Statistics for even (0) and odd (1) positions, three sets each (chosen per pass).
    grads: [[Grads; 3]; 2],
    q_bases: &'a [u8],
    /// Current main table.
    qt: &'c QTable,
    cur_q: i32,
    /// Set when the bit stream ran dry inside a zero run (checked once per line).
    overrun: bool,
}

impl<'c, 'a, const LOSSY: bool> BlockDecoder<'c, 'a, LOSSY> {
    fn new(c: &'c Common, data: &'a [u8], q_bases: &'a [u8]) -> Result<Self> {
        let stride = c.line_width + 2;
        let init = Grads { main: [new_stat(0, 1); 41], lossy: [[new_stat(0, 1); 5]; 3] };
        let first_q = if LOSSY { q_bases.first().copied().unwrap_or(0) as usize } else { 0 };
        let qt = c.main[first_q].as_ref().ok_or_else(|| Error::corrupt("missing Fuji quantisation table"))?;
        let mut d = BlockDecoder {
            c,
            bits: BitPumpMsb::new(data),
            data_len: data.len(),
            lines: vec![0; LINES * stride],
            stride,
            grads: [[init; 3]; 2],
            q_bases,
            qt,
            cur_q: -1,
            overrun: false,
        };
        if LOSSY {
            let extra = c.extra.as_ref().unwrap();
            for k in 0..3 {
                let max_diff = 2.max((extra[k].total_values + 0x20) >> 6);
                for g in d.grads.iter_mut().flatten() {
                    g.lossy[k] = [new_stat(max_diff, 1); 5];
                }
            }
        } else {
            d.cur_q = 0;
            d.init_main_grads();
        }
        Ok(d)
    }

    fn init_main_grads(&mut self) {
        let max_diff = 2.max((self.qt.total_values + 0x20) >> 6);
        for g in self.grads.iter_mut().flatten() {
            g.main = [new_stat(max_diff, 1); 41];
        }
    }

    fn decode_line(&mut self, line: usize) -> Result<()> {
        if LOSSY {
            let q = self.q_bases.get(line).copied().unwrap_or(0) as i32;
            if line == 0 || q != self.cur_q {
                self.qt = self.c.main[q as usize]
                    .as_ref()
                    .ok_or_else(|| Error::corrupt("missing Fuji quantisation table"))?;
                self.cur_q = q;
                self.init_main_grads();
            }
        }
        if self.c.xtrans {
            const S: usize = 514;
            self.pass::<S, false, false, true, true>(R2, G2, 0);
            self.extend_one(R3);
            self.extend_one(G3);
            self.pass::<S, true, true, false, false>(G3, B2, 1);
            self.extend_one(G4);
            self.extend_one(B3);
            self.pass::<S, false, true, false, false>(R3, G4, 2);
            self.extend_one(R4);
            self.extend_one(G5);
            self.pass::<S, true, true, true, false>(G5, B3, 0);
            self.extend_one(G6);
            self.extend_one(B4);
            self.pass::<S, true, false, true, true>(R4, G6, 1);
            self.extend_one(G7);
            self.pass::<S, false, false, false, true>(G7, B4, 2);
        } else {
            const S: usize = 386;
            self.pass::<S, true, true, true, true>(R2, G2, 0);
            self.extend_one(R3);
            self.extend_one(G3);
            self.pass::<S, true, true, true, true>(G3, B2, 1);
            self.extend_one(G4);
            self.extend_one(B3);
            self.pass::<S, true, true, true, true>(R3, G4, 2);
            self.extend_one(R4);
            self.extend_one(G5);
            self.pass::<S, true, true, true, true>(G5, B3, 0);
            self.extend_one(G6);
            self.extend_one(B4);
            self.pass::<S, true, true, true, true>(R4, G6, 1);
            self.extend_one(G7);
            self.pass::<S, true, true, true, true>(G7, B4, 2);
        }
        // A stripe is padded with a single zero byte and any further read fails:
        // consuming more than that means the stripe is truncated or corrupt.
        if self.overrun || self.bits_consumed() > 8 * self.data_len + 7 {
            return Err(Error::corrupt("truncated Fuji compressed block"));
        }
        Ok(())
    }

    fn bits_consumed(&self) -> usize {
        // byte_position() excludes whole buffered bytes; the partially consumed byte's
        // remaining bits are still buffered.
        (self.bits.byte_position() * 8).saturating_sub((self.bits.buffered_bits() % 8) as usize)
    }

    /// One coding pass over two colour lines `a` and `b`: even positions of both lines
    /// (decoded or interpolated per `A0`/`A2`/`B0`/`B2` at `ep & 3 == 0` and `2`), with
    /// odd positions lagging five even steps behind (`op = ep - 7`).
    #[inline(never)]
    fn pass<const S: usize, const A0: bool, const A2: bool, const B0: bool, const B2: bool>(
        &mut self,
        a: usize,
        b: usize,
        g: usize,
    ) {
        let lw = S - 2;
        let win_len = 2 * S + 11;
        let edge_len = 2 * S + 9;
        // Everything the samples touch is held in locals for the pass: the bit pump stays
        // in registers, and the line windows and statistics are borrowed once.
        let mut lines = std::mem::take(&mut self.lines);
        let mut st = Sampler {
            lut: &self.qt.lut,
            max_value: self.c.max_value,
            max_zeros: self.qt.max_zeros,
            raw_bits: self.qt.raw_bits,
            qt: self.qt,
            extra: self.c.extra.as_ref(),
            bits: self.bits,
            overrun: false,
        };
        let [even, odd] = &mut self.grads;
        let (ge, go) = (&mut even[g], &mut odd[g]);
        // `a` lies below `b` in the buffer and their three-line windows do not overlap.
        let (lo, hi) = lines.split_at_mut((b - 2) * S);
        let la = &mut lo[(a - 2) * S..(a + 1) * S];
        let lb = &mut hi[..3 * S];

        // Prologue: first four even steps (`ep = 0, 2, 4, 6`) before odd positions start.
        {
            let wa = &mut la[..edge_len];
            let wb = &mut lb[..edge_len];
            step_even_win::<S, LOSSY, A0, 0>(wa, &mut st, ge);
            step_even_win::<S, LOSSY, B0, 0>(wb, &mut st, ge);
            step_even_win::<S, LOSSY, A2, 2>(wa, &mut st, ge);
            step_even_win::<S, LOSSY, B2, 2>(wb, &mut st, ge);
            step_even_win::<S, LOSSY, A0, 4>(wa, &mut st, ge);
            step_even_win::<S, LOSSY, B0, 4>(wb, &mut st, ge);
            step_even_win::<S, LOSSY, A2, 6>(wa, &mut st, ge);
            step_even_win::<S, LOSSY, B2, 6>(wb, &mut st, ge);
        }

        // Steady-state main loop: two even/odd pairs per iteration (`ep` and `ep + 2`).
        for ep in (8..lw).step_by(4) {
            let wa = &mut la[ep - 7..ep - 7 + win_len];
            let wb = &mut lb[ep - 7..ep - 7 + win_len];
            step_even_win::<S, LOSSY, A0, 7>(wa, &mut st, ge);
            step_even_win::<S, LOSSY, B0, 7>(wb, &mut st, ge);
            decode_odd_win::<S, LOSSY, 0>(wa, &mut st, go);
            decode_odd_win::<S, LOSSY, 0>(wb, &mut st, go);

            step_even_win::<S, LOSSY, A2, 9>(wa, &mut st, ge);
            step_even_win::<S, LOSSY, B2, 9>(wb, &mut st, ge);
            decode_odd_win::<S, LOSSY, 2>(wa, &mut st, go);
            decode_odd_win::<S, LOSSY, 2>(wb, &mut st, go);
        }

        // Epilogue: last four odd steps (`op = lw - 7, lw - 5, lw - 3, lw - 1`).
        {
            let wa = &mut la[lw - 7..lw - 7 + edge_len];
            let wb = &mut lb[lw - 7..lw - 7 + edge_len];
            decode_odd_win::<S, LOSSY, 0>(wa, &mut st, go);
            decode_odd_win::<S, LOSSY, 0>(wb, &mut st, go);
            decode_odd_win::<S, LOSSY, 2>(wa, &mut st, go);
            decode_odd_win::<S, LOSSY, 2>(wb, &mut st, go);
            decode_odd_win::<S, LOSSY, 4>(wa, &mut st, go);
            decode_odd_win::<S, LOSSY, 4>(wb, &mut st, go);
            decode_odd_win::<S, LOSSY, 6>(wa, &mut st, go);
            decode_odd_win::<S, LOSSY, 6>(wb, &mut st, go);
        }

        self.bits = st.bits;
        self.overrun |= st.overrun;
        self.lines = lines;
    }

    /// Copy edge samples from the previous line `l - 1` into line `l`'s guard positions.
    #[inline(always)]
    fn extend_one(&mut self, l: usize) {
        let (s, lw) = (self.stride, self.c.line_width);
        self.lines[l * s] = self.lines[(l - 1) * s + 1];
        self.lines[l * s + lw + 1] = self.lines[(l - 1) * s + lw];
    }

    /// Rotate line context after a decoded line: the last two lines of each colour
    /// become the first two, and the first new line of each colour gets its guard samples.
    fn advance_line(&mut self) {
        let s = self.stride;
        let lw = self.c.line_width;
        for (dst, src) in [(R0, R3), (R0 + 1, R4), (G0, G6), (G0 + 1, G7), (B0, B3), (B0 + 1, B4)] {
            self.lines.copy_within(src * s..(src + 1) * s, dst * s);
        }
        for first in [R2, G2, B2] {
            self.lines[first * s] = self.lines[(first - 1) * s + 1];
            self.lines[first * s + lw + 1] = self.lines[(first - 1) * s + lw];
        }
    }
}

/// Cold fallback when the Rice unary prefix exceeds the fast window or escapes.
#[cold]
#[inline(never)]
fn decode_code_slow<'a>(
    mut bits: BitPumpMsb<'a>,
    mut overrun: bool,
    k: u32,
    max_zeros: i32,
    raw_bits: u32,
) -> (BitPumpMsb<'a>, bool, i32) {
    let mut sample = 0i32;
    loop {
        bits.fill();
        let w = bits.peek_no_fill(32);
        if w != 0 {
            let lz = w.leading_zeros();
            bits.skip_no_fill(lz + 1);
            sample += lz as i32;
            break;
        }
        bits.skip_no_fill(32);
        sample += 32;
        if bits.exhausted() {
            overrun = true;
            break;
        }
    }
    let code =
        if sample < max_zeros { ((sample as u32) << k | bits.get(k)) as i32 } else { bits.get(raw_bits) as i32 + 1 };
    (bits, overrun, code)
}

/// Entropy-decoding state of one pass, held in locals.
#[derive(Clone, Copy)]
struct Sampler<'c, 'a> {
    lut: &'c [i8; LUT_LEN],
    max_value: i32,
    max_zeros: i32,
    raw_bits: u32,
    /// Current main table (used by lossy mode).
    qt: &'c QTable,
    extra: Option<&'c [QTable; 3]>,
    bits: BitPumpMsb<'a>,
    /// Set when the bit stream ran dry inside a zero run.
    overrun: bool,
}

impl Sampler<'_, '_> {
    /// Decode one residual and reconstruct the sample from the prediction `interp`.
    /// `grads` are the statistics of the sample's side (even or odd) and pass.
    #[inline(always)]
    fn decode_residual<const LOSSY: bool>(&mut self, grads: &mut Grads, v1: i32, v2: i32, interp: i32) -> u16 {
        let max = self.max_value;
        if !LOSSY {
            let l1 = self.lut[v1 as usize & (LUT_LEN - 1)] as i32;
            let l2 = self.lut[v2 as usize & (LUT_LEN - 1)] as i32;
            let grad = 9 * l1 + l2;
            let gi = (grad.unsigned_abs() as usize).min(40);
            let stat = &mut grads.main[gi];
            let (sum, n, k) = (stat.sum, stat.n as i32, stat.k as u32);
            self.bits.fill();
            let w = self.bits.peek64_no_fill();
            let lz = w.leading_zeros();
            let total = lz + 1 + k;
            let code = if (lz as i32) < self.max_zeros && total <= self.bits.buffered_bits() {
                self.bits.skip64_no_fill(total);
                ((w >> (64 - total)) as u32).wrapping_add(lz.wrapping_sub(1) << k) as i32
            } else {
                let (bits, overrun, c) = decode_code_slow(self.bits, self.overrun, k, self.max_zeros, self.raw_bits);
                self.bits = bits;
                self.overrun = overrun;
                c
            };
            let code = (code >> 1) ^ -(code & 1);
            let h = (n == MIN_VALUE) as u32;
            let (sum, n) = ((sum + code.abs()) >> h, n >> h);
            *stat = new_stat(sum, n + 1);
            let delta = if grad < 0 { -code } else { code };
            let mut v = interp + delta;
            let span = max + 1;
            let adj = if v < 0 {
                span
            } else if v > max {
                -span
            } else {
                0
            };
            v += adj;
            return v.max(0).min(max) as u16;
        }

        let flat = v1.abs() + v2.abs();
        let mut qt = self.qt;
        let mut ctx = usize::MAX;
        let extra = self.extra.unwrap();
        let mut i = 1;
        while qt.q_base >= i && i < 4 {
            let t = &extra[(i - 1) as usize];
            if flat <= t.max_grad {
                qt = t;
                ctx = (i - 1) as usize;
                break;
            }
            i += 1;
        }
        let grad = qt.grad_mult * qt.level(v1) + qt.level(v2);
        let gi = grad.unsigned_abs() as usize;
        let stat = if ctx != usize::MAX { &mut grads.lossy[ctx][gi.min(4)] } else { &mut grads.main[gi.min(40)] };
        let (sum, n, k) = (stat.sum, stat.n as i32, stat.k as u32);
        self.bits.fill();
        let w = self.bits.peek64_no_fill();
        let lz = w.leading_zeros();
        let total = lz + 1 + k;
        let code = if (lz as i32) < qt.max_zeros && total <= self.bits.buffered_bits() {
            self.bits.skip64_no_fill(total);
            ((w >> (64 - total)) as u32).wrapping_add(lz.wrapping_sub(1) << k) as i32
        } else {
            let (bits, overrun, c) = decode_code_slow(self.bits, self.overrun, k, qt.max_zeros, qt.raw_bits);
            self.bits = bits;
            self.overrun = overrun;
            c
        };
        let code = (code >> 1) ^ -(code & 1);
        let h = (n == MIN_VALUE) as u32;
        let (sum, n) = ((sum + code.abs()) >> h, n >> h);
        *stat = new_stat(sum, n + 1);
        let (q_base, step) = (qt.q_base, 2 * qt.q_base + 1);
        let delta = (if grad < 0 { -code } else { code }) * step;
        let mut v = interp + delta;
        let span = qt.total_values * step;
        let adj = if v < -q_base {
            span
        } else if v > q_base + max {
            -span
        } else {
            0
        };
        v += adj;
        v.max(0).min(max) as u16
    }
}

#[inline(always)]
fn step_even_win<const S: usize, const LOSSY: bool, const DECODE: bool, const OFF: usize>(
    w: &mut [u16],
    st: &mut Sampler,
    grads: &mut Grads,
) {
    let rf = w[OFF + 1] as i32;
    let rc = w[S + OFF] as i32;
    let rb = w[S + OFF + 1] as i32;
    let rd = w[S + OFF + 2] as i32;
    let interp = even_prediction(rf, rc, rb, rd) >> 2;
    w[2 * S + OFF + 1] =
        if DECODE { st.decode_residual::<LOSSY>(grads, rb - rf, rc - rb, interp) } else { interp as u16 };
}

#[inline(always)]
fn decode_odd_win<const S: usize, const LOSSY: bool, const OFF: usize>(
    w: &mut [u16],
    st: &mut Sampler,
    grads: &mut Grads,
) {
    let rc = w[S + OFF] as i32;
    let rb = w[S + OFF + 1] as i32;
    let rd = w[S + OFF + 2] as i32;
    let ra = w[2 * S + OFF] as i32;
    let rg = w[2 * S + OFF + 2] as i32;
    let (x, y) = (rb - rc, rb - rd);
    let interp = if (x ^ y) >= 0 && x != 0 && y != 0 { (rg + ra + 2 * rb) >> 2 } else { (ra + rg) >> 1 };
    w[2 * S + OFF + 1] = st.decode_residual::<LOSSY>(grads, rb - rc, rc - ra, interp);
}

/// Edge-directed prediction for even positions (four times the predicted value).
#[inline(always)]
fn even_prediction(rf: i32, rc: i32, rb: i32, rd: i32) -> i32 {
    let (d_cb, d_fb, d_db) = ((rc - rb).abs(), (rf - rb).abs(), (rd - rb).abs());
    let dropped = if d_cb > d_fb && d_cb > d_db {
        rc
    } else if d_db > d_cb && d_db > d_fb {
        rd
    } else {
        rf
    };
    rf + rc + rd - dropped + 2 * rb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rice_parameter_matches_reference_loop() {
        for v1 in [0, 1, 2, 3, 5, 64, 100, 1000, 65535, 1 << 20, i32::MAX / 4] {
            for v2 in 1..=130 {
                // Loop form: while (d <= 14 && (v2 << ++d) < v1);
                let mut d = 0u32;
                if v2 < v1 {
                    loop {
                        if d > 14 {
                            break;
                        }
                        d += 1;
                        if ((v2 as i64) << d) >= v1 as i64 {
                            break;
                        }
                    }
                }
                assert_eq!(bit_diff(v1, v2), d, "v1 {v1} v2 {v2}");
            }
        }
    }

    fn header(lossless: u8, raw_type: u8, bits: u8, h: u16, rw: u16, w: u16, blocks: u8) -> Vec<u8> {
        let mut v = vec![0x49, 0x53, lossless, raw_type, bits];
        v.extend(h.to_be_bytes());
        v.extend(rw.to_be_bytes());
        v.extend(w.to_be_bytes());
        v.extend(0x300u16.to_be_bytes());
        v.push(blocks);
        v.extend((h / 6).to_be_bytes());
        v
    }

    #[test]
    fn header_validation() {
        let h = header(1, 16, 14, 4182, 6912, 6384, 9);
        let p = FujiHeader::parse(&h).unwrap();
        assert!(p.lossless && p.xtrans);
        assert_eq!((p.width, p.height, p.blocks, p.lines), (6384, 4182, 9, 697));
        assert_eq!(p.block_width(8), 6384 - 8 * 768);
        assert!(FujiHeader::parse(&header(1, 16, 13, 4182, 6912, 6384, 9)).is_err());
        assert!(FujiHeader::parse(&header(1, 16, 14, 4181, 6912, 6384, 9)).is_err());
        assert!(FujiHeader::parse(&header(1, 16, 14, 4182, 6912, 6384, 8)).is_err());
        assert!(FujiHeader::parse(&header(2, 16, 14, 4182, 6912, 6384, 9)).is_err());
        assert!(FujiHeader::parse(&h[..10]).is_err());
    }

    #[test]
    fn qtables() {
        let t = QTable::main(0, 16383);
        assert_eq!(t.total_values, 16384);
        assert_eq!(t.raw_bits, 14);
        assert_eq!(t.level(0), 0);
        assert_eq!(t.level(1), 1);
        assert_eq!(t.level(-1), -1);
        assert_eq!(t.level(0x12), 2);
        assert_eq!(t.level(16383), 4);
        assert_eq!(t.level(-16383), -4);
        let t = QTable::main(3, 16383);
        assert_eq!(t.total_values, (16383 + 6) / 7 + 1);
        let x = QTable::lossy_extra(4095);
        assert_eq!(x[1].total_values, (4095 + 2) / 3 + 1);
        assert_eq!(log2ceil(16384), 14);
        assert_eq!(log2ceil(16385), 15);
        assert_eq!(log2ceil(1), 0);
    }

    fn stream(lossless: bool, xtrans: bool, payload: &[u8]) -> Vec<u8> {
        // Single 768-wide block, 12 rows.
        let mut v = header(lossless as u8, if xtrans { 16 } else { 0 }, 14, 12, 768, 768, 1);
        v.extend((payload.len() as u32).to_be_bytes());
        v.extend([0u8; 12]);
        if !lossless {
            v.extend([0u8, 2]);
            v.extend([0u8; 14]);
        }
        v.extend_from_slice(payload);
        v
    }

    fn pseudo(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8 | 0x11
            })
            .collect()
    }

    const XT: ColorTile = [
        [1, 1, 0, 1, 1, 2],
        [1, 1, 2, 1, 1, 0],
        [2, 0, 1, 0, 2, 1],
        [1, 1, 2, 1, 1, 0],
        [1, 1, 0, 1, 1, 2],
        [0, 2, 1, 2, 0, 1],
    ];

    const BAYER: ColorTile = [
        [0, 1, 0, 1, 0, 1],
        [1, 2, 1, 2, 1, 2],
        [0, 1, 0, 1, 0, 1],
        [1, 2, 1, 2, 1, 2],
        [0, 1, 0, 1, 0, 1],
        [1, 2, 1, 2, 1, 2],
    ];

    #[test]
    fn decodes_synthetic_streams_without_panicking() {
        for (lossless, xtrans) in [(true, true), (true, false), (false, true), (false, false)] {
            for seed in 1..6 {
                let data = stream(lossless, xtrans, &pseudo(200_000, seed));
                let c = FujiCompressed::new(&data).unwrap();
                let mut out = vec![0u16; 768 * 12];
                let tile = if xtrans { XT } else { BAYER };
                // Random data either decodes (values within range) or reports corruption.
                if c.decode(&mut out, 768, 12, &tile, false).is_ok() {
                    assert!(out.iter().all(|&v| v < 1 << 14));
                }
            }
        }
    }

    #[test]
    fn region_decode_matches_full_decode() {
        // Two stripes (1536 wide), 12 rows.
        let mut v = header(1, 16, 14, 12, 1536, 1536, 2);
        let payload = vec![0xFFu8; 300_000];
        v.extend((payload.len() as u32).to_be_bytes());
        v.extend((payload.len() as u32).to_be_bytes());
        v.extend([0u8; 8]);
        v.extend_from_slice(&payload);
        v.extend_from_slice(&payload);
        let c = FujiCompressed::new(&v).unwrap();
        let mut full = vec![0u16; 1536 * 12];
        c.decode(&mut full, 1536, 12, &XT, true).unwrap();
        for r in [Rect::new(700, 3, 100, 5), Rect::new(0, 0, 10, 1), Rect::new(800, 6, 736, 6)] {
            let mut part = vec![0u16; (r.width * r.height) as usize];
            c.decode_region(&mut part, r, &XT, true).unwrap();
            for y in 0..r.height as usize {
                let src = &full[(r.y as usize + y) * 1536 + r.x as usize..][..r.width as usize];
                assert_eq!(&part[y * r.width as usize..][..r.width as usize], src, "{r:?} row {y}");
            }
        }
        let mut small = vec![0u16; 4];
        assert!(c.decode_region(&mut small, Rect::new(1530, 0, 10, 1), &XT, true).is_err());
    }

    #[test]
    fn truncated_stream_is_an_error() {
        let data = stream(true, true, &[0x00; 64]);
        let c = FujiCompressed::new(&data).unwrap();
        let mut out = vec![0u16; 768 * 12];
        assert!(c.decode(&mut out, 768, 12, &XT, true).is_err());
        // Missing block table.
        assert!(FujiCompressed::new(&data[..18]).is_err());
    }

    #[test]
    fn all_ones_stream_is_deterministic() {
        // Every residual reads a zero-length prefix; the result must not depend on threading.
        let data = stream(true, true, &vec![0xFF; 400_000]);
        let c = FujiCompressed::new(&data).unwrap();
        let mut a = vec![0u16; 768 * 12];
        let mut b = vec![0u16; 768 * 12];
        c.decode(&mut a, 768, 12, &XT, false).unwrap();
        c.decode(&mut b, 768, 12, &XT, true).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn output_map_matches_reference() {
        for (xtrans, colors) in [(true, &XT), (false, &BAYER)] {
            let h = FujiHeader::parse(&header(1, if xtrans { 16 } else { 0 }, 14, 12, 768, 768, 1)).unwrap();
            let stride = h.line_width() + 2;
            let lines: Vec<u16> = (0..LINES * stride).map(|i| (i * 37 + 11) as u16).collect();
            let map = OutputMap::new(&h, colors);
            for r in 0..6 {
                let reference: Vec<u16> = (0..768usize)
                    .map(|p| {
                        let c = colors[r][p % 6];
                        let line = match c {
                            0 => R2 + r / 2,
                            2 => B2 + r / 2,
                            _ => G2 + r,
                        };
                        let i = if xtrans { (((p * 2 / 3) & !1) | ((p % 3) & 1)) + ((p % 3) >> 1) } else { p >> 1 };
                        lines[line * stride + 1 + i]
                    })
                    .collect();
                for (sx, len) in [(0, 768), (0, 744), (0, 13), (6, 25), (3, 19), (7, 700)] {
                    let mut got = vec![0u16; len];
                    map.emit_row(r, sx, &mut got, &lines);
                    assert_eq!(&got[..], &reference[sx..sx + len], "xtrans={xtrans} r={r} sx={sx} len={len}");
                }
            }
        }
    }
}
