//! GGML tensor types and CPU dequantization kernels.
//!
//! The block layouts exactly mirror those in `ggml-common.h` and the reference
//! `dequantize_row_*` implementations in `ggml-quants.c`, so the numbers match
//! llama.cpp bit for bit.

use std::fmt;

/// GGML tensor type ids (subset).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::BF16,
            _ => return None,
        })
    }

    /// Number of elements in one quant block. Plain float types use a block of 1.
    pub fn block_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 => 32,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => 256,
        }
    }

    /// Number of bytes in one quant block.
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 | GgmlType::BF16 => 2,
            GgmlType::Q4_0 => 18,  // f16 d + 16 bytes nibbles
            GgmlType::Q4_1 => 20,  // 2*f16 + 16 bytes
            GgmlType::Q5_0 => 22,  // f16 d + 4 bytes qh + 16 bytes
            GgmlType::Q5_1 => 24,  // 2*f16 + 4 bytes qh + 16 bytes
            GgmlType::Q8_0 => 34,  // f16 d + 32 i8
            GgmlType::Q4K => 144,  // 2*f16 + 12 scales + 128 qs
            GgmlType::Q5K => 176,  // 2*f16 + 12 scales + 32 qh + 128 qs
            GgmlType::Q6K => 210,  // 128 ql + 64 qh + 16 scales + f16 d
        }
    }

    /// Bytes occupied by a single row of `n` elements.
    pub fn row_size(self, n: usize) -> usize {
        let bs = self.block_size();
        debug_assert_eq!(n % bs, 0, "row length {n} not divisible by block size {bs}");
        (n / bs) * self.type_size()
    }

    /// GGML type id used in the GGUF tensor-info table (inverse of `from_u32`).
    pub fn as_u32(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::BF16 => 30,
        }
    }

    pub fn name(self) -> &'static str {        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::BF16 => "BF16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::available() {
            return unsafe { f16_to_f32_f16c(h) };
        }
    }
    f16_to_f32_scalar(h)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
#[inline]
unsafe fn f16_to_f32_f16c(h: u16) -> f32 {
    let v = std::arch::x86_64::_mm_cvtsi32_si128(h as i32);
    let f = std::arch::x86_64::_mm_cvtph_ps(v);
    std::arch::x86_64::_mm_cvtss_f32(f)
}

pub fn f16_to_f32_scalar(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn read_f16(src: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
}

#[inline]
fn read_u32(src: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]])
}

/// `get_scale_min_k4` from `ggml-quants.c`.
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize one contiguous row of `dst.len()` elements from `src`.
///
/// `src` must contain at least `ty.row_size(dst.len())` bytes.
/// `dst.len()` must be a multiple of the type's block size.
pub fn dequant_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (i, d) in dst.iter_mut().enumerate() {
                let o = i * 4;
                *d = f32::from_le_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
            }
        }
        GgmlType::F16 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = read_f16(src, i * 2);
            }
        }
        GgmlType::BF16 => {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = bf16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
            }
        }
        GgmlType::Q4_0 => {
            let bs = 32;
            for (bi, block) in src.chunks_exact(18).enumerate() {
                let d = read_f16(block, 0);
                let qs = &block[2..18];
                let base = bi * bs;
                for j in 0..bs / 2 {
                    let x0 = ((qs[j] & 0x0F) as i32 - 8) as f32;
                    let x1 = ((qs[j] >> 4) as i32 - 8) as f32;
                    dst[base + j] = x0 * d;
                    dst[base + j + bs / 2] = x1 * d;
                }
            }
        }
        GgmlType::Q4_1 => {
            let bs = 32;
            for (bi, block) in src.chunks_exact(20).enumerate() {
                let d = read_f16(block, 0);
                let m = read_f16(block, 2);
                let qs = &block[4..20];
                let base = bi * bs;
                for j in 0..bs / 2 {
                    let x0 = (qs[j] & 0x0F) as f32;
                    let x1 = (qs[j] >> 4) as f32;
                    dst[base + j] = x0 * d + m;
                    dst[base + j + bs / 2] = x1 * d + m;
                }
            }
        }
        GgmlType::Q5_0 => {
            let bs = 32;
            for (bi, block) in src.chunks_exact(22).enumerate() {
                let d = read_f16(block, 0);
                let qh = read_u32(block, 2);
                let qs = &block[6..22];
                let base = bi * bs;
                for j in 0..bs / 2 {
                    let xh0 = (((qh >> j) << 4) & 0x10) as u8;
                    let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
                    let x0 = (((qs[j] & 0x0F) | xh0) as i32 - 16) as f32;
                    let x1 = (((qs[j] >> 4) | xh1) as i32 - 16) as f32;
                    dst[base + j] = x0 * d;
                    dst[base + j + bs / 2] = x1 * d;
                }
            }
        }
        GgmlType::Q5_1 => {
            let bs = 32;
            for (bi, block) in src.chunks_exact(24).enumerate() {
                let d = read_f16(block, 0);
                let m = read_f16(block, 2);
                let qh = read_u32(block, 4);
                let qs = &block[8..24];
                let base = bi * bs;
                for j in 0..bs / 2 {
                    let xh0 = (((qh >> j) << 4) & 0x10) as u8;
                    let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
                    let x0 = ((qs[j] & 0x0F) | xh0) as f32;
                    let x1 = ((qs[j] >> 4) | xh1) as f32;
                    dst[base + j] = x0 * d + m;
                    dst[base + j + bs / 2] = x1 * d + m;
                }
            }
        }
        GgmlType::Q8_0 => {
            let bs = 32;
            for (bi, block) in src.chunks_exact(34).enumerate() {
                let d = read_f16(block, 0);
                let base = bi * bs;
                for j in 0..bs {
                    dst[base + j] = (block[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q4K => {
            let bs = 256;
            for (bi, block) in src.chunks_exact(144).enumerate() {
                let d = read_f16(block, 0);
                let dmin = read_f16(block, 2);
                let scales = &block[4..16];
                let qs = &block[16..144];
                let base = bi * bs;
                let mut is = 0usize;
                let mut qoff = 0usize;
                let mut yoff = base;
                while yoff < base + bs {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d1 = d * sc1 as f32;
                    let mm1 = dmin * m1 as f32;
                    let d2 = d * sc2 as f32;
                    let mm2 = dmin * m2 as f32;
                    for l in 0..32 {
                        dst[yoff + l] = d1 * (qs[qoff + l] & 0x0F) as f32 - mm1;
                    }
                    for l in 0..32 {
                        dst[yoff + 32 + l] = d2 * (qs[qoff + l] >> 4) as f32 - mm2;
                    }
                    yoff += 64;
                    qoff += 32;
                    is += 2;
                }
            }
        }
        GgmlType::Q5K => {
            let bs = 256;
            for (bi, block) in src.chunks_exact(176).enumerate() {
                let d = read_f16(block, 0);
                let dmin = read_f16(block, 2);
                let scales = &block[4..16];
                let qh = &block[16..48];
                let qs = &block[48..176];
                let base = bi * bs;
                let mut is = 0usize;
                let mut qoff = 0usize;
                let mut yoff = base;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                while yoff < base + bs {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d1 = d * sc1 as f32;
                    let mm1 = dmin * m1 as f32;
                    let d2 = d * sc2 as f32;
                    let mm2 = dmin * m2 as f32;
                    for l in 0..32 {
                        let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                        dst[yoff + l] = d1 * ((qs[qoff + l] & 0x0F) + hi) as f32 - mm1;
                    }
                    for l in 0..32 {
                        let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                        dst[yoff + 32 + l] = d2 * ((qs[qoff + l] >> 4) + hi) as f32 - mm2;
                    }
                    yoff += 64;
                    qoff += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q6K => {
            let bs = 256;
            for (bi, block) in src.chunks_exact(210).enumerate() {
                let d = read_f16(block, 208);
                let base = bi * bs;
                let mut ql_off = 0usize;
                let mut qh_off = 128usize;
                let mut sc_off = 192usize;
                let mut yoff = base;
                let mut n = 0usize;
                while n < bs {
                    for l in 0..32 {
                        let ql = &block[ql_off..];
                        let qh = &block[qh_off..];
                        let sc = &block[sc_off..];
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let s = |k: usize| (sc[k] as i8) as f32;
                        dst[yoff + l] = d * s(is) * q1 as f32;
                        dst[yoff + l + 32] = d * s(is + 2) * q2 as f32;
                        dst[yoff + l + 64] = d * s(is + 4) * q3 as f32;
                        dst[yoff + l + 96] = d * s(is + 6) * q4 as f32;
                    }
                    yoff += 128;
                    ql_off += 64;
                    qh_off += 32;
                    sc_off += 8;
                    n += 128;
                }
            }
        }
    }
}

/// Fused dequantize-and-dot for a single row: returns `Σ dequant(src)[i] * x[i]`.
///
/// Dispatches to the hand-written AVX2/FMA kernels when the CPU supports them,
/// and falls back to the portable scalar implementation otherwise.
#[inline]
pub fn dequant_dot(ty: GgmlType, src: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(v) = crate::simd::dequant_dot(ty, src, x) {
            return v;
        }
    }
    dequant_dot_scalar(ty, src, x)
}

/// Portable fused dequantize-and-dot for a single row.
///
/// Instead of materializing the whole dequantized row into a temporary `f32`
/// buffer and then dotting (two passes over `cols` floats), this keeps the
/// intermediate values on the stack / in registers, roughly halving the memory
/// traffic of the matmul hot path. The result is numerically equivalent (up to
/// floating-point summation order).
pub fn dequant_dot_scalar(ty: GgmlType, src: &[u8], x: &[f32]) -> f32 {
    match ty {
        GgmlType::F32 => {
            let mut acc = 0f32;
            for (i, &xi) in x.iter().enumerate() {
                let o = i * 4;
                acc += f32::from_le_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]) * xi;
            }
            acc
        }
        GgmlType::F16 => {
            let mut acc = 0f32;
            for (i, &xi) in x.iter().enumerate() {
                acc += read_f16(src, i * 2) * xi;
            }
            acc
        }
        GgmlType::BF16 => {
            let mut acc = 0f32;
            for (i, &xi) in x.iter().enumerate() {
                acc += bf16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]])) * xi;
            }
            acc
        }
        GgmlType::Q4_0 => {
            let mut acc = 0f32;
            for (bi, block) in src.chunks_exact(18).enumerate() {
                let d = read_f16(block, 0);
                let qs = &block[2..18];
                let xb = &x[bi * 32..bi * 32 + 32];
                let mut s = 0f32;
                for j in 0..16 {
                    let x0 = ((qs[j] & 0x0F) as i32 - 8) as f32;
                    let x1 = ((qs[j] >> 4) as i32 - 8) as f32;
                    s += x0 * xb[j] + x1 * xb[j + 16];
                }
                acc += d * s;
            }
            acc
        }
        GgmlType::Q8_0 => {
            let mut acc = 0f32;
            for (bi, block) in src.chunks_exact(34).enumerate() {
                let d = read_f16(block, 0);
                let xb = &x[bi * 32..bi * 32 + 32];
                let mut s = 0f32;
                for j in 0..32 {
                    s += (block[2 + j] as i8) as f32 * xb[j];
                }
                acc += d * s;
            }
            acc
        }
        GgmlType::Q4K => {
            let mut acc = 0f32;
            for (bi, block) in src.chunks_exact(144).enumerate() {
                let d = read_f16(block, 0);
                let dmin = read_f16(block, 2);
                let scales = &block[4..16];
                let qs = &block[16..144];
                let xb = &x[bi * 256..bi * 256 + 256];
                let mut is = 0usize;
                let mut qoff = 0usize;
                let mut yoff = 0usize;
                let mut s = 0f32;
                while yoff < 256 {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d1 = d * sc1 as f32;
                    let mm1 = dmin * m1 as f32;
                    let d2 = d * sc2 as f32;
                    let mm2 = dmin * m2 as f32;
                    for l in 0..32 {
                        s += (d1 * (qs[qoff + l] & 0x0F) as f32 - mm1) * xb[yoff + l];
                        s += (d2 * (qs[qoff + l] >> 4) as f32 - mm2) * xb[yoff + 32 + l];
                    }
                    yoff += 64;
                    qoff += 32;
                    is += 2;
                }
                acc += s;
            }
            acc
        }
        GgmlType::Q6K => {
            let mut acc = 0f32;
            for (bi, block) in src.chunks_exact(210).enumerate() {
                let d = read_f16(block, 208);
                let xb = &x[bi * 256..bi * 256 + 256];
                let mut ql_off = 0usize;
                let mut qh_off = 128usize;
                let mut sc_off = 192usize;
                let mut yoff = 0usize;
                let mut n = 0usize;
                while n < 256 {
                    for l in 0..32 {
                        let ql = &block[ql_off..];
                        let qh = &block[qh_off..];
                        let sc = &block[sc_off..];
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let s = |k: usize| (sc[k] as i8) as f32;
                        acc += d * s(is) * q1 as f32 * xb[yoff + l];
                        acc += d * s(is + 2) * q2 as f32 * xb[yoff + l + 32];
                        acc += d * s(is + 4) * q3 as f32 * xb[yoff + l + 64];
                        acc += d * s(is + 6) * q4 as f32 * xb[yoff + l + 96];
                    }
                    yoff += 128;
                    ql_off += 64;
                    qh_off += 32;
                    sc_off += 8;
                    n += 128;
                }
            }
            acc
        }
        // Generic fallback (Q4_1/Q5_0/Q5_1/...): dequantize one block at a time.
        _ => {
            let bs = ty.block_size();
            let ts = ty.type_size();
            let mut tmp = [0f32; 256];
            let mut acc = 0f32;
            for (bi, block) in src.chunks_exact(ts).enumerate() {
                dequant_row(ty, block, &mut tmp[..bs]);
                acc += crate::ops::dot(&tmp[..bs], &x[bi * bs..bi * bs + bs]);
            }
            acc
        }
    }
}
