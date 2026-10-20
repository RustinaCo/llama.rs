//! Hand-written x86-64 AVX2 + FMA kernels for the matmul hot path.
//!
//! The portable code in [`crate::quant`] and [`crate::ops`] is correct but
//! scalar; on x86-64 these kernels dequantize (nibble/bit unpack) and multiply
//! eight `f32` lanes at a time. Runtime feature detection lives in
//! [`available`]; every public entry point returns `None` when AVX2 + FMA (and
//! F16C, used for the half-precision scale factors) are not present, so callers
//! transparently fall back to the scalar path.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU8, Ordering};

use crate::quant::{self, GgmlType};

/// Cached CPU capability check (AVX2 + FMA + F16C).
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn available() -> bool {
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let ok = std::env::var_os("RUST_LLAMA_NO_SIMD").is_none()
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c");
            STATE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}

/// AVX2 fused dequantize-and-dot; `None` if the CPU lacks the features or the
/// tensor type has no specialized kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dequant_dot(ty: GgmlType, src: &[u8], x: &[f32]) -> Option<f32> {
    if !available() {
        return None;
    }
    match ty {
        GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::F16
        | GgmlType::F32 => {}
        _ => return None,
    }
    // Safety: guarded by the runtime feature check above.
    Some(unsafe { dequant_dot_avx2(ty, src, x) })
}

/// AVX2 dot product of two `f32` slices; `None` if unsupported.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if !available() {
        return None;
    }
    // Safety: guarded by the runtime feature check above.
    Some(unsafe { dot_f32_avx2(a, b) })
}

/// AVX2 RMSNorm: `out = x / rms(x) * weight`.
///
/// Callers must have checked [`available`]; the tail preserves the exact
/// scalar op order (`x*scale*weight`), only the sum-of-squares reduction
/// order differs (last-ulp only).
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    rmsnorm_core(out.as_mut_ptr(), x.as_ptr(), weight.as_ptr(), x.len(), eps);
}

/// In-place AVX2 RMSNorm (`ptr[i] *= scale * weight[i]`).
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn rmsnorm_inplace(ptr: &mut [f32], weight: &[f32], eps: f32) {
    rmsnorm_core(ptr.as_mut_ptr(), ptr.as_ptr(), weight.as_ptr(), ptr.len(), eps);
}

/// Quantized activation row: `x` blocked into groups of 32 signed bytes.
///
/// `scales[b]` and `sums[b]` describe `qs[32*b..32*b+32]`; `dots` computed
/// against these blocks need one float combine per block. Same approximation
/// llama.cpp uses for activations (relative error ~1%).
#[derive(Clone, Debug)]
pub struct Q8Row {
    pub scales: Vec<f32>,
    pub qs: Vec<i8>,
    pub sums: Vec<i32>,
    /// Sum of each 16-value half of a 32-byte block: `sums16[2*b + h]`.
    ///
    /// The `Q6_K` kernel consumes 16-value halves, so its per-row loop needs
    /// these. Computing them here (once per activation) instead of inside the
    /// row loop removes a scalar re-sum of the activation from every weight
    /// row — that sum is identical for all rows and used to dominate the
    /// `Q6_K` kernel.
    pub sums16: Vec<i32>,
}

/// Quantize activation row into Q8 blocks, using AVX2 when available.
pub fn quantize_q8(x: &[f32]) -> Q8Row {
    #[cfg(target_arch = "x86_64")]
    {
        if available() {
            return unsafe { quantize_q8_avx2(x) };
        }
    }
    quantize_q8_scalar(x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_q8_avx2(x: &[f32]) -> Q8Row {
    let nb = x.len() / 32;
    let mut scales = Vec::with_capacity(nb);
    let mut qs: Vec<i8> = Vec::with_capacity(nb * 32);
    let mut sums = Vec::with_capacity(nb);
    let mut sums16 = Vec::with_capacity(nb * 2);

    let sign_mask = _mm256_set1_ps(-0.0);
    let ones = _mm_set1_epi16(1);

    for b in 0..nb {
        let blk_ptr = x.as_ptr().add(b * 32);
        let v0 = _mm256_loadu_ps(blk_ptr);
        let v1 = _mm256_loadu_ps(blk_ptr.add(8));
        let v2 = _mm256_loadu_ps(blk_ptr.add(16));
        let v3 = _mm256_loadu_ps(blk_ptr.add(24));

        let abs0 = _mm256_andnot_ps(sign_mask, v0);
        let abs1 = _mm256_andnot_ps(sign_mask, v1);
        let abs2 = _mm256_andnot_ps(sign_mask, v2);
        let abs3 = _mm256_andnot_ps(sign_mask, v3);

        let max01 = _mm256_max_ps(abs0, abs1);
        let max23 = _mm256_max_ps(abs2, abs3);
        let max_all = _mm256_max_ps(max01, max23);

        let max_lo = _mm256_castps256_ps128(max_all);
        let max_hi = _mm256_extractf128_ps(max_all, 1);
        let max_128 = _mm_max_ps(max_lo, max_hi);
        let max_64 = _mm_max_ps(max_128, _mm_movehl_ps(max_128, max_128));
        let max_32 = _mm_max_ss(max_64, _mm_shuffle_ps(max_64, max_64, 0x55));
        let amax = _mm_cvtss_f32(max_32);

        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        scales.push(d);

        let inv_v = _mm256_set1_ps(inv);

        // Convert 8 floats to 8 int32 with round-to-nearest
        let q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, inv_v));
        let q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, inv_v));
        let q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, inv_v));
        let q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, inv_v));

        // Pack 32-bit to 16-bit signed with saturation
        let pack01 = _mm256_packs_epi32(q0, q1);
        let pack23 = _mm256_packs_epi32(q2, q3);

        let pack01_perm = _mm256_permute4x64_epi64(pack01, 0xd8);
        let pack23_perm = _mm256_permute4x64_epi64(pack23, 0xd8);

        // Pack 16-bit to 8-bit signed with saturation
        let pack_all = _mm256_packs_epi16(pack01_perm, pack23_perm);
        let pack_all_perm = _mm256_permute4x64_epi64(pack_all, 0xd8);

        let qs_out_ptr: *mut i8 = qs.as_mut_ptr().add(b * 32);
        _mm256_storeu_si256(qs_out_ptr as *mut __m256i, pack_all_perm);
        qs.set_len((b + 1) * 32);

        // Compute sums
        let lo128 = _mm256_castsi256_si128(pack_all_perm);
        let hi128 = _mm256_extracti128_si256(pack_all_perm, 1);

        let s_lo_16 = _mm_maddubs_epi16(_mm_set1_epi8(1), lo128);
        let s_lo_32 = _mm_madd_epi16(s_lo_16, ones);
        let hi64 = _mm_shuffle_epi32(s_lo_32, 0xee);
        let s_32 = _mm_add_epi32(s_lo_32, hi64);
        let s_lo = _mm_cvtsi128_si32(_mm_add_epi32(s_32, _mm_shuffle_epi32(s_32, 0x55)));

        let s_hi_16 = _mm_maddubs_epi16(_mm_set1_epi8(1), hi128);
        let s_hi_32 = _mm_madd_epi16(s_hi_16, ones);
        let hi64_h = _mm_shuffle_epi32(s_hi_32, 0xee);
        let s_32_h = _mm_add_epi32(s_hi_32, hi64_h);
        let s_hi = _mm_cvtsi128_si32(_mm_add_epi32(s_32_h, _mm_shuffle_epi32(s_32_h, 0x55)));

        sums.push(s_lo + s_hi);
        sums16.push(s_lo);
        sums16.push(s_hi);
    }

    Q8Row {
        scales,
        qs,
        sums,
        sums16,
    }
}

/// Portable scalar activation quantization (exact output, any CPU).
pub fn quantize_q8_scalar(x: &[f32]) -> Q8Row {
    let nb = x.len() / 32;
    let mut scales = Vec::with_capacity(nb);
    let mut qs = Vec::with_capacity(nb * 32);
    let mut sums = Vec::with_capacity(nb);
    let mut sums16 = Vec::with_capacity(nb * 2);
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let mut amax = 0f32;
        for &v in blk {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        scales.push(d);
        let mut s = 0i32;
        let mut s_lo = 0i32;
        for (i, &v) in blk.iter().enumerate() {
            let q = (v * inv).round().clamp(-128.0, 127.0) as i8;
            s += q as i32;
            if i < 16 {
                s_lo += q as i32;
            }
            qs.push(q);
        }
        sums.push(s);
        sums16.push(s_lo);
        sums16.push(s - s_lo);
    }
    Q8Row {
        scales,
        qs,
        sums,
        sums16,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dequant_dot_avx2(ty: GgmlType, src: &[u8], x: &[f32]) -> f32 {
    match ty {
        GgmlType::Q4K => q4k(src, x),
        GgmlType::Q6K => q6k(src, x),
        GgmlType::Q8_0 => q8_0(src, x),
        GgmlType::Q4_0 => q4_0(src, x),
        GgmlType::F16 => f16(src, x),
        _ => f32_bytes(src, x),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let hi64 = _mm_movehl_ps(s, s);
    let s2 = _mm_add_ps(s, hi64);
    let hi32 = _mm_shuffle_ps(s2, s2, 0x55);
    let s3 = _mm_add_ss(s2, hi32);
    _mm_cvtss_f32(s3)
}

/// `out = a * b - c`, lane-wise (matches the scalar dequant op order).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn sub_mul(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_sub_ps(_mm256_mul_ps(a, b), c)
}

/// Shared RMSNorm core: `out[i] = x[i] * scale * w[i]` with
/// `scale = 1/sqrt(sumsq/n + eps)`. `out` may alias `x` (in-place use reads
/// each lane before writing it). Four accumulators hide FMA latency.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rmsnorm_core(out: *mut f32, x: *const f32, w: *const f32, n: usize, eps: f32) {
    let (mut a0, mut a1, mut a2, mut a3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut i = 0;
    while i + 32 <= n {
        let v0 = _mm256_loadu_ps(x.add(i));
        let v1 = _mm256_loadu_ps(x.add(i + 8));
        let v2 = _mm256_loadu_ps(x.add(i + 16));
        let v3 = _mm256_loadu_ps(x.add(i + 24));
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
        a2 = _mm256_fmadd_ps(v2, v2, a2);
        a3 = _mm256_fmadd_ps(v3, v3, a3);
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        a0 = _mm256_fmadd_ps(v, v, a0);
        i += 8;
    }
    let mut ss = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    while i < n {
        let v = *x.add(i);
        ss += v * v;
        i += 1;
    }
    let s = 1.0 / (ss / n as f32 + eps).sqrt();
    let sv = _mm256_set1_ps(s);
    let mut j = 0;
    while j + 8 <= n {
        let v = _mm256_mul_ps(_mm256_loadu_ps(x.add(j)), sv);
        _mm256_storeu_ps(out.add(j), _mm256_mul_ps(v, _mm256_loadu_ps(w.add(j))));
        j += 8;
    }
    while j < n {
        *out.add(j) = *x.add(j) * s * *w.add(j);
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (mut a0, mut a1, mut a2, mut a3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut i = 0;
    while i + 32 <= n {
        a0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            a0,
        );
        a1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            a1,
        );
        a2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 16)),
            _mm256_loadu_ps(b.as_ptr().add(i + 16)),
            a2,
        );
        a3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 24)),
            _mm256_loadu_ps(b.as_ptr().add(i + 24)),
            a3,
        );
        i += 32;
    }
    while i + 8 <= n {
        a0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            a0,
        );
        i += 8;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    let mut s = hsum(acc);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// `F32` tensor row stored as raw little-endian bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn f32_bytes(src: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_castsi256_ps(_mm256_loadu_si256(src.as_ptr().add(i * 4) as *const __m256i));
        acc = _mm256_fmadd_ps(v, _mm256_loadu_ps(x.as_ptr().add(i)), acc);
        i += 8;
    }
    let mut s = hsum(acc);
    while i < n {
        let o = i * 4;
        let v = f32::from_le_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
        s += v * x[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f16(src: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(src.as_ptr().add(i * 2) as *const __m128i);
        let v = _mm256_cvtph_ps(h);
        acc = _mm256_fmadd_ps(v, _mm256_loadu_ps(x.as_ptr().add(i)), acc);
        i += 8;
    }
    let mut s = hsum(acc);
    while i < n {
        s += quant::f16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]])) * x[i];
        i += 1;
    }
    s
}

/// `Q4_0`: 18-byte blocks, `d` then 16 bytes of packed nibbles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4_0(src: &[u8], x: &[f32]) -> f32 {
    let m0f = _mm_set1_epi8(0x0F);
    let c8 = _mm256_set1_ps(8.0);
    let blocks = x.len() / 32;
    let mut total = 0f32;
    for bi in 0..blocks {
        let base = bi * 18;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base], src[base + 1]]));
        let qs = src.as_ptr().add(base + 2);
        let xp = x.as_ptr().add(bi * 32);
        let mut a_lo = _mm256_setzero_ps();
        let mut a_hi = _mm256_setzero_ps();
        for k in 0..2 {
            let b = _mm_loadl_epi64(qs.add(k * 8) as *const __m128i);
            let lo = _mm_and_si128(b, m0f);
            let hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
            let lof = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), c8);
            let hif = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), c8);
            a_lo = _mm256_fmadd_ps(lof, _mm256_loadu_ps(xp.add(k * 8)), a_lo);
            a_hi = _mm256_fmadd_ps(hif, _mm256_loadu_ps(xp.add(16 + k * 8)), a_hi);
        }
        total += d * hsum(_mm256_add_ps(a_lo, a_hi));
    }
    total
}

/// `Q8_0`: 34-byte blocks, `d` then 32 signed bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q8_0(src: &[u8], x: &[f32]) -> f32 {
    let blocks = x.len() / 32;
    let mut total = 0f32;
    for bi in 0..blocks {
        let base = bi * 34;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base], src[base + 1]]));
        let qs = src.as_ptr().add(base + 2) as *const i8;
        let xp = x.as_ptr().add(bi * 32);
        let mut acc = _mm256_setzero_ps();
        for k in 0..4 {
            let b = _mm_loadl_epi64(qs.add(k * 8) as *const __m128i);
            let v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
            acc = _mm256_fmadd_ps(v, _mm256_loadu_ps(xp.add(k * 8)), acc);
        }
        total += d * hsum(acc);
    }
    total
}

/// `Q4_K`: 144-byte super-blocks of 256 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q4k(src: &[u8], x: &[f32]) -> f32 {
    let m0f = _mm_set1_epi8(0x0F);
    let nblocks = x.len() / 256;
    let mut total = 0f32;
    for bi in 0..nblocks {
        let base = bi * 144;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base], src[base + 1]]));
        let dmin = quant::f16_to_f32(u16::from_le_bytes([src[base + 2], src[base + 3]]));
        let scales = &src[base + 4..base + 16];
        let qs = src.as_ptr().add(base + 16);
        let xp = x.as_ptr().add(bi * 256);
        let mut block_dot = 0f32;
        for g in 0..4usize {
            let (sc1, m1) = quant::get_scale_min_k4(2 * g, scales);
            let (sc2, m2) = quant::get_scale_min_k4(2 * g + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            let qsg = qs.add(g * 32);
            let xlo = xp.add(g * 64);
            let xhi = xp.add(g * 64 + 32);
            let mut sq_lo = _mm256_setzero_ps();
            let mut sq_hi = _mm256_setzero_ps();
            let mut sx_lo = _mm256_setzero_ps();
            let mut sx_hi = _mm256_setzero_ps();
            for k in 0..4 {
                let b = _mm_loadl_epi64(qsg.add(k * 8) as *const __m128i);
                let lo = _mm_and_si128(b, m0f);
                let hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                let lof = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
                let hif = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
                let vlo = _mm256_loadu_ps(xlo.add(k * 8));
                let vhi = _mm256_loadu_ps(xhi.add(k * 8));
                sq_lo = _mm256_fmadd_ps(lof, vlo, sq_lo);
                sq_hi = _mm256_fmadd_ps(hif, vhi, sq_hi);
                sx_lo = _mm256_add_ps(sx_lo, vlo);
                sx_hi = _mm256_add_ps(sx_hi, vhi);
            }
            block_dot +=
                d1 * hsum(sq_lo) - mm1 * hsum(sx_lo) + d2 * hsum(sq_hi) - mm2 * hsum(sx_hi);
        }
        total += block_dot;
    }
    total
}

/// Horizontal sum of eight `i32` lanes (integer twin of [`hsum`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let hi64 = _mm_shuffle_epi32(s, 0xee);
    let s2 = _mm_add_epi32(s, hi64);
    let hi32 = _mm_shuffle_epi32(s2, 0x55);
    let s3 = _mm_add_epi32(s2, hi32);
    _mm_cvtsi128_si32(s3)
}

/// Integer-exact `Q4_K` × int8-activation dot (one 144-byte super-block row
/// against [`Q8Row`]).
///
/// Nibbles (0..15, unsigned) pair with signed activation bytes through
/// `_mm256_maddubs_epi16`, so every 32-lane partial sum is bit-exact; only the
/// per-group float combine (`d*S - m*sum`) rounds. No saturation risk: a
/// maddubs lane holds at most `2*15*128 = 3840`, the final `i32` at most
/// `32*15*128 = 61440`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4k_q8_avx2(src: &[u8], act: &Q8Row) -> f32 {
    let n256 = src.len() / 144;
    debug_assert_eq!(act.qs.len(), n256 * 256);
    let m0f = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut total = 0f32;
    for bi in 0..n256 {
        let base = bi * 144;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base], src[base + 1]]));
        let dmin = quant::f16_to_f32(u16::from_le_bytes([src[base + 2], src[base + 3]]));
        let scales = &src[base + 4..base + 16];
        let qs = src.as_ptr().add(base + 16);
        for g in 0..4usize {
            let (sc1, m1) = quant::get_scale_min_k4(2 * g, scales);
            let (sc2, m2) = quant::get_scale_min_k4(2 * g + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            let ab = bi * 8 + 2 * g;
            let sa = act.sums[ab] as f32;
            let sb = act.sums[ab + 1] as f32;
            let da = act.scales[ab];
            let db = act.scales[ab + 1];
            // 32 packed bytes: low nibbles = first 32 values, high = next 32.
            let p = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
            let lo = _mm256_and_si256(p, m0f);
            let hi = _mm256_and_si256(_mm256_srli_epi16(p, 4), m0f);
            let a = _mm256_loadu_si256(act.qs.as_ptr().add(ab * 32) as *const __m256i);
            let b =
                _mm256_loadu_si256(act.qs.as_ptr().add((ab + 1) * 32) as *const __m256i);
            let s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a), ones);
            let s2 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, b), ones);
            let i1 = hsum_i32(s1) as f32;
            let i2 = hsum_i32(s2) as f32;
            total += da * (d1 * i1 - mm1 * sa) + db * (d2 * i2 - mm2 * sb);
        }
    }
    total
}

/// Horizontal sum of four `i32` lanes in a 128-bit vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_128(v: __m128i) -> i32 {
    let hi64 = _mm_shuffle_epi32(v, 0xee);
    let s = _mm_add_epi32(v, hi64);
    let hi32 = _mm_shuffle_epi32(s, 0x55);
    let s2 = _mm_add_epi32(s, hi32);
    _mm_cvtsi128_si32(s2)
}

/// Integer-exact `Q6_K` × int8-activation dot.
///
/// The stored 6-bit values are unsigned (`0..63`, low 4 bits in `ql`, top 2
/// in `qh`); signed value = `u - 32`, so each dot is `Σu*a - 32*Σa` with the
/// first term bit-exact via `_mm_maddubs_epi16`. Layout forces 16-lane
/// chunks (scales change at `l = 16`); everything else mirrors the `Q4_K`
/// kernel above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q6k_q8_avx2(src: &[u8], act: &Q8Row) -> f32 {
    let n256 = src.len() / 210;
    debug_assert_eq!(act.qs.len(), n256 * 256);
    let m0f = _mm_set1_epi8(0x0F);
    let m03 = _mm_set1_epi8(0x03);
    let ones = _mm_set1_epi16(1);
    let mut total = 0f32;
    for bi in 0..n256 {
        let base = bi * 210;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base + 208], src[base + 209]]));
        for half in 0..2usize {
            let ql = src.as_ptr().add(base + half * 64);
            let qh = src.as_ptr().add(base + 128 + half * 32);
            let sc = &src[base + 192 + half * 8..base + 192 + half * 8 + 8];
            // Output blocks: u1..u4 cover act blocks ab..ab+3.
            let ab = bi * 8 + half * 4;
            for sub in 0..2usize {
                // 16 lanes; scales switch at l = 16.
                let o = sub * 16;
                let s1 = (sc[sub] as i8) as f32;
                let s2 = (sc[sub + 2] as i8) as f32;
                let s3 = (sc[sub + 4] as i8) as f32;
                let s4 = (sc[sub + 6] as i8) as f32;
                let ql0 = _mm_loadu_si128(ql.add(o) as *const __m128i);
                let ql1 = _mm_loadu_si128(ql.add(32 + o) as *const __m128i);
                let qh16 = _mm_loadu_si128(qh.add(o) as *const __m128i);
                let u1 = _mm_or_si128(
                    _mm_and_si128(ql0, m0f),
                    _mm_slli_epi16(_mm_and_si128(qh16, m03), 4),
                );
                let u2 = _mm_or_si128(
                    _mm_and_si128(ql1, m0f),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, 2), m03), 4),
                );
                let u3 = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql0, 4), m0f),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, 4), m03), 4),
                );
                let u4 = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(ql1, 4), m0f),
                    _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh16, 6), m03), 4),
                );
                let aq = act.qs.as_ptr();
                let a1 = _mm_loadu_si128(aq.add((ab) * 32 + o) as *const __m128i);
                let a2 = _mm_loadu_si128(aq.add((ab + 1) * 32 + o) as *const __m128i);
                let a3 = _mm_loadu_si128(aq.add((ab + 2) * 32 + o) as *const __m128i);
                let a4 = _mm_loadu_si128(aq.add((ab + 3) * 32 + o) as *const __m128i);
                let us = [u1, u2, u3, u4];
                let aa = [a1, a2, a3, a4];
                let scs = [s1, s2, s3, s4];
                for c in 0..4 {
                    let ss = hsum_i32_128(_mm_madd_epi16(_mm_maddubs_epi16(us[c], aa[c]), ones));
                    // 16-value half-block sum: precomputed in `quantize_q8`
                    // (identical integer sum, so the dot stays bit-exact).
                    let sa = act.sums16[(ab + c) * 2 + sub];
                    let da = act.scales[ab + c];
                    total += d * scs[c] * (ss as f32 - 32.0 * sa as f32) * da;
                }
            }
        }
    }
    total
}

/// AVX2 `dst[i] += alpha * x[i]` with plain multiply-then-add (NOT FMA), so
/// the result is bit-identical to the scalar loop. Callers must have checked
/// [`available`].
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn saxpy(dst: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(dst.len(), x.len());
    let n = dst.len();
    let a = _mm256_set1_ps(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(_mm256_mul_ps(a, v), d));
        i += 8;
    }
    while i < n {
        *dst.as_mut_ptr().add(i) += alpha * *x.as_ptr().add(i);
        i += 1;
    }
}

/// Fused multi-dot: dots of a 256-value `buf` (one superblock position)
/// against the same position of `n` activation rows.
///
/// `xs` holds `n` rows of `xs_stride` elements; this position starts at
/// `xs_off` within each row. One pass over `buf` serves all N tokens.
/// Results are ADDED (`+=`) so successive positions accumulate; the caller
/// zeroes the outputs first.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_multi(
    buf: &[f32],
    xs: &[f32],
    xs_off: usize,
    xs_stride: usize,
    n: usize,
    out: *mut f32,
    out_stride: usize,
) {
    let cols = buf.len();
    debug_assert!(cols > 0 && cols % 8 == 0);
    let mut k = 0;
    while k + 4 <= n {
        let x0 = xs.as_ptr().add(k * xs_stride + xs_off);
        let x1 = xs.as_ptr().add((k + 1) * xs_stride + xs_off);
        let x2 = xs.as_ptr().add((k + 2) * xs_stride + xs_off);
        let x3 = xs.as_ptr().add((k + 3) * xs_stride + xs_off);
        let (mut v0, mut v1, mut v2, mut v3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
        let mut i = 0;
        while i + 8 <= cols {
            let w = _mm256_loadu_ps(buf.as_ptr().add(i));
            v0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(i)), v0);
            v1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(i)), v1);
            v2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(i)), v2);
            v3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x3.add(i)), v3);
            i += 8;
        }
        while i < cols {
            let w = *buf.as_ptr().add(i);
            a0 += w * *x0.add(i);
            a1 += w * *x1.add(i);
            a2 += w * *x2.add(i);
            a3 += w * *x3.add(i);
            i += 1;
        }
        *out.add(k * out_stride) += hsum(v0) + a0;
        *out.add((k + 1) * out_stride) += hsum(v1) + a1;
        *out.add((k + 2) * out_stride) += hsum(v2) + a2;
        *out.add((k + 3) * out_stride) += hsum(v3) + a3;
        k += 4;
    }
    while k < n {
        let xk = &xs[k * xs_stride + xs_off..k * xs_stride + xs_off + cols];
        *out.add(k * out_stride) += dot_f32_avx2(buf, xk);
        k += 1;
    }
}

/// Fused multi-row × multi-token dot.
///
/// Dots `cnt` (1..=`RB`) dequantized rows held in `bufs` (`RB` rows of `cols`
/// f32) against `n` activation rows in `xs`, in a single pass over the
/// activations: each activation vector is loaded once and reused for every row
/// in the block, so the activation traffic drops from `rows` passes to
/// `rows / RB`.
///
/// The arithmetic is **identical** to calling [`dot_multi`] once per row: every
/// output keeps the same reduction order (sequential 8-lane FMA chain over the
/// row, then the scalar tail, then `hsum + tail`), and trailing tokens fall
/// back to the same [`dot_f32_avx2`] `dot_multi` uses. Only the loop nesting
/// changes. `TB` must be 4 for the results to match `dot_multi` exactly: then
/// the trailing tokens (`n % 4`) are the ones `dot_multi` also leaves to
/// `dot_f32_avx2`, so every output takes the same path.
///
/// Outputs are accumulated (`+=`), so the caller must zero `out` first (as
/// `matmul_batch` does).
///
/// # Safety
///
/// `available()` must hold. `bufs` must hold at least `RB * cols` elements,
/// `xs` at least `n * cols`, and for every `k < n` and `r < cnt` the address
/// `out + k * out_stride + r` must be valid and writable by this thread alone.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_multi_rows<const RB: usize, const TB: usize>(
    bufs: &[f32],
    cols: usize,
    xs: &[f32],
    n: usize,
    cnt: usize,
    out: *mut f32,
    out_stride: usize,
) {
    debug_assert!(cols > 0 && cols % 8 == 0);
    debug_assert!(cnt >= 1 && cnt <= RB);
    debug_assert!(bufs.len() >= RB * cols);
    let wp = bufs.as_ptr();
    let xp = xs.as_ptr();
    let mut k = 0;
    while k + TB <= n {
        let mut acc = [[_mm256_setzero_ps(); TB]; RB];
        let mut i = 0;
        while i + 8 <= cols {
            let mut xv = [_mm256_setzero_ps(); TB];
            for (t, x) in xv.iter_mut().enumerate() {
                *x = _mm256_loadu_ps(xp.add((k + t) * cols + i));
            }
            for r in 0..RB {
                let w = _mm256_loadu_ps(wp.add(r * cols + i));
                for t in 0..TB {
                    acc[r][t] = _mm256_fmadd_ps(w, xv[t], acc[r][t]);
                }
            }
            i += 8;
        }
        for r in 0..cnt {
            for t in 0..TB {
                let mut a = 0f32;
                let mut j = i;
                while j < cols {
                    a += *wp.add(r * cols + j) * *xp.add((k + t) * cols + j);
                    j += 1;
                }
                *out.add((k + t) * out_stride + r) += hsum(acc[r][t]) + a;
            }
        }
        k += TB;
    }
    while k < n {
        for r in 0..cnt {
            let v = dot_f32_avx2(
                bufs.get_unchecked(r * cols..r * cols + cols),
                xs.get_unchecked(k * cols..k * cols + cols),
            );
            *out.add(k * out_stride + r) += v;
        }
        k += 1;
    }
}

/// Integer-exact `Q4_0` × int8-activation dot (18-byte blocks).
///
/// Values are `nibble - 8`: `Σ(u-8)*a = Σu*a - 8*Σa` with the first term
/// bit-exact via `_mm_maddubs_epi16` on 16-lane halves.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q40_q8_avx2(src: &[u8], act: &Q8Row) -> f32 {
    let n32 = src.len() / 18;
    debug_assert_eq!(act.qs.len(), n32 * 32);
    let m0f = _mm_set1_epi8(0x0F);
    let ones = _mm_set1_epi16(1);
    let mut total = 0f32;
    for bi in 0..n32 {
        let base = bi * 18;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base], src[base + 1]]));
        let b = _mm_loadu_si128(src.as_ptr().add(base + 2) as *const __m128i);
        let lo = _mm_and_si128(b, m0f);
        let hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
        let a0 = _mm_loadu_si128(act.qs.as_ptr().add(bi * 32) as *const __m128i);
        let a1 = _mm_loadu_si128(act.qs.as_ptr().add(bi * 32 + 16) as *const __m128i);
        // lo holds vals 0..16, hi holds vals 16..32 (see layout note above).
        let s0 = _mm_madd_epi16(_mm_maddubs_epi16(lo, a0), ones);
        let s1 = _mm_madd_epi16(_mm_maddubs_epi16(hi, a1), ones);
        let i0 = hsum_i32_128(s0);
        let i1 = hsum_i32_128(s1);
        total += d * (i0 + i1 - 8 * act.sums[bi]) as f32 * act.scales[bi];
    }
    total
}

/// Integer-exact `Q8_0` × int8-activation dot (34-byte blocks).
///
/// Both sides are signed bytes: widen to 16 bits and `_mm256_madd_epi16`.
/// Bit-exact until the per-block float combine.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q80_q8_avx2(src: &[u8], act: &Q8Row) -> f32 {
    let n32 = src.len() / 34;
    debug_assert_eq!(act.qs.len(), n32 * 32);
    let mut total = 0f32;
    for bi in 0..n32 {
        let base = bi * 34;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base], src[base + 1]]));
        let w = src.as_ptr().add(base + 2);
        let a = act.qs.as_ptr().add(bi * 32);
        let mut acc = _mm256_setzero_si256();
        for k in 0..2 {
            let w16 = _mm_loadu_si128(w.add(k * 16) as *const __m128i);
            let a16 = _mm_loadu_si128(a.add(k * 16) as *const __m128i);
            let w32 = _mm256_cvtepi8_epi16(w16);
            let a32 = _mm256_cvtepi8_epi16(a16);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(w32, a32));
        }
        total += d * hsum_i32(acc) as f32 * act.scales[bi];
    }
    total
}

/// Integer dot dispatcher over the supported K-quant row against [`Q8Row`].
/// `None` when AVX2 is unavailable or the type has no integer kernel.
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_qk_q8_row(ty: GgmlType, src: &[u8], act: &Q8Row) -> Option<f32> {
    if !available() {
        return None;
    }
    Some(match ty {
        GgmlType::Q4K => dot_q4k_q8_avx2(src, act),
        GgmlType::Q6K => dot_q6k_q8_avx2(src, act),
        GgmlType::Q4_0 => dot_q40_q8_avx2(src, act),
        GgmlType::Q8_0 => dot_q80_q8_avx2(src, act),
        _ => return None,
    })
}

/// AVX2 split dequantization: unpack one `Q4_K` row to `f32` (no dot).
///
/// Same unpack/scale math as the fused kernel, but stores intermediates so a
/// batched caller can dot them against N activation rows. Used by the prompt
/// prefill path, where the fused kernel's per-(row,token) re-unpack would
/// repeat the same unpack N times.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dequant_row_q4k(dst: *mut f32, src: *const u8, n: usize) {
    let m0f = _mm_set1_epi8(0x0F);
    let nblocks = n / 256;
    for bi in 0..nblocks {
        let base = bi * 144;
        let d = quant::f16_to_f32(u16::from_le_bytes([
            *src.add(base),
            *src.add(base + 1),
        ]));
        let dmin = quant::f16_to_f32(u16::from_le_bytes([
            *src.add(base + 2),
            *src.add(base + 3),
        ]));
        let scales = std::slice::from_raw_parts(src.add(base + 4), 12);
        let qs = src.add(base + 16);
        let out = dst.add(bi * 256);
        for g in 0..4usize {
            let (sc1, m1) = quant::get_scale_min_k4(2 * g, scales);
            let (sc2, m2) = quant::get_scale_min_k4(2 * g + 1, scales);
            let d1 = _mm256_set1_ps(d * sc1 as f32);
            let mm1 = _mm256_set1_ps(dmin * m1 as f32);
            let d2 = _mm256_set1_ps(d * sc2 as f32);
            let mm2 = _mm256_set1_ps(dmin * m2 as f32);
            let qsg = qs.add(g * 32);
            let olo = out.add(g * 64);
            let ohi = out.add(g * 64 + 32);
            for k in 0..4 {
                let b = _mm_loadl_epi64(qsg.add(k * 8) as *const __m128i);
                let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(b, m0f)));
                let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                    _mm_srli_epi16(b, 4),
                    m0f,
                )));
                _mm256_storeu_ps(olo.add(k * 8), sub_mul(lo, d1, mm1));
                _mm256_storeu_ps(ohi.add(k * 8), sub_mul(hi, d2, mm2));
            }
        }
    }
}

/// Public split-dequant entry: unpack a `Q4_K` row into `dst` (`dst.len()`
/// elements, multiple of 256). Falls back to scalar for other layouts.
#[cfg(target_arch = "x86_64")]
pub unsafe fn dequant_row(dst: &mut [f32], ty: GgmlType, src: &[u8]) -> bool {
    if !available() || ty != GgmlType::Q4K || dst.len() % 256 != 0 {
        return false;
    }
    dequant_row_q4k(dst.as_mut_ptr(), src.as_ptr(), dst.len());
    true
}

/// `Q6_K`: 210-byte super-blocks of 256 values (6-bit).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn q6k(src: &[u8], x: &[f32]) -> f32 {
    let m0f = _mm_set1_epi8(0x0F);
    let m03 = _mm_set1_epi8(0x03);
    let c32 = _mm256_set1_ps(32.0);
    let nblocks = x.len() / 256;
    let mut total = 0f32;
    for bi in 0..nblocks {
        let base = bi * 210;
        let d = quant::f16_to_f32(u16::from_le_bytes([src[base + 208], src[base + 209]]));
        let xp = x.as_ptr().add(bi * 256);
        let mut block_dot = 0f32;
        let mut yoff = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 128usize;
        let mut sc_off = 192usize;
        let mut n = 0usize;
        while n < 256 {
            for half in 0..2usize {
                let is = half;
                let s1 = (src[base + sc_off + is] as i8) as f32;
                let s2 = (src[base + sc_off + is + 2] as i8) as f32;
                let s3 = (src[base + sc_off + is + 4] as i8) as f32;
                let s4 = (src[base + sc_off + is + 6] as i8) as f32;
                let mut a1 = _mm256_setzero_ps();
                let mut a2 = _mm256_setzero_ps();
                let mut a3 = _mm256_setzero_ps();
                let mut a4 = _mm256_setzero_ps();
                for k in 0..2usize {
                    let l = half * 16 + k * 8;
                    let ql = src.as_ptr().add(base + ql_off + l);
                    let qh = src.as_ptr().add(base + qh_off + l);
                    let b0 = _mm_loadl_epi64(ql as *const __m128i);
                    let b1 = _mm_loadl_epi64(ql.add(32) as *const __m128i);
                    let h = _mm_loadl_epi64(qh as *const __m128i);
                    let n0 = _mm_and_si128(b0, m0f);
                    let n1 = _mm_and_si128(b1, m0f);
                    let n2 = _mm_and_si128(_mm_srli_epi16(b0, 4), m0f);
                    let n3 = _mm_and_si128(_mm_srli_epi16(b1, 4), m0f);
                    let h0 = _mm_slli_epi16(_mm_and_si128(h, m03), 4);
                    let h1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), m03), 4);
                    let h2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), m03), 4);
                    let h3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), m03), 4);
                    let f1 = _mm256_sub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_or_si128(n0, h0))),
                        c32,
                    );
                    let f2 = _mm256_sub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_or_si128(n1, h1))),
                        c32,
                    );
                    let f3 = _mm256_sub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_or_si128(n2, h2))),
                        c32,
                    );
                    let f4 = _mm256_sub_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_or_si128(n3, h3))),
                        c32,
                    );
                    a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp.add(yoff + l)), a1);
                    a2 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xp.add(yoff + l + 32)), a2);
                    a3 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xp.add(yoff + l + 64)), a3);
                    a4 = _mm256_fmadd_ps(f4, _mm256_loadu_ps(xp.add(yoff + l + 96)), a4);
                }
                block_dot += d * (s1 * hsum(a1) + s2 * hsum(a2) + s3 * hsum(a3) + s4 * hsum(a4));
            }
            yoff += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            n += 128;
        }
        total += block_dot;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequant_dot_scalar, GgmlType};

    struct Lcg(u64);
    impl Lcg {
        fn u8(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
        fn f32(&mut self) -> f32 {
            (self.u8() as f32 / 255.0) * 2.0 - 1.0
        }
    }

    fn f16_le(bits: u16) -> [u8; 2] {
        bits.to_le_bytes()
    }

    fn check(ty: GgmlType, block_bytes: usize, block_elems: usize, set_scales: impl Fn(&mut [u8])) {
        if !available() {
            return;
        }
        let mut rng = Lcg(0x1234_5678);
        let nblocks = 7;
        let mut src = vec![0u8; block_bytes * nblocks];
        for b in src.iter_mut() {
            *b = rng.u8();
        }
        for i in 0..nblocks {
            set_scales(&mut src[i * block_bytes..(i + 1) * block_bytes]);
        }
        let n = block_elems * nblocks;
        let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let s = dequant_dot_scalar(ty, &src, &x);
        let v = dequant_dot(ty, &src, &x).expect("avx2 path");
        let denom = s.abs().max(1.0);
        assert!(
            (s - v).abs() / denom < 1e-4,
            "{ty:?}: scalar={s} avx2={v}"
        );
    }

    #[test]
    fn q4k_matches_scalar() {
        check(GgmlType::Q4K, 144, 256, |b| {
            b[0..2].copy_from_slice(&f16_le(0x3800)); // d   = 0.5
            b[2..4].copy_from_slice(&f16_le(0x3400)); // min = 0.25
        });
    }

    #[test]
    fn q6k_matches_scalar() {
        check(GgmlType::Q6K, 210, 256, |b| {
            b[208..210].copy_from_slice(&f16_le(0x3800));
        });
    }

    #[test]
    fn q8_0_matches_scalar() {
        check(GgmlType::Q8_0, 34, 32, |b| {
            b[0..2].copy_from_slice(&f16_le(0x3800));
        });
    }

    #[test]
    fn q4_0_matches_scalar() {
        check(GgmlType::Q4_0, 18, 32, |b| {
            b[0..2].copy_from_slice(&f16_le(0x3800));
        });
    }

    #[test]
    fn f32_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = Lcg(99);
        let n = 40usize;
        let vals: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let mut src = Vec::with_capacity(n * 4);
        for v in &vals {
            src.extend_from_slice(&v.to_le_bytes());
        }
        let s = dequant_dot_scalar(GgmlType::F32, &src, &vals);
        let v = dequant_dot(GgmlType::F32, &src, &vals).unwrap();
        assert!((s - v).abs() < 1e-4);
    }

    #[test]
    fn dot_f32_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = Lcg(7);
        for n in [1usize, 7, 8, 9, 16, 33, 128] {
            let a: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let s: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let v = dot_f32(&a, &b).unwrap();
            assert!((s - v).abs() < 1e-4, "n={n}: {s} vs {v}");
        }
    }

    fn rmsnorm_ref(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
        let n = x.len();
        let mut ss = 0f32;
        for v in x {
            ss += v * v;
        }
        let s = 1.0 / (ss / n as f32 + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] * s * w[i];
        }
    }

    #[test]
    fn rmsnorm_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = Lcg(31);
        for n in [1usize, 7, 8, 9, 64, 128, 1024] {
            let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let w: Vec<f32> = (0..n).map(|_| rng.f32().abs() + 0.25).collect();
            let mut expect = vec![0f32; n];
            let mut got = vec![0f32; n];
            rmsnorm_ref(&mut expect, &x, &w, 1e-6);
            unsafe {
                rmsnorm(&mut got, &x, &w, 1e-6);
            }
            for i in 0..n {
                assert!(
                    (expect[i] - got[i]).abs() < 1e-5,
                    "n={n} i={i}: {} vs {}",
                    expect[i],
                    got[i]
                );
            }
            // In-place variant must agree too.
            let mut inplace = x.clone();
            unsafe {
                rmsnorm_inplace(&mut inplace, &w, 1e-6);
            }
            for i in 0..n {
                assert!((expect[i] - inplace[i]).abs() < 1e-5, "inplace n={n} i={i}");
            }
        }
    }

    #[test]
    fn q8_row_roundtrip() {
        // Quantize/dequantize invertibility + exact block sums.
        let mut rng = Lcg(55);
        let n = 256usize;
        let x: Vec<f32> = (0..n).map(|_| rng.f32() * 3.0).collect();
        let q = quantize_q8(&x);
        assert_eq!(q.scales.len(), 8);
        assert_eq!(q.qs.len(), n);
        for b in 0..8 {
            let mut s = 0i32;
            let mut s_lo = 0i32;
            let mut max_err = 0f32;
            for i in 0..32 {
                s += q.qs[b * 32 + i] as i32;
                if i < 16 {
                    s_lo += q.qs[b * 32 + i] as i32;
                }
                let back = q.qs[b * 32 + i] as f32 * q.scales[b];
                max_err = max_err.max((back - x[b * 32 + i]).abs());
            }
            assert_eq!(s, q.sums[b], "block {b} sum");
            assert_eq!(s_lo, q.sums16[b * 2], "block {b} low half sum");
            assert_eq!(s - s_lo, q.sums16[b * 2 + 1], "block {b} high half sum");
            assert!(max_err <= q.scales[b] * 0.5 + 1e-6, "block {b} err {max_err}");
        }
    }

    #[test]
    fn q4k_int_dot_matches_scalar() {
        if !available() {
            return;
        }
        // Random Q4_K row bytes; reference = scalar fused dequant-dot.
        // Tolerance is loose on purpose: int8 activation quantization itself
        // is a ~1% approximation (same one llama.cpp uses everywhere).
        let mut rng = Lcg(0x9e37_4b21);
        let nblocks = 3usize;
        let mut src = vec![0u8; 144 * nblocks];
        for b in src.iter_mut() {
            *b = rng.u8();
        }
        for i in 0..nblocks {
            src[i * 144..i * 144 + 2].copy_from_slice(&f16_le(0x3800)); // d=0.5
            src[i * 144 + 2..i * 144 + 4].copy_from_slice(&f16_le(0x3400)); // min=0.25
        }
        let n = 256 * nblocks;
        let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let act = quantize_q8(&x);
        let s = dequant_dot_scalar(GgmlType::Q4K, &src, &x);
        let v = unsafe { dot_qk_q8_row(GgmlType::Q4K, &src, &act) }.expect("avx2 path");
        let denom = s.abs().max(1.0);
        assert!(
            (s - v).abs() / denom < 0.02,
            "scalar={s} intdot={v}"
        );
    }

    #[test]
    fn q6k_int_dot_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = Lcg(0x51ab_6e09);
        let nblocks = 3usize;
        let mut src = vec![0u8; 210 * nblocks];
        for b in src.iter_mut() {
            *b = rng.u8();
        }
        for i in 0..nblocks {
            src[i * 210 + 208..i * 210 + 210].copy_from_slice(&f16_le(0x3800)); // d=0.5
        }
        let n = 256 * nblocks;
        let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let act = quantize_q8(&x);
        let s = dequant_dot_scalar(GgmlType::Q6K, &src, &x);
        let v = unsafe { dot_qk_q8_row(GgmlType::Q6K, &src, &act) }.expect("avx2 path");
        let denom = s.abs().max(1.0);
        assert!(
            (s - v).abs() / denom < 0.02,
            "scalar={s} intdot={v}"
        );
    }

    #[test]
    fn q40_int_dot_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = Lcg(0x40e0_1234);
        let nblocks = 5usize;
        let mut src = vec![0u8; 18 * nblocks];
        for b in src.iter_mut() {
            *b = rng.u8();
        }
        for i in 0..nblocks {
            src[i * 18..i * 18 + 2].copy_from_slice(&f16_le(0x3800)); // d=0.5
        }
        let n = 32 * nblocks;
        let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let act = quantize_q8(&x);
        let s = dequant_dot_scalar(GgmlType::Q4_0, &src, &x);
        let v = unsafe { dot_qk_q8_row(GgmlType::Q4_0, &src, &act) }.expect("avx2 path");
        let denom = s.abs().max(1.0);
        assert!(
            (s - v).abs() / denom < 0.02,
            "scalar={s} intdot={v}"
        );
    }

    #[test]
    fn q80_int_dot_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = Lcg(0x8080_8080);
        let nblocks = 5usize;
        let mut src = vec![0u8; 34 * nblocks];
        for b in src.iter_mut() {
            *b = rng.u8();
        }
        for i in 0..nblocks {
            src[i * 34..i * 34 + 2].copy_from_slice(&f16_le(0x3800)); // d=0.5
        }
        let n = 32 * nblocks;
        let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
        let act = quantize_q8(&x);
        let s = dequant_dot_scalar(GgmlType::Q8_0, &src, &x);
        let v = unsafe { dot_qk_q8_row(GgmlType::Q8_0, &src, &act) }.expect("avx2 path");
        let denom = s.abs().max(1.0);
        assert!(
            (s - v).abs() / denom < 0.02,
            "scalar={s} intdot={v}"
        );
    }

    #[test]
    fn split_dequant_q4k_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = Lcg(0xd3c0_17);
        let n = 512usize;
        let mut src = vec![0u8; 144 * (n / 256)];
        for b in src.iter_mut() {
            *b = rng.u8();
        }
        for i in 0..n / 256 {
            src[i * 144..i * 144 + 2].copy_from_slice(&f16_le(0x3800));
            src[i * 144 + 2..i * 144 + 4].copy_from_slice(&f16_le(0x3400));
        }
        let mut got = vec![0f32; n];
        let ok = unsafe { dequant_row(&mut got, GgmlType::Q4K, &src) };
        assert!(ok);
        let mut expect = vec![0f32; n];
        crate::quant::dequant_row(crate::quant::GgmlType::Q4K, &src, &mut expect);
        for i in 0..n {
            let denom = expect[i].abs().max(1.0);
            assert!((got[i] - expect[i]).abs() / denom < 1e-5, "i={i}");
        }
        // Non-Q4K layouts and bad lengths fall back (return false).
        let mut tmp = vec![0f32; 32];
        assert!(!unsafe { dequant_row(&mut tmp, GgmlType::Q8_0, &src[..34]) });
        assert!(!unsafe { dequant_row(&mut tmp, GgmlType::Q4K, &src[..100]) });
    }

    #[test]
    fn saxpy_is_bit_identical() {
        if !available() {
            return;
        }
        let mut rng = Lcg(0x5a07_9911);
        for n in [1usize, 7, 8, 9, 64, 128, 1024] {
            let mut dst: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let x: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let mut expect = dst.clone();
            for i in 0..n {
                expect[i] += 0.37 * x[i];
            }
            unsafe {
                saxpy(&mut dst, 0.37, &x);
            }
            for i in 0..n {
                assert_eq!(
                    dst[i].to_bits(),
                    expect[i].to_bits(),
                    "n={n} i={i}: {} vs {}",
                    dst[i],
                    expect[i]
                );
            }
        }
    }

    #[test]
    fn dot_multi_wide_matches_sequential() {
        // n = 13 exercises the 4-wide vector path (k = 0,4,8) plus the scalar
        // tail (k = 12) — the exact shape production prefill uses.
        if !available() {
            return;
        }
        let mut rng = Lcg(0xd07c_417e);
        let cols = 1024usize;
        let n = 13usize;
        let buf: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
        let xs: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
        let mut got = vec![0f32; n];
        unsafe {
            dot_multi(&buf, &xs, 0, cols, n, got.as_mut_ptr(), 1);
        }
        for k in 0..n {
            let s: f32 = buf.iter().zip(&xs[k * cols..k * cols + cols]).map(|(a, b)| a * b).sum();
            let denom = s.abs().max(1.0);
            assert!((got[k] - s).abs() / denom < 1e-4, "k={k}: {} vs {s}", got[k]);
        }
    }

    #[test]
    fn dot_multi_rows_matches_serial_bits() {
        // The row-blocked kernel must be *bit*-identical to one-row-at-a-time
        // `dot_multi`: blocking only reorders loops, never the reduction order
        // of any single output. Covers the token tail (n % TB), a partial row
        // block (rows % RB), and the 8-wide/scalar tails of `cols`.
        if !available() {
            return;
        }
        // The engine's shipped split. `TB` must be 4 for exactness (see the
        // kernel docs); the row/token tails are exercised below.
        const RB: usize = 3;
        const TB: usize = 4;
        let mut rng = Lcg(0x5eed_1234);
        for cols in [1024usize, 3072] {
            for n in [1usize, 2, 3, 5, 13, 32] {
                for rows in [1usize, 3, 4, 5, 9, 11] {
                    let xs: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
                    let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
                    let mut want = vec![0f32; n * rows];
                    let mut got = vec![0f32; n * rows];
                    unsafe {
                        for r in 0..rows {
                            dot_multi(
                                &w[r * cols..(r + 1) * cols],
                                &xs,
                                0,
                                cols,
                                n,
                                want.as_mut_ptr().add(r),
                                rows,
                            );
                        }
                        let mut bufs = vec![0f32; RB * cols];
                        let mut r = 0;
                        while r < rows {
                            let cnt = (rows - r).min(RB);
                            bufs[..cnt * cols].copy_from_slice(&w[r * cols..(r + cnt) * cols]);
                            for j in cnt..RB {
                                bufs[j * cols..(j + 1) * cols].fill(0.0);
                            }
                            dot_multi_rows::<RB, TB>(
                                &bufs,
                                cols,
                                &xs,
                                n,
                                cnt,
                                got.as_mut_ptr().add(r),
                                rows,
                            );
                            r += cnt;
                        }
                    }
                    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "cols={cols} n={n} rows={rows} idx={i}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }
}
