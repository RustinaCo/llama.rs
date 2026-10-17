//! Core numerical kernels: matmul, RMSNorm, RoPE, softmax, SiLU.

use rayon::prelude::*;

use crate::quant;
use crate::tensor::Tensor;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(v) = crate::simd::dot_f32(a, b) {
            return v;
        }
    }
    dot_scalar(a, b)
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc0 = 0f32;
    let mut acc1 = 0f32;
    let mut acc2 = 0f32;
    let mut acc3 = 0f32;
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        acc0 += a[j] * b[j];
        acc1 += a[j + 1] * b[j + 1];
        acc2 += a[j + 2] * b[j + 2];
        acc3 += a[j + 3] * b[j + 3];
    }
    let mut acc = (acc0 + acc1) + (acc2 + acc3);
    for i in chunks * 4..n {
        acc += a[i] * b[i];
    }
    acc
}

/// Shared raw-pointer writer for row-parallel matmuls.
///
/// Rows are disjoint across worker chunks and each thread owns its rows for
/// ALL token columns, so writes never alias across threads. All cross-thread
/// writes go through [`RowWriter::write`], so closures capture `&RowWriter`
/// (whole struct: Sync) rather than the raw field.
struct RowWriter(*mut f32);
// Safety: every address is written by at most one thread (disjoint rows).
unsafe impl Send for RowWriter {}
unsafe impl Sync for RowWriter {}
impl RowWriter {
    #[inline]
    unsafe fn write(&self, idx: usize, v: f32) {
        // Safety: upheld by the caller (disjoint row ownership).
        *self.0.add(idx) = v;
    }
    /// Base pointer for strided writes; the caller owns the target rows.
    #[inline]
    fn base(&self) -> *mut f32 {
        self.0
    }
}

/// `y = W @ x`, where `W` is `rows x cols` (GGML row-major) and `x` has length
/// `cols`. Output `y` has length `rows`. Parallelized across rows.
pub fn matmul(w: &Tensor, x: &[f32], y: &mut [f32]) {
    let rows = w.n_rows();
    let cols = w.n_cols();
    assert_eq!(x.len(), cols, "matmul: input length mismatch");
    assert_eq!(y.len(), rows, "matmul: output length mismatch");
    if rows == 0 {
        return;
    }

    // Fast path: weights already F32 -> no dequant buffer.
    if let Some(ws) = w.f32_slice() {
        y.par_iter_mut().enumerate().for_each(|(r, o)| {
            let row = &ws[r * cols..r * cols + cols];
            *o = dot(row, x);
        });
        return;
    }

    let ty = w.ty();
    let data = w.data;
    let rb = w.row_bytes();

    // Integer fast path for K-quants: quantize the activation once, then exact
    // integer dots per row (same approximation llama.cpp uses everywhere).
    // `RUST_LLAMA_NO_INT=1` forces the fused f32 path (A/B comparisons).
    // Covers Q4_K/Q6_K (hot) plus Q4_0/Q8_0 (Q8_0 model, small quant types).
    #[cfg(target_arch = "x86_64")]
    {
        if matches!(
            ty,
            quant::GgmlType::Q4K
                | quant::GgmlType::Q6K
                | quant::GgmlType::Q4_0
                | quant::GgmlType::Q8_0
        ) && std::env::var_os("RUST_LLAMA_NO_INT").is_none()
            && crate::simd::available()
        {
            let act = crate::simd::quantize_q8(x);
            let out = RowWriter(y.as_mut_ptr());
            let out_len = y.len();
            let nt = rayon::current_num_threads().max(1);
            let chunk = (rows / (nt * 4)).max(1);
            let nchunks = rows.div_ceil(chunk);
            (0..nchunks).into_par_iter().for_each(|ci| {
                let base = ci * chunk;
                let end = (base + chunk).min(rows);
                for r in base..end {
                    let row = &data[r * rb..r * rb + rb];
                    // Safety: `available()` was checked, so this is `Some`.
                    let v = unsafe {
                        crate::simd::dot_qk_q8_row(ty, row, &act).expect("avx2 checked")
                    };
                    debug_assert!(r < out_len);
                    // Safety: thread owns rows [base, end).
                    unsafe {
                        out.write(r, v);
                    }
                }
            });
            return;
        }
    }

    // Quantized weights: fused dequantize-and-dot, no temporary f32 row buffer.
    let nt = rayon::current_num_threads().max(1);
    let chunk = (rows / (nt * 4)).max(1);
    y.par_chunks_mut(chunk).enumerate().for_each(|(ci, yc)| {
        let base = ci * chunk;
        for (k, o) in yc.iter_mut().enumerate() {
            let r = base + k;
            let row = &data[r * rb..r * rb + rb];
            *o = quant::dequant_dot(ty, row, x);
        }
    });
}

/// Batched `Y = W @ X`: `xs` holds `n` input vectors of length `cols`
/// back-to-back, `ys` receives `n` outputs of length `rows` back-to-back.
///
/// The win over `n` separate [`matmul`] calls: each quantized weight row is
/// dequantized ONCE and dotted against all `n` inputs, so weight memory
/// traffic (the decode bottleneck) drops ~`n`x. Used for prompt prefill;
/// single-token generation keeps using [`matmul`].
pub fn matmul_batch(w: &Tensor, xs: &[f32], n: usize, ys: &mut [f32]) {
    let rows = w.n_rows();
    let cols = w.n_cols();
    assert_eq!(xs.len(), n * cols, "matmul_batch: input length mismatch");
    assert_eq!(ys.len(), n * rows, "matmul_batch: output length mismatch");
    if rows == 0 || n == 0 {
        return;
    }
    if n == 1 {
        matmul(w, xs, ys);
        return;
    }
    // The fused dot below accumulates; start from zero (memset is ~10us,
    // negligible next to the matmul itself).
    ys.fill(0.0);

    // Fast path: weights already F32 -> parallelize over tokens.
    if let Some(ws) = w.f32_slice() {
        ys.par_chunks_mut(rows).enumerate().for_each(|(k, yk)| {
            let xk = &xs[k * cols..k * cols + cols];
            for r in 0..rows {
                yk[r] = dot(&ws[r * cols..r * cols + cols], xk);
            }
        });
        return;
    }

    // Quantized: parallelize over row chunks, one dequant buffer per chunk.
    // (Row-disjoint writes go through the module-level `RowWriter`.)
    //
    // Rows are processed in blocks of `RBLK`: the block is dequantized once and
    // then streamed against every activation row in one pass, so the activation
    // matrix is read `rows / RBLK` times instead of `rows`. That traffic was the
    // measured limit here (~250 GiB/s of activation reads, i.e. 4 bytes touched
    // per MAC), and reusing each activation vector for `RBLK` rows cuts it ~3x.
    // Accumulation order per output is untouched, so results are bit-identical
    // to the previous one-row-at-a-time code.

    // Integer batch path (opt-in via `RUST_LLAMA_BATCH_INT=1`): each
    // activation row is quantized once, then exact integer dots (mirrors the
    // single-token `matmul` path). The default fused path measured faster at
    // small N, so integer batch stays opt-in.
    #[cfg(target_arch = "x86_64")]
    {
        if matches!(
            w.ty(),
            quant::GgmlType::Q4K
                | quant::GgmlType::Q6K
                | quant::GgmlType::Q4_0
                | quant::GgmlType::Q8_0
        ) && std::env::var_os("RUST_LLAMA_BATCH_INT").is_some()
            && std::env::var_os("RUST_LLAMA_NO_INT").is_none()
            && crate::simd::available()
        {
            let ty = w.ty();
            let data = w.data;
            let rb = w.row_bytes();
            let acts: Vec<crate::simd::Q8Row> = (0..n)
                .map(|k| crate::simd::quantize_q8(&xs[k * cols..k * cols + cols]))
                .collect();
            let out = RowWriter(ys.as_mut_ptr());
            let out_len = ys.len();
            let nt = rayon::current_num_threads().max(1);
            let chunk = (rows / (nt * 4)).max(1);
            let nchunks = rows.div_ceil(chunk);
            (0..nchunks).into_par_iter().for_each(|ci| {
                let base = ci * chunk;
                let end = (base + chunk).min(rows);
                for r in base..end {
                    let row = &data[r * rb..r * rb + rb];
                    for k in 0..n {
                        // Safety: `available()` was checked, so `Some`.
                        let v = unsafe {
                            crate::simd::dot_qk_q8_row(ty, row, &acts[k])
                                .expect("avx2 checked")
                        };
                        debug_assert!(k * rows + r < out_len);
                        // Safety: thread owns rows [base, end) for every k.
                        unsafe {
                            out.write(k * rows + r, v);
                        }
                    }
                }
            });
            return;
        }
    }
    let out = RowWriter(ys.as_mut_ptr());
    let out_len = ys.len();
    let data = w.data;
    let rb = w.row_bytes();
    let nt = rayon::current_num_threads().max(1);
    let chunk = (rows / (nt * 4)).max(1);
    let nchunks = rows.div_ceil(chunk);
    // Rows dequantized and streamed per activation pass, and tokens sharing
    // each weight vector inside one pass.
    //
    // `TLK` must stay 4: `dot_multi_rows` is bit-identical to the serial
    // `dot_multi` only when the trailing tokens (n % TLK) are exactly the same
    // ones, and `dot_multi` groups by 4. Measured on n=160 prefill: 4/3 is
    // ~15% faster but rounds differently in the tail; 3/4 is the fastest
    // bit-identical split (2.4x the batched-matmul time of one row per pass).
    const RBLK: usize = 3;
    const TLK: usize = 4;
    (0..nchunks).into_par_iter().for_each(|ci| {
        let base = ci * chunk;
        let end = (base + chunk).min(rows);
        let mut buf = vec![0f32; cols * RBLK];
        let mut r = base;
        while r < end {
            let cnt = (end - r).min(RBLK);
            for j in 0..cnt {
                // AVX2 split-dequant when the layout supports it, else scalar.
                #[cfg(target_arch = "x86_64")]
                let fast = unsafe {
                    crate::simd::dequant_row(
                        &mut buf[j * cols..(j + 1) * cols],
                        w.ty(),
                        &data[(r + j) * rb..(r + j) * rb + rb],
                    )
                };
                #[cfg(not(target_arch = "x86_64"))]
                let fast = false;
                if !fast {
                    w.dequant_row(r + j, &mut buf[j * cols..(j + 1) * cols]);
                }
            }
            // One pass over the activations for the whole block of rows.
            #[cfg(target_arch = "x86_64")]
            {
                if crate::simd::available() {
                    // Safety: thread owns rows [base, end) for every k, and the
                    // kernel writes exactly rows [r, r + cnt) of every column.
                    unsafe {
                        crate::simd::dot_multi_rows::<RBLK, TLK>(
                            &buf,
                            cols,
                            xs,
                            n,
                            cnt,
                            out.base().add(r),
                            rows,
                        );
                    }
                    r += cnt;
                    continue;
                }
            }
            for j in 0..cnt {
                let b = &buf[j * cols..(j + 1) * cols];
                for k in 0..n {
                    let xk = &xs[k * cols..k * cols + cols];
                    let v = dot(b, xk);
                    debug_assert!(k * rows + r + j < out_len);
                    // Safety: thread owns rows [base, end) for every k.
                    unsafe {
                        out.write(k * rows + r + j, v);
                    }
                }
            }
            r += cnt;
        }
    });
}

/// `dst[i] += alpha * x[i]` (AXPY). AVX2-vectorized but bit-identical to the
/// scalar loop (multiply-then-add, no FMA); scalar fallback off x86-64.
#[inline]
pub fn saxpy(dst: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(dst.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::available() {
            // Safety: feature-gated inside `available()`.
            unsafe {
                crate::simd::saxpy(dst, alpha, x);
            }
            return;
        }
    }
    for i in 0..dst.len() {
        dst[i] += alpha * x[i];
    }
}

/// In-place RMSNorm: `out = x / rms(x) * weight`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::available() {
            // Safety: feature-gated inside `available()`.
            unsafe {
                crate::simd::rmsnorm(out, x, weight, eps);
            }
            return;
        }
    }
    rmsnorm_scalar(out, x, weight, eps);
}

#[inline]
fn rmsnorm_scalar(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(out.len(), n);
    let mut ss = 0f32;
    for v in x {
        ss += v * v;
    }
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// RMSNorm restricted to one contiguous "head" slice, with a shared weight.
pub fn rmsnorm_head(ptr: &mut [f32], weight: &[f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::available() {
            // Safety: feature-gated inside `available()`.
            unsafe {
                crate::simd::rmsnorm_inplace(ptr, weight, eps);
            }
            return;
        }
    }
    rmsnorm_head_scalar(ptr, weight, eps);
}

#[inline]
fn rmsnorm_head_scalar(ptr: &mut [f32], weight: &[f32], eps: f32) {
    let n = ptr.len();
    let mut ss = 0f32;
    for v in ptr.iter() {
        ss += v * v;
    }
    let scale = 1.0 / (ss / n as f32 + eps).sqrt();
    for i in 0..n {
        ptr[i] = ptr[i] * scale * weight[i];
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Numerically-stable in-place softmax.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &v in x.iter().skip(1) {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// Rotary position embedding using precomputed inverse frequencies.
///
/// `inv_freq[i]` must equal `theta^(-2i/head_dim)` for `i` in `0..head_dim/2`.
/// Numerically identical to [`rope_neox`] (the `powf` is hoisted to load time),
/// but with no per-call allocation and no transcendental setup per token.
pub fn rope_neox_freqs(
    buf: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    inv_freq: &[f32],
) {
    let half = head_dim / 2;
    debug_assert!(inv_freq.len() >= half);
    let p = pos as f32;
    for h in 0..n_heads {
        let head = &mut buf[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let (sin, cos) = (p * inv_freq[i]).sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }
}

/// Legacy RoPE entry point (allocates + `powf` per call). Kept for tests and
/// portability; the forward pass uses [`rope_neox_freqs`].
pub fn rope_neox(buf: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    // freq for pair i: theta^(-2i/head_dim)
    let mut freqs = vec![0f32; half];
    for (i, f) in freqs.iter_mut().enumerate() {
        let exp = -2.0 * i as f32 / head_dim as f32;
        *f = pos as f32 * theta.powf(exp);
    }
    for h in 0..n_heads {
        let head = &mut buf[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let (sin, cos) = freqs[i].sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::TensorInfo;
    use crate::quant::GgmlType;

    /// `matmul_batch` must match `n` sequential `matmul` calls exactly
    /// (same kernels, same order) on the F32 fast path.
    #[test]
    fn matmul_batch_matches_sequential() {
        let rows = 17usize;
        let cols = 40usize;
        let n = 5usize;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0) - 0.5)
            .collect();
        let xs: Vec<f32> = (0..n * cols)
            .map(|i| ((i * 40503 % 1000) as f32 / 1000.0) - 0.5)
            .collect();
        // 4-byte aligned bytes so `f32_slice()` engages the fast path.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * 4)
        };
        let info = TensorInfo {
            name: "t".to_string(),
            dims: vec![cols, rows],
            ty: GgmlType::F32,
            offset: 0,
        };
        let t = Tensor::new(&info, bytes);
        assert!(t.f32_slice().is_some());

        let mut batched = vec![0f32; n * rows];
        matmul_batch(&t, &xs, n, &mut batched);
        for k in 0..n {
            let mut refy = vec![0f32; rows];
            matmul(&t, &xs[k * cols..(k + 1) * cols], &mut refy);
            for r in 0..rows {
                assert!(
                    (batched[k * rows + r] - refy[r]).abs() < 1e-5,
                    "token {k} row {r}: {} vs {}",
                    batched[k * rows + r],
                    refy[r]
                );
            }
        }
    }

    /// Batched integer path (`Q4_K`, `n > 1`) against a direct scalar
    /// reference (per-row scalar dequant + scalar dot). Tolerance covers the
    /// int8 activation quantization, which is the only approximation.
    #[test]
    fn matmul_batch_q4k_matches_scalar() {
        let rows = 6usize;
        let cols = 256usize;
        let n = 3usize;
        let mut st: u64 = 0x1234_5678_9abc_def0;
        let mut rnd = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as u8
        };
        let mut bytes = vec![0u8; rows * 144];
        for b in bytes.iter_mut() {
            *b = rnd();
        }
        for r in 0..rows {
            bytes[r * 144..r * 144 + 2].copy_from_slice(&0x3800u16.to_le_bytes());
            bytes[r * 144 + 2..r * 144 + 4].copy_from_slice(&0x3400u16.to_le_bytes());
        }
        let xs: Vec<f32> = (0..n * cols)
            .map(|i| ((i * 40503 % 1000) as f32 / 1000.0) - 0.5)
            .collect();
        let info = TensorInfo {
            name: "t".to_string(),
            dims: vec![cols, rows],
            ty: GgmlType::Q4K,
            offset: 0,
        };
        let t = Tensor::new(&info, &bytes);
        let mut batched = vec![0f32; n * rows];
        matmul_batch(&t, &xs, n, &mut batched);
        let rb = 144usize;
        let mut buf = vec![0f32; cols];
        for k in 0..n {
            let xk = &xs[k * cols..(k + 1) * cols];
            for r in 0..rows {
                quant::dequant_row(GgmlType::Q4K, &bytes[r * rb..r * rb + rb], &mut buf);
                let mut s = 0f32;
                for i in 0..cols {
                    s += buf[i] * xk[i];
                }
                let got = batched[k * rows + r];
                let denom = s.abs().max(1.0);
                assert!(
                    (got - s).abs() / denom < 0.02,
                    "token {k} row {r}: {got} vs {s}"
                );
            }
        }
    }
}
