//! Model loading and the transformer forward pass (Qwen3 / Llama family).

use std::io;
use std::path::Path;

use rayon::prelude::*;

use crate::gguf::{GgufFile, TensorInfo};
use crate::ops;
use crate::sampler::{Sampler, SamplingParams};
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub has_qk_norm: bool,
    pub has_output: bool,
}

/// A tensor handle resolved once at load time.
///
/// The old hot path rebuilt tensor names with `format!` and did a hash-map
/// lookup for every tensor, every layer, every token. Holding the owned
/// [`TensorInfo`] instead removes all of that.
struct Weight {
    info: TensorInfo,
}

impl Weight {
    fn new(gguf: &GgufFile, name: &str) -> io::Result<Self> {
        let info = gguf
            .info(name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("model is missing required tensor '{name}'"),
                )
            })?
            .clone();
        Ok(Weight { info })
    }
}

/// Per-layer weights plus small norm vectors pre-dequantized to `f32`.
struct LayerW {
    attn_norm: Vec<f32>,
    attn_q: Weight,
    attn_k: Weight,
    attn_v: Weight,
    attn_q_bias: Option<Vec<f32>>,
    attn_k_bias: Option<Vec<f32>>,
    attn_v_bias: Option<Vec<f32>>,
    attn_q_norm: Option<Vec<f32>>,
    attn_k_norm: Option<Vec<f32>>,
    attn_output: Weight,
    ffn_norm: Vec<f32>,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}

/// Reusable scratch buffers for one forward pass.
///
/// Allocating these per token produced hundreds of small heap allocations per
/// token; they now live for the lifetime of the [`KvCache`] and are reused.
struct Workspace {
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
}

impl Workspace {
    fn new(cfg: &Config, max_ctx: usize) -> Self {
        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        Workspace {
            x: vec![0f32; cfg.n_embd],
            xb: vec![0f32; cfg.n_embd],
            q: vec![0f32; q_dim],
            k: vec![0f32; kv_dim],
            v: vec![0f32; kv_dim],
            attn: vec![0f32; q_dim],
            gate: vec![0f32; cfg.n_ff],
            up: vec![0f32; cfg.n_ff],
            scores: vec![0f32; max_ctx],
        }
    }
}

pub struct KvCache {
    pub max_ctx: usize,
    pub len: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    ws: Workspace,
}

impl KvCache {
    pub fn new(cfg: &Config, max_ctx: usize) -> Self {
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        KvCache {
            max_ctx,
            len: 0,
            k: (0..cfg.n_layer)
                .map(|_| vec![0f32; max_ctx * kv_dim])
                .collect(),
            v: (0..cfg.n_layer)
                .map(|_| vec![0f32; max_ctx * kv_dim])
                .collect(),
            ws: Workspace::new(cfg, max_ctx),
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

pub struct Model {
    pub gguf: GgufFile,
    pub cfg: Config,
    pub tok: Tokenizer,
    layers: Vec<LayerW>,
    token_embd: Weight,
    output_norm: Vec<f32>,
    output: Option<Weight>,
    /// `theta^(-2i/head_dim)` for `i` in `0..head_dim/2`, computed once at
    /// load so the forward pass pays no `powf` and no allocation per token.
    rope_inv_freq: Vec<f32>,
}

impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let gguf = GgufFile::open(path)?;
        let cfg = Self::read_config(&gguf)?;
        let tok = Self::read_tokenizer(&gguf, cfg.vocab_size)?;

        // Pre-dequantize a small (norm) tensor to f32.
        let read_f32 = |name: &str| -> io::Result<Vec<f32>> {
            let w = Weight::new(&gguf, name)?;
            let data = gguf
                .tensor_bytes_info(&w.info)
                .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, name.to_string()))?;
            Ok(Tensor::new(&w.info, data).to_f32())
        };

        let token_embd = Weight::new(&gguf, "token_embd.weight")?;
        let output_norm = read_f32("output_norm.weight")?;
        let output = if gguf.info("output.weight").is_some() {
            Some(Weight::new(&gguf, "output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = format!("blk.{l}.");
            let attn_norm = read_f32(&format!("{p}attn_norm.weight"))?;
            let ffn_norm = read_f32(&format!("{p}ffn_norm.weight"))?;
            let attn_q_norm = if cfg.has_qk_norm {
                Some(read_f32(&format!("{p}attn_q_norm.weight"))?)
            } else {
                None
            };
            let attn_k_norm = if cfg.has_qk_norm {
                Some(read_f32(&format!("{p}attn_k_norm.weight"))?)
            } else {
                None
            };
            let attn_q_bias = if gguf.info(&format!("{p}attn_q.bias")).is_some() {
                Some(read_f32(&format!("{p}attn_q.bias"))?)
            } else {
                None
            };
            let attn_k_bias = if gguf.info(&format!("{p}attn_k.bias")).is_some() {
                Some(read_f32(&format!("{p}attn_k.bias"))?)
            } else {
                None
            };
            let attn_v_bias = if gguf.info(&format!("{p}attn_v.bias")).is_some() {
                Some(read_f32(&format!("{p}attn_v.bias"))?)
            } else {
                None
            };
            layers.push(LayerW {
                attn_norm,
                attn_q: Weight::new(&gguf, &format!("{p}attn_q.weight"))?,
                attn_k: Weight::new(&gguf, &format!("{p}attn_k.weight"))?,
                attn_v: Weight::new(&gguf, &format!("{p}attn_v.weight"))?,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_q_norm,
                attn_k_norm,
                attn_output: Weight::new(&gguf, &format!("{p}attn_output.weight"))?,
                ffn_norm,
                ffn_gate: Weight::new(&gguf, &format!("{p}ffn_gate.weight"))?,
                ffn_up: Weight::new(&gguf, &format!("{p}ffn_up.weight"))?,
                ffn_down: Weight::new(&gguf, &format!("{p}ffn_down.weight"))?,
            });
        }

        // Hoist the RoPE `powf` setup out of the per-token hot path.
        let half = cfg.head_dim / 2;
        let rope_inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exp = -2.0 * i as f32 / cfg.head_dim as f32;
                cfg.rope_theta.powf(exp)
            })
            .collect();

        Ok(Model {
            gguf,
            cfg,
            tok,
            layers,
            token_embd,
            output_norm,
            output,
            rope_inv_freq,
        })
    }

    fn read_config(gguf: &GgufFile) -> io::Result<Config> {
        let arch = gguf
            .meta("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();
        check_arch(&arch)?;

        let getu = |k: &str| -> Option<usize> {
            gguf.meta(&format!("{arch}.{k}"))
                .or_else(|| gguf.meta(k))
                .and_then(|v| v.as_usize())
        };
        let getf = |k: &str| -> Option<f32> {
            gguf.meta(&format!("{arch}.{k}"))
                .or_else(|| gguf.meta(k))
                .and_then(|v| v.as_f32())
        };

        let n_layer = getu("block_count").ok_or_else(|| bad("block_count"))?;
        let n_embd = getu("embedding_length").ok_or_else(|| bad("embedding_length"))?;
        let n_ff = getu("feed_forward_length").ok_or_else(|| bad("feed_forward_length"))?;
        let n_head = getu("attention.head_count").ok_or_else(|| bad("attention.head_count"))?;
        let n_head_kv = getu("attention.head_count_kv").unwrap_or(n_head);
        let head_dim = getu("attention.key_length").unwrap_or(n_embd / n_head);
        let rope_theta = getf("rope.freq_base").unwrap_or(10000.0);
        let rms_eps = getf("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let n_ctx = getu("context_length").unwrap_or(4096);

        let vocab_size = gguf
            .info("token_embd.weight")
            .map(|i| i.n_rows())
            .unwrap_or(0);

        let has_qk_norm = gguf.info("blk.0.attn_q_norm.weight").is_some();
        let has_output = gguf.info("output.weight").is_some();

        Ok(Config {
            arch,
            n_layer,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_theta,
            rms_eps,
            n_ctx,
            vocab_size,
            has_qk_norm,
            has_output,
        })
    }

    fn read_tokenizer(gguf: &GgufFile, vocab_size: usize) -> io::Result<Tokenizer> {
        let tokens = gguf
            .meta("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_vec())
            .ok_or_else(|| bad("tokenizer.ggml.tokens"))?;

        let token_types: Vec<u32> = gguf
            .meta("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_u32().unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1u32; tokens.len()]);

        let merges = gguf
            .meta("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_vec())
            .unwrap_or_default();

        // The BPE engine below implements the GPT-2/Qwen2 byte-level scheme.
        // Anything else would tokenize wrong, so say so loudly (the arch gate
        // above already refuses unknown families; this catches odd files).
        if gguf.meta("tokenizer.ggml.model").and_then(|v| v.as_str()) != Some("gpt2") {
            eprintln!("warning: unverified tokenizer model; bulk BPE assumes gpt2-style");
        }

        let get_id = |k: &str, d: u32| -> u32 {
            gguf.meta(k).and_then(|v| v.as_u32()).unwrap_or(d)
        };
        let bos_id = get_id("tokenizer.ggml.bos_token_id", 1);
        let eos_id = get_id("tokenizer.ggml.eos_token_id", 2);

        let _ = vocab_size;
        Ok(Tokenizer::new(tokens, &token_types, merges, bos_id, eos_id))
    }

    /// Build a zero-lookup [`Tensor`] view over a resolved weight.
    #[inline]
    fn tensor<'a>(&'a self, w: &'a Weight) -> Tensor<'a> {
        let data = self
            .gguf
            .tensor_bytes_info(&w.info)
            .expect("validated tensor bytes");
        Tensor::new(&w.info, data)
    }

    pub fn new_cache(&self, max_ctx: usize) -> KvCache {
        let max_ctx = max_ctx.min(self.cfg.n_ctx);
        KvCache::new(&self.cfg, max_ctx)
    }

    /// Run the transformer over `tokens`, returning logits for the last token.
    pub fn forward(&self, tokens: &[u32], start_pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let mut out = vec![0f32; self.cfg.vocab_size];
        self.forward_into(tokens, start_pos, cache, &mut out);
        out
    }

    /// Like [`Model::forward`], but writes logits into a caller-owned buffer so
    /// the generation loop does not allocate a vocab-sized vector per token.
    pub fn forward_into(
        &self,
        tokens: &[u32],
        start_pos: usize,
        cache: &mut KvCache,
        out: &mut [f32],
    ) {
        debug_assert!(out.len() >= self.cfg.vocab_size);
        let n = tokens.len();
        if n > 1 {
            // Prompt prefill: batch the big projection matmuls across tokens.
            self.forward_batch(tokens, start_pos, cache, out);
            return;
        }
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            if i == n - 1 {
                self.forward_one(tok, pos, cache, Some(out));
            } else {
                self.forward_one(tok, pos, cache, None);
            }
        }
    }

    /// Batched prompt prefill over `tokens` (length > 1), logits for the last
    /// token only.
    ///
    /// Per-token op order is identical to [`Model::forward_one`] (same kernels,
    /// same accumulation order), so greedy outputs match bit-for-bit; only the
    /// seven projection matmuls per layer are shared across tokens via
    /// [`ops::matmul_batch`], cutting weight memory traffic ~`n`x.
    fn forward_batch(
        &self,
        tokens: &[u32],
        start_pos: usize,
        cache: &mut KvCache,
        out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let n = tokens.len().min(cache.max_ctx.saturating_sub(start_pos));
        if n == 0 {
            return;
        }
        let tokens = &tokens[..n];
        let n_embd = cfg.n_embd;
        let hd = cfg.head_dim;
        let n_head = cfg.n_head;
        let n_kv = cfg.n_head_kv;
        let n_ff = cfg.n_ff;
        let kv_dim = n_kv * hd;
        let q_dim = n_head * hd;
        let eps = cfg.rms_eps;

        let KvCache {
            k: kcache,
            v: vcache,
            len,
            ..
        } = cache;

        // Embed all prompt tokens.
        let embd = self.tensor(&self.token_embd);
        let mut xs = vec![0f32; n * n_embd];
        for (i, &tok) in tokens.iter().enumerate() {
            embd.dequant_row(tok as usize, &mut xs[i * n_embd..(i + 1) * n_embd]);
        }

        let mut xb_all = vec![0f32; n * n_embd];
        let mut q_all = vec![0f32; n * q_dim];
        let mut k_all = vec![0f32; n * kv_dim];
        let mut v_all = vec![0f32; n * kv_dim];
        let mut attn_all = vec![0f32; n * q_dim];
        let mut gate_all = vec![0f32; n * n_ff];
        let mut up_all = vec![0f32; n * n_ff];
        let scale = 1.0 / (hd as f32).sqrt();
        let group = (n_head / n_kv).max(1);

        for (l, lw) in self.layers.iter().enumerate() {
            // --- self-attention ---
            for i in 0..n {
                ops::rmsnorm(
                    &mut xb_all[i * n_embd..(i + 1) * n_embd],
                    &xs[i * n_embd..(i + 1) * n_embd],
                    &lw.attn_norm,
                    eps,
                );
            }
            ops::matmul_batch(&self.tensor(&lw.attn_q), &xb_all, n, &mut q_all);
            ops::matmul_batch(&self.tensor(&lw.attn_k), &xb_all, n, &mut k_all);
            ops::matmul_batch(&self.tensor(&lw.attn_v), &xb_all, n, &mut v_all);

            if let Some(qb) = &lw.attn_q_bias {
                for i in 0..n {
                    for j in 0..q_dim {
                        q_all[i * q_dim + j] += qb[j];
                    }
                }
            }
            if let Some(kb) = &lw.attn_k_bias {
                for i in 0..n {
                    for j in 0..kv_dim {
                        k_all[i * kv_dim + j] += kb[j];
                    }
                }
            }
            if let Some(vb) = &lw.attn_v_bias {
                for i in 0..n {
                    for j in 0..kv_dim {
                        v_all[i * kv_dim + j] += vb[j];
                    }
                }
            }

            if let (Some(qn), Some(kn)) = (&lw.attn_q_norm, &lw.attn_k_norm) {
                for i in 0..n {
                    for h in 0..n_head {
                        ops::rmsnorm_head(
                            &mut q_all[i * q_dim + h * hd..i * q_dim + (h + 1) * hd],
                            qn,
                            eps,
                        );
                    }
                    for h in 0..n_kv {
                        ops::rmsnorm_head(
                            &mut k_all[i * kv_dim + h * hd..i * kv_dim + (h + 1) * hd],
                            kn,
                            eps,
                        );
                    }
                }
            }

            for i in 0..n {
                let pos = start_pos + i;
                ops::rope_neox_freqs(
                    &mut q_all[i * q_dim..(i + 1) * q_dim],
                    n_head,
                    hd,
                    pos,
                    &self.rope_inv_freq,
                );
                ops::rope_neox_freqs(
                    &mut k_all[i * kv_dim..(i + 1) * kv_dim],
                    n_kv,
                    hd,
                    pos,
                    &self.rope_inv_freq,
                );
                let o = pos * kv_dim;
                kcache[l][o..o + kv_dim]
                    .copy_from_slice(&k_all[i * kv_dim..(i + 1) * kv_dim]);
                vcache[l][o..o + kv_dim]
                    .copy_from_slice(&v_all[i * kv_dim..(i + 1) * kv_dim]);
            }

            // Attention over the prompt, one query position per task.
            //
            // Each query row is attended independently and each (head, query)
            // pair reads its own K/V slice, so the work is embarrassingly
            // parallel. It used to run on ONE thread, which made prefill
            // bandwidth-bound at single-core speed (~5 GB/s) and dominate
            // long-prompt latency (the loop is O(n^2)).
            //
            // Rows are split with `par_chunks_mut`, so every task owns a
            // disjoint slice of `attn_all`; `kcache`/`vcache` are read-only
            // here, and each task keeps a private score row. Arithmetic is
            // unchanged, so results stay bit-identical.
            let (kc, vc): (&[Vec<f32>], &[Vec<f32>]) = (&kcache[..], &vcache[..]);
            attn_all.par_chunks_mut(q_dim).enumerate().for_each_init(
                || vec![0f32; start_pos + n],
                |scores, (i, ah_row)| {
                    let t = start_pos + i + 1;
                    for h in 0..n_head {
                        let kv = h / group;
                        let qh = &q_all[i * q_dim + h * hd..i * q_dim + h * hd + hd];
                        for j in 0..t {
                            let base = j * kv_dim + kv * hd;
                            let kj = &kc[l][base..base + hd];
                            scores[j] = ops::dot(qh, kj) * scale;
                        }
                        ops::softmax_inplace(&mut scores[..t]);
                        let ah = &mut ah_row[h * hd..h * hd + hd];
                        for a in ah.iter_mut() {
                            *a = 0.0;
                        }
                        for j in 0..t {
                            let base = j * kv_dim + kv * hd;
                            let vj = &vc[l][base..base + hd];
                            ops::saxpy(ah, scores[j], vj);
                        }
                    }
                },
            );

            ops::matmul_batch(&self.tensor(&lw.attn_output), &attn_all, n, &mut xb_all);
            for i in 0..n {
                for j in 0..n_embd {
                    xs[i * n_embd + j] += xb_all[i * n_embd + j];
                }
            }

            // --- feed-forward (SwiGLU) ---
            for i in 0..n {
                ops::rmsnorm(
                    &mut xb_all[i * n_embd..(i + 1) * n_embd],
                    &xs[i * n_embd..(i + 1) * n_embd],
                    &lw.ffn_norm,
                    eps,
                );
            }
            ops::matmul_batch(&self.tensor(&lw.ffn_gate), &xb_all, n, &mut gate_all);
            ops::matmul_batch(&self.tensor(&lw.ffn_up), &xb_all, n, &mut up_all);
            for i in 0..n * n_ff {
                gate_all[i] = ops::silu(gate_all[i]) * up_all[i];
            }
            ops::matmul_batch(&self.tensor(&lw.ffn_down), &gate_all, n, &mut xb_all);
            for i in 0..n {
                for j in 0..n_embd {
                    xs[i * n_embd + j] += xb_all[i * n_embd + j];
                }
            }
        }

        *len = start_pos + n;

        // Logits for the last prompt token only.
        let mut xb = vec![0f32; n_embd];
        ops::rmsnorm(&mut xb, &xs[(n - 1) * n_embd..], &self.output_norm, eps);
        let w = self.output.as_ref().unwrap_or(&self.token_embd);
        ops::matmul(&self.tensor(w), &xb, out);
    }

    fn forward_one(
        &self,
        token: u32,
        pos: usize,
        cache: &mut KvCache,
        logits: Option<&mut [f32]>,
    ) {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd;
        let hd = cfg.head_dim;
        let n_head = cfg.n_head;
        let n_kv = cfg.n_head_kv;
        let n_ff = cfg.n_ff;
        let kv_dim = n_kv * hd;
        let eps = cfg.rms_eps;
        let t = pos + 1;

        // Split the cache into disjoint field borrows so the workspace buffers
        // and the KV storage can be used at the same time.
        let KvCache {
            k: kcache,
            v: vcache,
            ws,
            len,
            ..
        } = cache;

        // Embedding row -> x.
        self.tensor(&self.token_embd)
            .dequant_row(token as usize, &mut ws.x);

        let scale = 1.0 / (hd as f32).sqrt();
        let group = (n_head / n_kv).max(1);

        for (l, lw) in self.layers.iter().enumerate() {
            // --- self-attention ---
            ops::rmsnorm(&mut ws.xb, &ws.x, &lw.attn_norm, eps);
            ops::matmul(&self.tensor(&lw.attn_q), &ws.xb, &mut ws.q);
            ops::matmul(&self.tensor(&lw.attn_k), &ws.xb, &mut ws.k);
            ops::matmul(&self.tensor(&lw.attn_v), &ws.xb, &mut ws.v);

            if let Some(qb) = &lw.attn_q_bias {
                for (a, &b) in ws.q.iter_mut().zip(qb.iter()) {
                    *a += b;
                }
            }
            if let Some(kb) = &lw.attn_k_bias {
                for (a, &b) in ws.k.iter_mut().zip(kb.iter()) {
                    *a += b;
                }
            }
            if let Some(vb) = &lw.attn_v_bias {
                for (a, &b) in ws.v.iter_mut().zip(vb.iter()) {
                    *a += b;
                }
            }

            if let (Some(qn), Some(kn)) = (&lw.attn_q_norm, &lw.attn_k_norm) {
                for h in 0..n_head {
                    ops::rmsnorm_head(&mut ws.q[h * hd..(h + 1) * hd], qn, eps);
                }
                for h in 0..n_kv {
                    ops::rmsnorm_head(&mut ws.k[h * hd..(h + 1) * hd], kn, eps);
                }
            }

            ops::rope_neox_freqs(&mut ws.q, n_head, hd, pos, &self.rope_inv_freq);
            ops::rope_neox_freqs(&mut ws.k, n_kv, hd, pos, &self.rope_inv_freq);

            let o = pos * kv_dim;
            kcache[l][o..o + kv_dim].copy_from_slice(&ws.k);
            vcache[l][o..o + kv_dim].copy_from_slice(&ws.v);

            // Attention for this token.
            //
            // Heads are independent here (each reads its own K/V slice and
            // writes its own `attn` slice), so above a small work threshold we
            // spread them over the thread pool. The loop is O(context), so
            // running it on one core made every generated token after a long
            // prompt single-core bandwidth-bound (~5 GB/s).
            //
            // Arithmetic per (head, position) is unchanged, so results stay
            // bit-identical.
            if t * n_head >= 1024 {
                let qs: &[f32] = &ws.q;
                let kc: &[Vec<f32>] = &kcache[..];
                let vc: &[Vec<f32>] = &vcache[..];
                ws.attn.par_chunks_mut(hd).enumerate().for_each_init(
                    || vec![0f32; t],
                    |scores, (h, ah)| {
                        let kv = h / group;
                        let qh = &qs[h * hd..h * hd + hd];
                        for j in 0..t {
                            let base = j * kv_dim + kv * hd;
                            scores[j] = ops::dot(qh, &kc[l][base..base + hd]) * scale;
                        }
                        ops::softmax_inplace(&mut scores[..t]);
                        for a in ah.iter_mut() {
                            *a = 0.0;
                        }
                        for j in 0..t {
                            let base = j * kv_dim + kv * hd;
                            ops::saxpy(ah, scores[j], &vc[l][base..base + hd]);
                        }
                    },
                );
            } else {
                for h in 0..n_head {
                    let kv = h / group;
                    let qh = &ws.q[h * hd..(h + 1) * hd];
                    for j in 0..t {
                        let base = j * kv_dim + kv * hd;
                        let kj = &kcache[l][base..base + hd];
                        ws.scores[j] = ops::dot(qh, kj) * scale;
                    }
                    ops::softmax_inplace(&mut ws.scores[..t]);
                    let ah = &mut ws.attn[h * hd..(h + 1) * hd];
                    for a in ah.iter_mut() {
                        *a = 0.0;
                    }
                    for j in 0..t {
                        let base = j * kv_dim + kv * hd;
                        let vj = &vcache[l][base..base + hd];
                        ops::saxpy(ah, ws.scores[j], vj);
                    }
                }
            }

            ops::matmul(&self.tensor(&lw.attn_output), &ws.attn, &mut ws.xb);
            for i in 0..n_embd {
                ws.x[i] += ws.xb[i];
            }

            // --- feed-forward (SwiGLU) ---
            ops::rmsnorm(&mut ws.xb, &ws.x, &lw.ffn_norm, eps);
            ops::matmul(&self.tensor(&lw.ffn_gate), &ws.xb, &mut ws.gate);
            ops::matmul(&self.tensor(&lw.ffn_up), &ws.xb, &mut ws.up);
            for i in 0..n_ff {
                ws.gate[i] = ops::silu(ws.gate[i]) * ws.up[i];
            }
            ops::matmul(&self.tensor(&lw.ffn_down), &ws.gate, &mut ws.xb);
            for i in 0..n_embd {
                ws.x[i] += ws.xb[i];
            }
        }

        *len = t;

        if let Some(out) = logits {
            ops::rmsnorm(&mut ws.xb, &ws.x, &self.output_norm, eps);
            let w = self.output.as_ref().unwrap_or(&self.token_embd);
            ops::matmul(&self.tensor(w), &ws.xb, out);
        }
    }

    /// Greedy/–sampled generation. `on_token` returns `false` to stop.
    pub fn generate<F>(
        &self,
        prompt_ids: &[u32],
        max_new: usize,
        cache: &mut KvCache,
        params: &SamplingParams,
        mut on_token: F,
    ) -> Vec<u32>
    where
        F: FnMut(u32) -> bool,
    {
        let mut sampler = Sampler::new(params.seed);
        let mut out = Vec::new();
        cache.reset();

        let mut logits = vec![0f32; self.cfg.vocab_size];
        self.forward_into(prompt_ids, 0, cache, &mut logits);
        let mut recent: Vec<u32> = prompt_ids.to_vec();

        for _ in 0..max_new {
            let next = sampler.sample(&logits, params, &recent);
            if next == self.tok.eos_id {
                break;
            }
            if !on_token(next) {
                break;
            }
            out.push(next);
            recent.push(next);
            if cache.len >= cache.max_ctx {
                break;
            }
            self.forward_into(&[next], cache.len, cache, &mut logits);
        }
        out
    }
}

fn bad(k: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("missing metadata key: {k}"),
    )
}

/// Architectures this engine has actually verified end to end (forward pass,
/// tokenizer, sampling, differential-tested against a reference).
///
/// Anything else fails LOUDLY at load: tensor names look similar across
/// families (so a silent run would produce confident garbage), while the
/// semantics differ (sliding-window attention, MLA/MoE routing, GeGLU,
/// SPM tokenizers...). New families are added by user demand, each with its
/// own forward path and reference tests — never by relaxing this gate.
fn check_arch(arch: &str) -> io::Result<()> {
    const SUPPORTED: &[&str] = &["qwen3", "qwen2"];
    if SUPPORTED.contains(&arch) {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "unsupported architecture '{arch}': this engine verifies {:?} end to end; \
             anything else would silently generate garbage. Open an issue to \
             request a new family (each needs its own forward path + tests).",
            SUPPORTED
        ),
    ))
}

#[cfg(test)]
mod arch_tests {
    use super::check_arch;

    #[test]
    fn allowlist_accepts_qwen_families() {
        assert!(check_arch("qwen3").is_ok());
        assert!(check_arch("qwen2").is_ok());
        for other in ["llama", "mistral", "gemma", "phi3", "deepseek_v3", ""] {
            assert!(check_arch(other).is_err(), "{other} must be refused loudly");
        }
    }
}

impl Config {
    pub fn describe(&self) -> String {
        format!(
            "arch={} layers={} embd={} ff={} heads={}/{} head_dim={} rope_theta={} ctx={} vocab={} qk_norm={} tied_output={}",
            self.arch,
            self.n_layer,
            self.n_embd,
            self.n_ff,
            self.n_head,
            self.n_head_kv,
            self.head_dim,
            self.rope_theta,
            self.n_ctx,
            self.vocab_size,
            self.has_qk_norm,
            !self.has_output,
        )
    }
}
