//! # rust-llama
//!
//! A lightweight, dependency-light, CPU-only inference engine for LLM weights
//! stored in the GGUF format. It is written from scratch, modelled on the
//! ideas of `llama.cpp`, but with a smaller surface area and idiomatic Rust.
//!
//! Current scope (v0.1):
//!   * GGUF v2/v3 reader (memory-mapped, lazy loading)
//!   * GGML tensor dequantization: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
//!     Q8_0, Q4_K, Q5_K, Q6_K
//!   * Qwen3 / Llama-family transformer forward pass (RMSNorm, RoPE, GQA,
//!     SwiGLU, optional per-head QK-norm, tied embeddings)
//!   * Byte-level BPE tokenizer (GPT-2 / Qwen2 style) read from GGUF
//!   * Temperature / top-k / top-p / repetition sampling
//!
//! v0.2 additions:
//!   * Hand-written AVX2 + FMA fused dequantize-and-dot kernels for the hot
//!     quant types (Q4_K, Q6_K, Q8_0, Q4_0, F16, F32), selected at runtime with
//!     a portable scalar fallback.
//!   * Persistent per-sequence workspace (no per-token heap allocations) and
//!     pre-dequantized, pre-resolved weights (no name formatting / map lookups
//!     on the hot path).
//!   * Precomputed RoPE inverse frequencies (no `powf` / allocation per token).
//!   * `prune`: vocabulary pruning (`prune-vocab`) that rewrites a GGUF keeping
//!     only Latin + Arabic/Persian (+ symbols/emoji) tokens, with remapped ids,
//!     filtered merges and sliced embedding/output matrices.
//!
//! v0.3 additions:
//!   * Batched prompt prefill (`forward_batch` / `matmul_batch`): projection
//!     matmuls shared across prompt tokens, bit-identical outputs.
//!   * AVX2 RMSNorm kernels (out-of-place and in-place).
//!   * `server`: std-only OpenAI-compatible HTTP server (`serve` command,
//!     `GET /health`, `POST /v1/chat/completions`, `POST /v1/completions`,
//!     non-streaming + SSE streaming).
//!
//! The design goal is a *small, readable* engine that already runs a real
//! model end-to-end, so perf work can be layered on top later.

pub mod gguf;
pub mod model;
pub mod ops;
pub mod prune;
pub mod quant;
pub mod sampler;
pub mod server;
#[cfg(target_arch = "x86_64")]
pub mod simd;
pub mod tensor;
pub mod tokenizer;

pub use gguf::GgufFile;
pub use model::{Config, Model};
pub use sampler::{Sampler, SamplingParams};
pub use tokenizer::Tokenizer;
