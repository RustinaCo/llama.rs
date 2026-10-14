# rust-llama 🦀⚡

**A featherweight, CPU-only LLM inference engine in pure Rust — ~1 MB static
binary, 2 dependencies, zero C toolchain.**

`rust-llama` runs modern quantized GGUF models (Qwen3 family and friends) on
ordinary CPUs — Linux, Windows and macOS — with competitive speed, a tiny
footprint, and readable code (~6,000 lines) you can actually hack on. It also ships engine-side
tooling: vocabulary pruning, an OpenAI-compatible HTTP server, and
benchmarks.

> **Scope today:** lightweight systems — **Linux, Windows and macOS on CPU**
> (x86-64 AVX2 fast path, portable scalar fallback elsewhere).
> GPU, ARM and other architectures are on the roadmap — contributions welcome!

---

## Why rust-llama?

| | rust-llama | llama.cpp |
|---|---|---|
| Binary size | **~1 MB, single file** | ~9.4 MB (binary + 6 shared libs) + full C++ built at compile time |
| Dependencies | **2 crates** (`memmap2`, `rayon`) | C/C++ toolchain, libclang, CMake, tokio stack… |
| Generation speed (Qwen3-0.6B-Q4_K_M, i5-10500, 12 threads) | **~21–32 tok/s (≈70–100%)** | ~22–43 tok/s |
| Prompt processing | ~85–120 tok/s (≈45–90%) | ~120–300 tok/s |
| Peak RAM, same workload | **~450 MB** | ~4.4 GB |
| Architectures | Qwen3 / Llama family (focused) | dozens (general) |
| GPU | CPU-only by design | yes |
| Extras | `prune-vocab`, std-only OpenAI server, `bench`/`profile` | full ecosystem |

*Measured same-session, same model/prompt/threads; absolute numbers vary with
machine state — the ratios above held across repeated sessions.*

### Per-quantization throughput (Qwen3-0.6B, i5-10500, 12 threads, `-n 48`)

| Weights | File | Prompt | Generation |
|---|---|---|---|
| Q4_K_M | 462 MB | 124.5 tok/s | 28.3 tok/s |
| Q8_0 | 610 MB | 90.3 tok/s | 20.5 tok/s |
| F32 | 2279 MB | 46.3 tok/s | 6.7 tok/s |
| Q4_K_M pruned fa/en (109k vocab, tied) | 342 MB | 122.7 tok/s | **31.0 tok/s** |

Smaller quants win twice: less memory traffic per token, and (pruned) fewer
logits. Differential-tested against llama.cpp: byte-identical greedy output
on factual prompts (`tools/logits_diff.py` in the sister repo).

### Correctness, not just speed

Throughput means nothing if the math is wrong. Every hot kernel is tested
against a scalar reference, and the whole engine is differential-tested
against llama.cpp on identical files: **byte-identical greedy completions on
2/4 prompts** (world capitals), fluent-and-valid on the other two where
creative text legitimately diverges after the same prefix. 26 unit tests, all
green, run with `cargo test --release`.

Robustness is tested too: malformed GGUF files fail loudly with an error —
never a panic, hang or abort (the test suite caught and fixed a real integer
overflow in the length parser). Unknown architectures are refused at load
instead of silently generating garbage.

**In short:** for a light, offline CPU product, rust-llama gives you
near-parity speed at a tenth of the weight — and code you can read
in an afternoon.

**Built for machines without a GPU.** No accelerator, no 16 GB VRAM, no
datacenter needed: a ~1 MB binary, ~450 MB peak RAM for a 0.6B model, and
full speed on an ordinary office CPU. Old laptops, office PCs and minimal
servers are first-class targets — not afterthoughts.

## Features

- **GGUF v2/v3 reader** — memory-mapped, lazy, zero-copy.
- **Quantizations** — `F32/F16/BF16`, `Q4_0/Q4_1/Q5_0/Q5_1`, `Q8_0`, `Q4_K/Q5_K/Q6_K`.
- **Modern transformer** — GQA, NeoX RoPE (precomputed), SwiGLU, per-head
  QK-norm, RMSNorm, tied embeddings, KV cache.
- **Hand-written AVX2+FMA kernels** — fused dequant-dot, integer Q×Q8 dots,
  RMSNorm, batched prefill matmuls — runtime-dispatched with portable scalar
  fallback (`RUST_LLAMA_NO_SIMD=1`). 26 unit tests, all green.
- **Byte-level BPE** straight from the GGUF file (no network, no downloads).
- **Sampling** — temperature, top-k, top-p, repetition penalty.
- **`prune-vocab`** — drop unneeded scripts and shrink embedding tables:
  vocab 151,936 → 109,051, file −82 MB, *byte-identical greedy output*.
- **`serve`** — OpenAI-compatible HTTP API (`/health`, `/v1/chat/completions`,
  `/v1/completions`, SSE streaming), standard library only.
- **`bench` / `profile` / `info`** — throughput, per-matmul breakdown, inspection.

## Model support philosophy: narrow and verified

A general engine that runs everything usually verifies nothing. This one does
the opposite: **only architectures proven end to end are allowed to load**
(anything else fails loudly at startup — never silent garbage), and each
supported family gets its own forward path plus reference tests.

- **Supported today:** Qwen3 dense (0.6B–32B, same code path, no changes).
- **Quantizations:** `F32/F16/BF16`, `Q4_0/Q4_1/Q5_0/Q5_1`, `Q8_0`,
  `Q4_K/Q5_K/Q6_K` (integer fast paths for the hot ones).
- **Not supported:** Mistral (sliding-window attention), Gemma/Phi (different
  norms/tokenizers), MoE and MLA families (routing), IQ/Q2_K/Q3_K quants,
  multimodal and embedding models.
- **More families are added by user demand**, one at a time, each with
  differential tests — open an issue with the exact model file to request one.

## Quickstart

Prebuilt binaries for **Linux (x86-64), Windows (x86-64) and macOS (ARM64)**
are attached to every [GitHub Release](https://github.com/Rustinaz/llama.rs/releases)
— rebuilt by CI on every update, only the newest version is kept. Or build it
yourself (Rust stable only):

```bash
# Need: Rust stable. No C compiler, no CMake, no downloads.
cargo build --release
# Deployment build for a fixed x86-64 machine (hot scalar code goes native;
# AVX2 kernels are runtime-dispatched either way):
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Chat with a model:
./target/release/rust-llama run Qwen3-0.6B-Q4_K_M.gguf \
    -p "The capital of France is" --temp 0 -t 12

# Interactive chat, benchmark, inspect, prune, serve:
./target/release/rust-llama run  model.gguf -i
./target/release/rust-llama bench model.gguf -n 64 -t 12
./target/release/rust-llama info  model.gguf
./target/release/rust-llama prune-vocab in.gguf --out pruned.gguf
./target/release/rust-llama serve model.gguf --port 8080
```

Env toggles: `RUST_LLAMA_NO_SIMD=1` (scalar path), `RUST_LLAMA_NO_INT=1`
(fused-f32 instead of integer dots), `RUST_LLAMA_BATCH_INT=1` (integer prefill).

## Use it as a library

```toml
[dependencies]
rust-llama = { git = "https://github.com/Rustinaz/llama.rs" }
```

```rust
use rust_llama::{model::Model, SamplingParams};

let model = Model::load("model.gguf")?;
let mut cache = model.new_cache(2048);
let ids = model.tok.encode("Hello!");
let mut logits = vec![0.0; model.cfg.vocab_size];
model.forward_into(&ids, 0, &mut cache, &mut logits);
```

## Models and training (sister project)

Fine-tunes, datasets, training/eval tooling and ready-to-run model files live
in the companion repo **[`Rustinaz/llama-fa`](https://github.com/Rustinaz/llama-fa)**
— this repo stays engine-only on purpose (nothing extra).

## Roadmap

- [x] Core engine, SIMD, prefill batching, prune-vocab, server, SFT tooling
- [ ] More architectures (Mistral/Gemma/Phi) and quant int-paths
- [ ] Fused attention, f16 KV cache, ARM NEON port
- [ ] GPU backend (much later — CPU-first is the point)
- [ ] Larger-model distillation recipes

## Contributing

All effort is welcome — kernels, architectures, docs, evals, tooling.
The codebase is small on purpose: `src/` is ~6k lines, every hot kernel has a
unit test against a scalar reference. Open an issue or PR on GitHub.

## License

MIT — see [LICENSE](LICENSE). Build products, ship them, make them better.

## Coming soon

A lightweight Persian–English language model, tuned for this engine, will be
released separately — watch the [Releases](https://github.com/Rustinaz/llama.rs/releases)
page.

## Author

**Sina Khanzadeh**
- Discord / Telegram / Instagram: **[@PROGRAMMER_1998](https://github.com/PROGRAMMER_1998)**
- Email: khanzadeh.1377@gmail.com
- Web: [sina-khanzadeh.ir](https://sina-khanzadeh.ir)
