# rust-llama 🦀⚡

**A featherweight, CPU-only LLM inference engine in pure Rust — ~1 MB static
binary, 2 dependencies, zero C toolchain.**

`rust-llama` runs modern quantized GGUF models (Qwen3 family and friends) on
ordinary CPUs — Linux, Windows and macOS — with competitive speed, a tiny
footprint, and readable code (~6,400 lines) you can actually hack on. It also ships engine-side
tooling: vocabulary pruning, an OpenAI-compatible HTTP server, and
benchmarks.

> **Scope today:** lightweight systems — **Linux, Windows and macOS on CPU**
> (x86-64 AVX2 fast path, portable scalar fallback elsewhere).
> GPU, ARM and other architectures are on the roadmap — contributions welcome!

---

## Why rust-llama?

| | rust-llama | llama.cpp |
|---|---|---|
| Binary size | **~1 MB, single file (1005 KB)** | ~9.4 MB (binary + 6 shared libs) + full C++ built at compile time |
| Dependencies | **2 crates** (`memmap2`, `rayon`) | C/C++ toolchain, libclang, CMake, tokio stack… |
| Generation speed (Qwen3-0.6B-Q4_K_M, i5-10500, 11-12 threads) | **~28–29.5 tok/s (≈90–92%)** | ~31.8–36 tok/s |
| Prompt processing | **~175–270 tok/s** | ~265–420 tok/s |
| Peak RAM, same workload | **~447 MB** | ~480–520 MB |
| Architectures | Qwen3 / Llama family (focused) | dozens (general) |
| GPU | CPU-only by design | yes |
| Extras | `prune-vocab`, std-only OpenAI server, `bench`/`profile` | full ecosystem |

*Measured same-session, same model/prompt/threads; absolute numbers vary with
machine state — the ratios above held across repeated sessions.*

### Per-quantization throughput (Qwen3-0.6B, i5-10500, `-t 12`, 48-token prompt, `-n 32`)

| Weights | File | Prompt | Generation |
|---|---|---|---|
| Q4_K_M | 462 MB | 262 tok/s | 29.3 tok/s |
| Q8_0 | 610 MB | 220 tok/s | 20.8 tok/s |

Smaller quants win twice: less memory traffic per token, and (after vocabulary
pruning) fewer logits. Differential-tested against llama.cpp: byte-identical
greedy output on factual prompts.

### Correctness, not just speed

Throughput means nothing if the math is wrong. Every hot kernel is tested
against a scalar reference, and the whole engine is differential-tested
against llama.cpp on identical files: **byte-identical greedy completions on
2/4 prompts** (world capitals), fluent-and-valid on the other two where
creative text legitimately diverges after the same prefix. 32 unit tests, all
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
- **Hand-written AVX2+FMA+F16C kernels** — hardware F16C dequantization,
  fused dequant-dot, vectorized Q8 activation quantization (`quantize_q8_avx2`),
  row-blocked batched prefill matmuls (each activation vector feeds several weight
  rows in one pass), integer Q×Q8 dots, RMSNorm — runtime-dispatched with portable
  scalar fallback (`RUST_LLAMA_NO_SIMD=1`). 32 unit tests, all green.
- **Byte-level BPE** straight from the GGUF file (no network, no downloads).
- **Sampling** — zero-allocation greedy fast path, temperature, top-k, top-p,
  repetition penalty (windowed, once-per-token, matching llama.cpp semantics).
- **`prune-vocab`** — drop unneeded scripts and shrink embedding tables:
  vocab 151,936 → 109,051, file −82 MB, *byte-identical greedy output*.
- **`serve`** — OpenAI-compatible HTTP API (`/health`, `/v1/chat/completions`,
  `/v1/completions`, SSE streaming), standard library only.
- **`bench` / `profile` / `info` / `tokenize`** — throughput, per-matmul
  breakdown, inspection, and token-id dumps for reference diffing.

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
are attached to every [GitHub Release](https://github.com/RustinaCo/llama.rs/releases)
— rebuilt by CI on every update, only the newest version is kept. Or build it
yourself (Rust stable only):

```bash
# Need: Rust stable. No C compiler, no CMake, no downloads.
cargo build --release
# Deployment build for a fixed x86-64 machine (hot scalar code goes native;
# AVX2 kernels are runtime-dispatched either way):
RUSTFLAGS="-C target-cpu=native" cargo build --release

### Official Compatible Models ⚡

`rust-llama` is fully optimized and synced for the **Rustina Modular AI Models**:
- 🇮🇷 **Rustina V1 Fa (Persian Expert):** [github.com/programmer-1998/Rustina-Fa-Model](https://github.com/programmer-1998/Rustina-Fa-Model) — Download `Rustina-V1-Fa-Q4_K_M.gguf` from Releases and run it directly.

```bash
# Chat with the official Persian expert model:
./target/release/rust-llama run Rustina-V1-Fa-Q4_K_M.gguf \
    -p "پایتخت فرانسه کجاست؟" --temp 0 -t 12

# Interactive chat (Multi-turn session memory active by default):
./target/release/rust-llama run Rustina-V1-Fa-Q4_K_M.gguf -i

# Interactive chat with persistent cross-session memory (across restarts):
./target/release/rust-llama run Rustina-V1-Fa-Q4_K_M.gguf -i --memory

# Modular Expert Routing (Multi-Model Delegation):
# Delegates to specialized models if available; otherwise gracefully alerts the user:
./target/release/rust-llama run Rustina-V1-Fa-Q4_K_M.gguf -i \
    --expert-en Rustina-V1-En-Q4_K_M.gguf \
    --expert-py Rustina-V1-Python-Q4_K_M.gguf

# Benchmark, inspect, prune, serve:
./target/release/rust-llama bench Rustina-V1-Fa-Q4_K_M.gguf -n 64 -t 12
./target/release/rust-llama info  Rustina-V1-Fa-Q4_K_M.gguf
./target/release/rust-llama prune-vocab in.gguf --out pruned.gguf
./target/release/rust-llama serve Rustina-V1-Fa-Q4_K_M.gguf --port 8080
```

### Interactive Memory & Modular Routing

- **Session Multi-turn Memory (Default in `-i`):** Remembers conversation context across turns during a session with zero allocation leaks and automatic sliding window. Supports in-chat `/clear` to start fresh.
- **Persistent Cross-Session Memory (`--memory`):** Saves user facts to a structured external profile file without touching or corrupting model weights (`.gguf`). Use `/save <fact>` during chat to remember user details across restarts.
- **Modular Expert Delegation:** Enables specialized single-domain models (e.g. Persian, English, Python) to work together. When a user prompt requires an uninstalled expert module, the engine politely guides the user to enable the respective expert package.

Env toggles: `RUST_LLAMA_NO_SIMD=1` (scalar path), `RUST_LLAMA_NO_INT=1`
(fused-f32 instead of integer dots), `RUST_LLAMA_BATCH_INT=1` (integer prefill).

## Use it as a library

```toml
[dependencies]
rust-llama = { git = "https://github.com/RustinaCo/llama.rs" }
```

```rust
use rust_llama::{model::Model, SamplingParams};

let model = Model::load("model.gguf")?;
let mut cache = model.new_cache(2048);
let ids = model.tok.encode("Hello!");
let mut logits = vec![0.0; model.cfg.vocab_size];
model.forward_into(&ids, 0, &mut cache, &mut logits);
```

## Roadmap

- [x] Core engine, SIMD, prefill batching, prune-vocab, server
- [ ] More architectures (Mistral/Gemma/Phi) and quant int-paths
- [ ] Fused attention, f16 KV cache, ARM NEON port
- [ ] GPU backend (much later — CPU-first is the point)

## Contributing

All effort is welcome — kernels, architectures, docs, evals, tooling.
The codebase is small on purpose: `src/` is ~6.4k lines, every hot kernel has a
unit test against a scalar reference. Open an issue or PR on GitHub.

## License

MIT — see [LICENSE](LICENSE). Build products, ship them, make them better.

## Author

**Sina Khanzadeh**
- Discord / Telegram / Instagram: **[@PROGRAMMER_1998](https://github.com/PROGRAMMER_1998)**
- Email: khanzadeh.1377@gmail.com
- Web: [sina-khanzadeh.ir](https://sina-khanzadeh.ir)
