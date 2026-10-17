//! Command-line interface for rust-llama.

use std::io::{self, Write};
use std::time::Instant;

use rust_llama::model::Model;
use rust_llama::{GgufFile, SamplingParams};
use rust_llama::tokenizer::format_qwen_chat;

struct Args {
    cmd: String,
    model: Option<String>,
    prompt: Option<String>,
    system: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    threads: usize,
    ctx: usize,
    seed: u64,
    raw: bool,
    no_think: bool,
    interactive: bool,
    show_help: bool,
    out: Option<String>,
    host: String,
    port: u16,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            cmd: String::new(),
            model: None,
            prompt: None,
            system: None,
            max_tokens: 256,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            threads: 0,
            ctx: 2048,
            seed: 0x1234_5678_9abc_def0,
            raw: false,
            no_think: false,
            interactive: false,
            show_help: false,
            out: None,
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

const HELP: &str = "\
rust-llama - lightweight CPU-only LLM inference (GGUF)

USAGE:
    rust-llama run <model.gguf> [options]
    rust-llama bench <model.gguf> [options]
    rust-llama info <model.gguf>
    rust-llama tokenize <model.gguf> -p <TEXT>   (print token ids, no chat template)
    rust-llama prune-vocab <model.gguf> --out <pruned.gguf>
    rust-llama serve <model.gguf> [--host H] [--port P] [sampling options]
    rust-llama profile <model.gguf> [-t N] [-n TOKENS]   (per-matmul prefill breakdown)

OPTIONS (run):
    -p, --prompt <TEXT>       Prompt text
    -s, --system <TEXT>       System message (chat template)
    -n, --max-tokens <N>      Max new tokens (default 256)
        --temp <F>            Temperature (default 0.7; 0 = greedy)
        --top-k <N>           Top-k (default 40; 0 = disabled)
        --top-p <F>           Nucleus sampling (default 0.95)
        --repeat-penalty <F>  Repetition penalty (default 1.1)
        --repeat-last-n <N>   Repetition-penalty window (default 64)
    -t, --threads <N>         CPU threads (default: all)
        --ctx <N>             KV-cache context size (default 2048)
        --seed <N>            RNG seed
        --raw                 Do not apply the chat template
        --no-think            Disable Qwen3 thinking preamble
    -i, --interactive         Interactive chat loop
    -h, --help                Show help
";

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        a.show_help = true;
        return a;
    }
    let mut i = 0;
    // first positional = command
    if !argv[0].starts_with('-') {
        a.cmd = argv[0].clone();
        i = 1;
    }
    while i < argv.len() {
        let arg = &argv[i];
        match arg.as_str() {
            "-h" | "--help" => a.show_help = true,
            "-p" | "--prompt" => {
                i += 1;
                a.prompt = argv.get(i).cloned();
            }
            "-s" | "--system" => {
                i += 1;
                a.system = argv.get(i).cloned();
            }
            "-n" | "--max-tokens" => {
                i += 1;
                a.max_tokens = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(a.max_tokens);
            }
            "--temp" | "--temperature" => {
                i += 1;
                a.temperature = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(a.temperature);
            }
            "--top-k" => {
                i += 1;
                a.top_k = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(a.top_k);
            }
            "--top-p" => {
                i += 1;
                a.top_p = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(a.top_p);
            }
            "--repeat-penalty" => {
                i += 1;
                a.repeat_penalty = argv
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(a.repeat_penalty);
            }
            "--repeat-last-n" => {
                i += 1;
                a.repeat_last_n = argv
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(a.repeat_last_n);
            }
            "-t" | "--threads" => {
                i += 1;
                a.threads = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--ctx" => {
                i += 1;
                a.ctx = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(a.ctx);
            }
            "--seed" => {
                i += 1;
                a.seed = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(a.seed);
            }
            "--raw" => a.raw = true,
            "--no-think" => a.no_think = true,
            "-i" | "--interactive" => a.interactive = true,
            "-o" | "--out" => {
                i += 1;
                a.out = argv.get(i).cloned();
            }
            "--host" => {
                i += 1;
                a.host = argv.get(i).cloned().unwrap_or(a.host);
            }
            "--port" => {
                i += 1;
                a.port = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(a.port);
            }
            other => {
                if a.model.is_none() && !other.starts_with('-') {
                    a.model = Some(other.to_string());
                } else {
                    eprintln!("warning: unknown argument '{other}'");
                }
            }
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();
    if args.show_help {
        print!("{HELP}");
        return;
    }
    let model_path = match &args.model {
        Some(m) => m.clone(),
        None => {
            eprintln!("error: no model path given\n\n{HELP}");
            std::process::exit(1);
        }
    };

    if args.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global();
    }

    match args.cmd.as_str() {
        "info" => cmd_info(&model_path),
        "tokenize" => cmd_tokenize(&args, &model_path),
        "run" | "" => cmd_run(&args, &model_path),
        "bench" => cmd_bench(&args, &model_path),
        "prune-vocab" => cmd_prune(&args, &model_path),
        "serve" => cmd_serve(&args, &model_path),
        "profile" => cmd_profile(&args, &model_path),
        other => {
            eprintln!("error: unknown command '{other}'\n\n{HELP}");
            std::process::exit(1);
        }
    }
}

/// Print the model's token ids for `-p TEXT` (no chat template applied).
///
/// Exists so tokenization can be diffed against a reference (HF tokenizer /
/// llama.cpp) without instrumenting the generation path.
fn cmd_tokenize(args: &Args, path: &str) {
    let model = match Model::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let text = args.prompt.clone().unwrap_or_default();
    let ids = model.tok.encode(&text);
    let joined: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    println!("{}", joined.join(" "));
}

fn cmd_prune(args: &Args, path: &str) {
    let out = match &args.out {
        Some(o) => o.clone(),
        None => {
            eprintln!("error: prune-vocab needs --out <pruned.gguf>");
            std::process::exit(1);
        }
    };
    let t0 = Instant::now();
    match rust_llama::prune::prune_vocab(path, &out) {
        Ok(s) => {
            println!("pruned vocabulary: {} -> {} tokens (dropped {})", s.old_vocab, s.new_vocab, s.dropped);
            println!("  kept: {} special/control, {} byte, {} normal", s.kept_special, s.kept_byte, s.kept_normal);
            println!("  merges: {} -> {}", s.merges_old, s.merges_new);
            println!("  embedding/output bytes saved: {:.1} MiB", s.embd_bytes_saved as f64 / 1048576.0);
            println!("  output file: {out} ({:.1} MiB) in {:.1?}", s.out_bytes as f64 / 1048576.0, t0.elapsed());
        }
        Err(e) => {
            eprintln!("error pruning model: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_serve(args: &Args, path: &str) {
    let cfg = rust_llama::server::ServerConfig::new(&args.host, args.port, args.ctx);
    let params = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        seed: args.seed,
    };
    eprintln!("serving {} on {}:{}", path, args.host, args.port);
    if let Err(e) = rust_llama::server::serve(path, &cfg, &params) {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}

/// Micro-profile prefill matmuls through the public API only (no engine
/// intrusion): times one N=13 batched matmul per distinct weight shape, then
/// compares the extrapolated sum against a measured full prefill. The gap =
/// attention + norms + rope + embedding + logits.
fn cmd_profile(args: &Args, path: &str) {
    use rust_llama::ops;
    use rust_llama::tensor::Tensor;
    let model = match Model::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error loading model: {e}");
            std::process::exit(1);
        }
    };
    // `-n` overrides the profiled batch size; without it, keep the historical
    // N=13 prefill shape that the row-per-token multipliers below assume.
    // (`max_tokens` defaults to 256 for generation, hence the comparison.)
    const DEFAULT_N: usize = 13;
    let n = if args.max_tokens == Args::default().max_tokens {
        DEFAULT_N
    } else {
        args.max_tokens.max(1)
    };
    let reps = 5usize;
    // (label, tensor name, times-per-prefill-token-layer)
    let shapes = [
        ("attn_q  (1024->2048)", "blk.0.attn_q.weight", 28.0),
        ("attn_kv (1024->1024)", "blk.0.attn_k.weight", 56.0),
        ("attn_o  (2048->1024)", "blk.0.attn_output.weight", 28.0),
        ("ffn_gu  (1024->3072)", "blk.0.ffn_gate.weight", 56.0),
        ("ffn_d   (3072->1024)", "blk.0.ffn_down.weight", 28.0),
    ];
    println!("batched matmul (n={n}, {reps} reps, {} threads):", rayon::current_num_threads());
    let mut matmul_sum = 0.0f64;
    for (label, name, count) in shapes {
        let info = model.gguf.info(name).unwrap_or_else(|| {
            eprintln!("error: missing tensor '{name}'");
            std::process::exit(1);
        });
        let data = model.gguf.tensor_bytes(name).unwrap();
        let t = Tensor::new(info, data);
        let cols = t.n_cols();
        let rows = t.n_rows();
        let xs: Vec<f32> = (0..n * cols).map(|i| ((i * 40503 % 1000) as f32 / 1000.0) - 0.5).collect();
        let mut ys = vec![0f32; n * rows];
        ops::matmul_batch(&t, &xs, n, &mut ys); // warmup
        let t0 = Instant::now();
        for _ in 0..reps {
            ops::matmul_batch(&t, &xs, n, &mut ys);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let mb = info.nbytes() as f64 / 1048576.0;
        matmul_sum += ms * count;
        println!("  {label:22} {ms:8.2} ms/call  ({mb:7.1} MiB weights)  x{count:<4} = {:.1} ms", ms * count);
    }
    // Reference: one full measured prefill of the same 13-token prompt.
    let ids = vec![model.tok.bos_id; n];
    let mut cache = model.new_cache(args.ctx.min(model.cfg.n_ctx).max(64));
    let mut logits = vec![0f32; model.cfg.vocab_size];
    model.forward_into(&ids, 0, &mut cache, &mut logits); // warmup
    cache.reset();
    let t0 = Instant::now();
    model.forward_into(&ids, 0, &mut cache, &mut logits);
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("extrapolated matmuls: {matmul_sum:.1} ms");
    println!("measured full prefill: {full_ms:.1} ms");
    println!("remainder (attention+norms+rope+embd+logits): {:.1} ms ({:.0}%)",
        (full_ms - matmul_sum).max(0.0),
        (full_ms - matmul_sum).max(0.0) / full_ms * 100.0);
}

fn cmd_info(path: &str) {
    let gguf = match GgufFile::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("GGUF version : {}", gguf.version);
    println!("tensors      : {}", gguf.tensors.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    println!("metadata:");
    for k in keys {
        if k.starts_with("tokenizer.ggml.") {
            println!("  {k:42} <skipped>");
            continue;
        }
        let v = &gguf.metadata[k];
        let s = match v {
            rust_llama::gguf::MetaValue::Str(s) => format!("{s:?}"),
            rust_llama::gguf::MetaValue::Array(a) => format!("<array len={}>", a.len()),
            other => format!("{other:?}"),
        };
        println!("  {k:42} {s}");
    }
    println!("tensors (name / type / shape):");
    for name in &gguf.tensor_order {
        let t = &gguf.tensors[name];
        println!("  {name:44} {:5} {:?}", t.ty.name(), t.dims);
    }
}

fn cmd_run(args: &Args, path: &str) {
    let t0 = Instant::now();
    let model = match Model::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error loading model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("loaded in {:.2?}", t0.elapsed());
    eprintln!("config: {}", model.cfg.describe());
    eprintln!(
        "threads: {}",
        rayon::current_num_threads()
    );

    let ctx = args.ctx.min(model.cfg.n_ctx).max(16);
    let mut cache = model.new_cache(ctx);
    let params = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        seed: args.seed,
    };

    if args.interactive {
        interactive(&model, &mut cache, &params, args);
    } else {
        let prompt = args.prompt.clone().unwrap_or_else(|| "Hello!".to_string());
        run_once(&model, &mut cache, &params, args, &prompt, true);
    }
}

fn build_prompt_ids(model: &Model, args: &Args, prompt: &str) -> Vec<u32> {
    let text = if args.raw {
        prompt.to_string()
    } else {
        format_qwen_chat(prompt, args.system.as_deref(), args.no_think)
    };
    model.tok.encode(&text)
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Benchmark prompt-eval and token-generation throughput (greedy, deterministic).
fn cmd_bench(args: &Args, path: &str) {
    let model = match Model::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error loading model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("config: {}", model.cfg.describe());
    eprintln!("threads: {}", rayon::current_num_threads());

    let ctx = args.ctx.min(model.cfg.n_ctx).max(64);
    let mut cache = model.new_cache(ctx);
    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let ids = build_prompt_ids(&model, args, &prompt);
    if ids.is_empty() {
        eprintln!("error: empty prompt after tokenization");
        return;
    }
    let mut logits = vec![0f32; model.cfg.vocab_size];

    // Warm up (page in weights, prime the allocator).
    cache.reset();
    model.forward_into(&ids, 0, &mut cache, &mut logits);

    // Prompt evaluation.
    cache.reset();
    let t0 = Instant::now();
    model.forward_into(&ids, 0, &mut cache, &mut logits);
    let pt = t0.elapsed().as_secs_f64();

    // Token generation.
    let n = args.max_tokens.max(1);
    let mut generated = 0usize;
    let mut next = argmax(&logits);
    let t1 = Instant::now();
    for _ in 0..n {
        if cache.len >= cache.max_ctx {
            break;
        }
        model.forward_into(&[next], cache.len, &mut cache, &mut logits);
        generated += 1;
        next = argmax(&logits);
        if next == model.tok.eos_id {
            break;
        }
    }
    let gt = t1.elapsed().as_secs_f64();

    let p_tps = ids.len() as f64 / pt.max(1e-9);
    let g_tps = generated as f64 / gt.max(1e-9);
    eprintln!(
        "prompt: {} tok in {:.3}s ({:.1} tok/s) | gen: {} tok in {:.3}s ({:.1} tok/s)",
        ids.len(),
        pt,
        p_tps,
        generated,
        gt,
        g_tps
    );
    println!("{p_tps:.1} {g_tps:.1}");
}

fn run_once(
    model: &Model,
    cache: &mut rust_llama::model::KvCache,
    params: &SamplingParams,
    args: &Args,
    prompt: &str,
    print_prompt: bool,
) {
    let ids = build_prompt_ids(model, args, prompt);
    if ids.is_empty() {
        eprintln!("error: empty prompt after tokenization");
        return;
    }
    eprintln!("prompt tokens: {}", ids.len());

    let t0 = Instant::now();
    let mut logits = vec![0f32; model.cfg.vocab_size];
    model.forward_into(&ids, 0, cache, &mut logits);
    let prompt_ms = t0.elapsed().as_secs_f64();

    let mut sampler = rust_llama::Sampler::new(params.seed);
    let mut recent = ids.clone();
    let mut stream = ByteStream::default();

    if print_prompt {
        print!("{prompt}");
    }
    io::stdout().flush().ok();

    let gen_start = Instant::now();
    let mut generated = 0usize;
    for _ in 0..args.max_tokens {
        let next = sampler.sample(&logits, params, &recent);
        if next == model.tok.eos_id {
            break;
        }
        stream.push(model, next);
        recent.push(next);
        generated += 1;
        if cache.len >= cache.max_ctx {
            eprintln!("\n[context limit reached]");
            break;
        }
        model.forward_into(&[next], cache.len, cache, &mut logits);
        io::stdout().flush().ok();
    }
    stream.flush();

    let gen_s = gen_start.elapsed().as_secs_f64();
    let tps = if gen_s > 0.0 {
        generated as f64 / gen_s
    } else {
        0.0
    };
    println!();
    eprintln!(
        "\n[prompt: {} tok in {:.2}s | generated: {} tok in {:.2}s | {:.2} tok/s]",
        ids.len(),
        prompt_ms,
        generated,
        gen_s,
        tps
    );
}

fn interactive(
    model: &Model,
    cache: &mut rust_llama::model::KvCache,
    params: &SamplingParams,
    args: &Args,
) {
    eprintln!("interactive mode - type your message (Ctrl-D to quit)");
    let stdin = io::stdin();
    loop {
        print!("\n> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }
        let prompt = line.trim_end().to_string();
        if prompt.is_empty() {
            continue;
        }
        run_once(model, cache, params, args, &prompt, false);
        cache.reset();
    }
}

/// Incrementally decodes token bytes, only flushing complete UTF-8.
#[derive(Default)]
struct ByteStream {
    pending: Vec<u8>,
}

impl ByteStream {
    fn push(&mut self, model: &Model, id: u32) {
        self.pending.extend(model.tok.token_bytes(id));
        self.flush_valid();
    }

    fn flush_valid(&mut self) {
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                print!("{s}");
                self.pending.clear();
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    // Safety: `valid_up_to` guarantees this prefix is valid UTF-8.
                    let s = unsafe { std::str::from_utf8_unchecked(&self.pending[..valid]) };
                    print!("{s}");
                    self.pending.drain(..valid);
                }
            }
        }
    }

    fn flush(&mut self) {
        // Flush any remaining bytes lossily.
        if !self.pending.is_empty() {
            print!("{}", String::from_utf8_lossy(&self.pending));
            self.pending.clear();
        }
    }
}
