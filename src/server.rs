//! Minimal OpenAI-compatible HTTP server (std only, no new dependencies).
//! Wire-up: `pub mod server;` in lib.rs, then call `serve()` from a `serve`
//! subcommand (see main.rs).
//!
//! Concurrency: the accept loop is intentionally SEQUENTIAL — one request at a
//! time. `KvCache` + its reusable `Workspace` scratch buffers are single-owner
//! `&mut` state (`forward_into` takes `&mut KvCache`); sharing them across
//! threads would need locking or per-request arenas, which buys nothing on a
//! CPU-bound GGUF engine where one forward pass already saturates all cores
//! via rayon. Sequential serving stays correct with zero locking.
//!
//! Hardening rules (every byte here is attacker-controlled):
//! * read/write timeouts so one stalled connection cannot hang the server,
//! * strict `Content-Length` handling (bad value -> 400, over cap -> 413,
//!   short body -> 400), chunked transfer rejected, header count/size caps,
//! * a real depth-and-string-aware JSON scanner (message text can never spoof
//!   top-level keys or object boundaries),
//! * validated params (finite temperature, clamped max_tokens, capped prompt)
//!   so generation always terminates and never panics the sampler,
//! * true chunked SSE streaming with per-token flush.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use crate::model::Model;
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::format_qwen_chat;

/// Server bind + context configuration.
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// KV-cache context size (clamped to `model.cfg.n_ctx`, min 16).
    pub ctx: usize,
    /// Placeholder for thread intent; rayon pool setup stays in `main.rs`.
    pub threads_note: (),
}

impl ServerConfig {
    pub fn new(host: &str, port: u16, ctx: usize) -> Self {
        ServerConfig { host: host.to_string(), port, ctx, threads_note: () }
    }
}

const READ_TIMEOUT: Duration = Duration::from_secs(15);
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_BODY: usize = 8 << 20;
const MAX_PROMPT_CHARS: usize = 32 << 10;
const MAX_TOKENS_CAP: usize = 4096;
const HEADER_LINE_MAX: usize = 8192;
const MAX_HEADERS: usize = 100;

/// Load the model ONCE, then serve forever over a sequential accept loop.
pub fn serve(model_path: &str, cfg: &ServerConfig, params: &SamplingParams) -> std::io::Result<()> {
    let model = Model::load(model_path)?;
    let mut cache = model.new_cache(cfg.ctx.min(model.cfg.n_ctx).max(16));
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr)?;
    eprintln!("rust-llama server listening on http://{addr}");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                if let Err(e) = handle_connection(s, &model, &mut cache, params) {
                    eprintln!("connection error: {e}");
                }
                cache.reset(); // no cross-request KV leakage
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    model: &Model,
    cache: &mut crate::model::KvCache,
    base: &SamplingParams,
) -> std::io::Result<()> {
    // Timeouts first: without these one slowloris stalls the whole server.
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    match read_request(&mut stream) {
        Ok((method, path, body)) => dispatch(&mut stream, &method, &path, &body, model, cache, base),
        Err((status, msg)) => {
            // Best effort: the peer may already be gone; never fail the loop.
            let _ = write_response(&mut stream, status, "application/json", &error_json(&msg));
            Ok(())
        }
    }
}

// --- dispatch + validation (pure validation, streaming writes) ---

fn dispatch(
    stream: &mut TcpStream,
    method: &str,
    path: &str,
    body: &str,
    model: &Model,
    cache: &mut crate::model::KvCache,
    base: &SamplingParams,
) -> std::io::Result<()> {
    match (method, path) {
        ("GET", "/health") => {
            write_response(stream, 200, "application/json", r#"{"status":"ok"}"#)
        }
        ("POST", "/v1/chat/completions") => match parse_chat_body(body) {
            Err(msg) => write_response(stream, 400, "application/json", &error_json(msg)),
            Ok(req) => {
                let prompt = build_chat_prompt_text(&req.messages);
                if prompt.chars().count() > MAX_PROMPT_CHARS {
                    return write_response(stream, 413, "application/json", &error_json("prompt too long"));
                }
                let params = req.effective_params(base);
                if req.stream {
                    write_sse_stream(stream, model, cache, &prompt, req.max_tokens, &params)
                } else {
                    let mut text = String::new();
                    run_generator(model, cache, &prompt, req.max_tokens, &params, &mut |s| {
                        text.push_str(s);
                    });
                    write_response(stream, 200, "application/json", &format_chat_response(&text))
                }
            }
        },
        ("POST", "/v1/completions") => match parse_completion_body(body) {
            Err(msg) => write_response(stream, 400, "application/json", &error_json(msg)),
            Ok((prompt, max_tokens, stream_out, params)) => {
                let params = merge_temp(base, params);
                if prompt.chars().count() > MAX_PROMPT_CHARS {
                    return write_response(stream, 413, "application/json", &error_json("prompt too long"));
                }
                if stream_out {
                    write_sse_stream(stream, model, cache, &prompt, max_tokens, &params)
                } else {
                    let mut text = String::new();
                    run_generator(model, cache, &prompt, max_tokens, &params, &mut |s| {
                        text.push_str(s);
                    });
                    write_response(stream, 200, "application/json", &format_completion_response(&text))
                }
            }
        },
        _ if method == "GET" || method == "POST" => {
            write_response(stream, 404, "application/json", &error_json("not found"))
        }
        _ => write_response(stream, 400, "application/json", &error_json("bad request")),
    }
}

// --- generation core: sample loop + incremental UTF-8 decode, text sink ---

/// Run the model, calling `emit` with each freshly-decoded text slice.
/// Returns the number of generated tokens. `emit` may write to a socket
/// (streaming) or accumulate into a `String`.
fn run_generator(
    model: &Model,
    cache: &mut crate::model::KvCache,
    prompt_text: &str,
    max_tokens: usize,
    params: &SamplingParams,
    emit: &mut dyn FnMut(&str),
) -> usize {
    let ids = model.tok.encode(prompt_text);
    if ids.is_empty() {
        return 0;
    }
    cache.reset();
    let mut logits = vec![0f32; model.cfg.vocab_size];
    model.forward_into(&ids, 0, cache, &mut logits);
    let mut sampler = Sampler::new(params.seed);
    let mut recent = ids;
    let mut pending: Vec<u8> = Vec::new();
    let mut n = 0usize;
    for _ in 0..max_tokens {
        let next = sampler.sample(&logits, params, &recent);
        if next == model.tok.eos_id {
            break;
        }
        recent.push(next);
        pending.extend(model.tok.token_bytes(next));
        match std::str::from_utf8(&pending) {
            Ok(s) => {
                emit(s);
                pending.clear();
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    // SAFETY: `valid_up_to` guarantees a valid UTF-8 prefix.
                    emit(unsafe { std::str::from_utf8_unchecked(&pending[..valid]) });
                    pending.drain(..valid);
                }
            }
        }
        n += 1;
        if cache.len >= cache.max_ctx {
            break;
        }
        model.forward_into(&[next], cache.len, cache, &mut logits);
    }
    if !pending.is_empty() {
        emit(&String::from_utf8_lossy(&pending));
    }
    n
}

/// True streaming: chunked transfer encoding, one SSE event per model slice,
/// flushed immediately so clients see tokens as they are generated.
fn write_sse_stream(
    stream: &mut TcpStream,
    model: &Model,
    cache: &mut crate::model::KvCache,
    prompt_text: &str,
    max_tokens: usize,
    params: &SamplingParams,
) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()?;
    let mut failed: Option<std::io::Error> = None;
    run_generator(model, cache, prompt_text, max_tokens, params, &mut |s| {
        if failed.is_some() {
            return;
        }
        // Slices from the decoder are always char-boundary aligned, so
        // per-slice escaping is boundary-safe.
        if let Err(e) = write_chunk(stream, format_sse_chunk(s).as_bytes()) {
            failed = Some(e);
        }
    });
    if let Some(e) = failed {
        return Err(e);
    }
    write_chunk(stream, b"data: [DONE]\n\n")?;
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

fn write_chunk(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    write!(stream, "{:X}\r\n", bytes.len())?;
    stream.write_all(bytes)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

// --- chat prompt mapping ---

/// System message (first only) -> Qwen template; user/assistant turns
/// concatenated as the user text (assistant history folded in for context).
fn build_chat_prompt_text(messages: &[(String, String)]) -> String {
    let mut system: Option<&str> = None;
    let mut user = String::new();
    for (role, content) in messages {
        if role == "system" && system.is_none() {
            system = Some(content);
            continue;
        }
        if !user.is_empty() {
            user.push_str("\n\n");
        }
        user.push_str(content);
    }
    format_qwen_chat(&user, system, false)
}

// --- minimal HTTP/1.1 parsing (request line, headers, Content-Length, body) ---

/// Parse `"METHOD /path?query HTTP/1.x"` -> (METHOD, path sans query). Pure.
pub fn parse_request_line(line: &str) -> Option<(String, String)> {
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?;
    if parts.next().map_or(true, |v| !v.starts_with("HTTP/")) {
        return None;
    }
    if method.is_empty() || !target.starts_with('/') {
        return None;
    }
    // Strip any query string; routes match on the path only.
    let path = target.split('?').next().unwrap_or("/").to_string();
    Some((method, path))
}

/// Read headers line-by-line, then exactly Content-Length body bytes.
/// All failures map to an (HTTP status, message) the caller writes back.
fn read_request(stream: &mut TcpStream) -> Result<(String, String, String), (u16, String)> {
    let err = |msg: &str| (400u16, msg.to_string());
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| (400u16, e.to_string()))?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).map_err(|_| err("unreadable request"))?;
    if request_line.len() > HEADER_LINE_MAX {
        return Err(err("request line too long"));
    }
    let (method, path) = parse_request_line(request_line.trim_end()).ok_or_else(|| err("bad request line"))?;
    let mut content_length: Option<i64> = None;
    let mut saw_chunked = false;
    for _ in 0..MAX_HEADERS {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|_| err("unreadable headers"))?;
        if line.len() > HEADER_LINE_MAX {
            return Err(err("header line too long"));
        }
        let t = line.trim_end();
        if t.is_empty() {
            break;
        }
        if let Some(v) = header_value(t, "content-length") {
            // Strict: garbage, negatives, and conflicting duplicates are 400.
            let n: i64 = v.trim().parse().map_err(|_| err("bad content-length"))?;
            if n < 0 {
                return Err(err("bad content-length"));
            }
            if let Some(prev) = content_length {
                if prev != n {
                    return Err(err("conflicting content-length"));
                }
            }
            content_length = Some(n);
        }
        if let Some(v) = header_value(t, "transfer-encoding") {
            if !v.trim().is_empty() {
                saw_chunked = true;
            }
        }
    }
    if saw_chunked {
        return Err((501u16, "chunked transfer-encoding not supported".to_string()));
    }
    let len = content_length.unwrap_or(0) as usize;
    if len > MAX_BODY {
        return Err((413u16, "body too large".to_string()));
    }
    let mut body = vec![0u8; len];
    // A short/failed body is a 400, never silent zero-padding.
    reader.read_exact(&mut body).map_err(|_| err("incomplete body"))?;
    Ok((method, path, String::from_utf8_lossy(&body).into_owned()))
}

/// Match a `Name: value` header line case-insensitively. Pure.
fn header_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let (k, v) = line.split_once(':')?;
    if k.trim().eq_ignore_ascii_case(name) { Some(v) } else { None }
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Content Too Large",
        431 => "Headers Too Large",
        501 => "Not Implemented",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

// --- strict JSON scanner (depth- and string-aware; message text cannot spoof
// --- top-level keys or object boundaries) ---

#[derive(Debug)]
enum JVal<'a> {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
    /// Raw slice of a nested array/object (re-parsed by the caller as needed).
    Raw(&'a str),
}

fn is_jws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && is_jws(b[*i]) {
        *i += 1;
    }
}

fn hex4(b: &[u8], i: &mut usize) -> Option<u32> {
    if *i + 4 > b.len() {
        return None;
    }
    let mut v = 0u32;
    for k in 0..4 {
        v = v * 16 + (b[*i + k] as char).to_digit(16)?;
    }
    *i += 4;
    Some(v)
}

/// Parse a JSON string starting at the opening quote; advances past it.
/// Full escapes incl. `\uXXXX` with surrogate-pair combination.
fn parse_string(b: &[u8], i: &mut usize) -> Option<String> {
    if b.get(*i) != Some(&b'"') {
        return None;
    }
    *i += 1;
    let mut out = String::new();
    loop {
        let c = *b.get(*i)?;
        *i += 1;
        match c {
            b'"' => return Some(out),
            b'\\' => {
                let e = *b.get(*i)?;
                *i += 1;
                match e {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hi = hex4(b, i)?;
                        let ch = if (0xD800..0xDC00).contains(&hi) {
                            // Surrogate pair: must be followed by \uDC00..DFFF.
                            if b.get(*i) != Some(&b'\\') || b.get(*i + 1) != Some(&b'u') {
                                return None;
                            }
                            *i += 2;
                            let lo = hex4(b, i)?;
                            if !(0xDC00..0xE000).contains(&lo) {
                                return None;
                            }
                            let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                            char::from_u32(cp)?
                        } else {
                            char::from_u32(hi)?
                        };
                        out.push(ch);
                    }
                    _ => return None,
                }
            }
            _ => {
                // Decode one UTF-8 scalar directly from the bytes.
                let rest = &b[*i - 1..];
                let s = std::str::from_utf8(rest).ok()?;
                let ch = s.chars().next()?;
                *i += ch.len_utf8() - 1;
                out.push(ch);
            }
        }
    }
}

/// Parse a JSON number starting at `i`; advances past it. Strict: at least one
/// digit, fully consumed token, then a valid delimiter.
fn parse_number(b: &[u8], i: &mut usize) -> Option<f64> {
    let start = *i;
    if b.get(*i) == Some(&b'-') {
        *i += 1;
    }
    let mut digits = 0usize;
    while *i < b.len() && b[*i].is_ascii_digit() {
        *i += 1;
        digits += 1;
    }
    if *i < b.len() && b[*i] == b'.' {
        *i += 1;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
            digits += 1;
        }
    }
    if *i < b.len() && (b[*i] == b'e' || b[*i] == b'E') {
        let mut j = *i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let ds = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == ds {
            return None;
        }
        *i = j;
    }
    if digits == 0 || *i == start {
        return None;
    }
    // Delimiter must follow (or end of input).
    if *i < b.len() && !matches!(b[*i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
        return None;
    }
    std::str::from_utf8(&b[start..*i]).ok()?.parse().ok()
}

fn parse_literal(b: &[u8], i: &mut usize, word: &[u8]) -> bool {
    if b.len() >= *i + word.len() && &b[*i..*i + word.len()] == word {
        let after = *i + word.len();
        if after >= b.len() || matches!(b[after], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
            *i = after;
            return true;
        }
    }
    false
}

/// Parse any JSON value at `i`; advances past it. Arrays/objects return a raw
/// slice (balanced, string-aware) for the caller to re-parse.
fn parse_value<'a>(b: &'a [u8], i: &mut usize) -> Option<JVal<'a>> {
    skip_ws(b, i);
    match b.get(*i)? {
        b'"' => Some(JVal::Str(parse_string(b, i)?)),
        b'{' | b'[' => {
            let open = b[*i];
            let close = if open == b'{' { b'}' } else { b']' };
            let start = *i;
            let mut depth = 0usize;
            let mut instr = false;
            let mut esc = false;
            while *i < b.len() {
                let c = b[*i];
                *i += 1;
                if instr {
                    if esc {
                        esc = false;
                    } else if c == b'\\' {
                        esc = true;
                    } else if c == b'"' {
                        instr = false;
                    }
                } else if c == b'"' {
                    instr = true;
                } else if c == open {
                    depth += 1;
                } else if c == close {
                    depth -= 1;
                    if depth == 0 {
                        return std::str::from_utf8(&b[start..*i])
                            .ok()
                            .map(JVal::Raw);
                    }
                }
            }
            None
        }
        b't' => parse_literal(b, i, b"true").then_some(JVal::Bool(true)),
        b'f' => parse_literal(b, i, b"false").then_some(JVal::Bool(false)),
        b'n' => parse_literal(b, i, b"null").then_some(JVal::Null),
        c if *c == b'-' || c.is_ascii_digit() => Some(JVal::Num(parse_number(b, i)?)),
        _ => None,
    }
}

/// Parse a top-level `{...}` object into (key, value) pairs. `None` =
/// malformed (caller maps to 400).
fn object_fields(body: &str) -> Option<Vec<(String, JVal<'_>)>> {
    let b = body.as_bytes();
    let mut i = 0usize;
    skip_ws(b, &mut i);
    if b.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    let mut out = Vec::new();
    loop {
        skip_ws(b, &mut i);
        if b.get(i) == Some(&b'}') {
            i += 1;
            skip_ws(b, &mut i);
            if i != b.len() {
                return None; // trailing garbage
            }
            return Some(out);
        }
        if !out.is_empty() {
            if b.get(i) != Some(&b',') {
                return None;
            }
            i += 1;
            skip_ws(b, &mut i);
        }
        let key = parse_string(b, &mut i)?;
        skip_ws(b, &mut i);
        if b.get(i) != Some(&b':') {
            return None;
        }
        i += 1;
        let val = parse_value(b, &mut i)?;
        out.push((key, val));
    }
}

/// Parse a top-level `[...]` array into values.
fn array_items(raw: &str) -> Option<Vec<JVal<'_>>> {
    let b = raw.as_bytes();
    let mut i = 0usize;
    skip_ws(b, &mut i);
    if b.get(i) != Some(&b'[') {
        return None;
    }
    i += 1;
    let mut out = Vec::new();
    loop {
        skip_ws(b, &mut i);
        if b.get(i) == Some(&b']') {
            return Some(out);
        }
        if !out.is_empty() {
            if b.get(i) != Some(&b',') {
                return None;
            }
            i += 1;
        }
        out.push(parse_value(b, &mut i)?);
    }
}

fn field<'a>(fields: &'a [(String, JVal<'a>)], key: &str) -> Option<&'a JVal<'a>> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// `messages: [{role, content}...]` pairs in order. `None` = malformed.
fn extract_messages(body: &str) -> Option<Vec<(String, String)>> {
    let fields = object_fields(body)?;
    let raw = match field(&fields, "messages")? {
        JVal::Raw(r) => *r,
        _ => return None,
    };
    let items = array_items(raw)?;
    let mut out = Vec::new();
    for it in &items {
        let JVal::Raw(obj) = it else { return None };
        let fs = object_fields(obj)?;
        let role = match field(&fs, "role") {
            Some(JVal::Str(s)) => s.clone(),
            _ => String::new(),
        };
        let content = match field(&fs, "content") {
            Some(JVal::Str(s)) => s.clone(),
            _ => String::new(),
        };
        if !role.is_empty() || !content.is_empty() {
            out.push((role, content));
        }
    }
    Some(out)
}

// --- request model + validation (pure; failures become 400s) ---

struct ChatRequest {
    messages: Vec<(String, String)>,
    temperature: Option<f32>,
    max_tokens: usize,
    stream: bool,
}

impl ChatRequest {
    fn effective_params(&self, base: &SamplingParams) -> SamplingParams {
        let mut p = base.clone();
        if let Some(t) = self.temperature {
            p.temperature = t;
        }
        p
    }
}

fn get_temperature(fields: &[(String, JVal<'_>)]) -> Result<Option<f32>, &'static str> {
    match field(fields, "temperature") {
        None => Ok(None),
        Some(JVal::Num(v)) if v.is_finite() => Ok(Some(*v as f32)),
        _ => Err("bad temperature (must be a finite number)"),
    }
}

fn get_max_tokens(fields: &[(String, JVal<'_>)]) -> Result<usize, &'static str> {
    match field(fields, "max_tokens") {
        None => Ok(256),
        Some(JVal::Num(v)) if v.is_finite() && *v >= 1.0 => Ok((*v as usize).clamp(1, MAX_TOKENS_CAP)),
        _ => Err("bad max_tokens (must be a number >= 1)"),
    }
}

fn get_stream(fields: &[(String, JVal<'_>)]) -> Result<bool, &'static str> {
    match field(fields, "stream") {
        None => Ok(false),
        Some(JVal::Bool(b)) => Ok(*b),
        _ => Err("bad stream (must be true/false)"),
    }
}

/// Parse a chat-completions body. `Err` message becomes the 400 body.
fn parse_chat_body(body: &str) -> Result<ChatRequest, &'static str> {
    let fields = object_fields(body).ok_or("invalid json")?;
    let messages = extract_messages(body).ok_or("missing messages")?;
    if messages.is_empty() {
        return Err("missing messages");
    }
    Ok(ChatRequest {
        messages,
        temperature: get_temperature(&fields)?,
        max_tokens: get_max_tokens(&fields)?,
        stream: get_stream(&fields)?,
    })
}

/// Parse a completions body into (prompt, max_tokens, stream, temp-override).
fn parse_completion_body(body: &str) -> Result<(String, usize, bool, Option<f32>), &'static str> {
    let fields = object_fields(body).ok_or("invalid json")?;
    let prompt = match field(&fields, "prompt") {
        Some(JVal::Str(s)) if !s.is_empty() => s.clone(),
        _ => return Err("missing prompt"),
    };
    Ok((
        prompt,
        get_max_tokens(&fields)?,
        get_stream(&fields)?,
        get_temperature(&fields)?,
    ))
}

fn merge_temp(base: &SamplingParams, t: Option<f32>) -> SamplingParams {
    let mut p = base.clone();
    if let Some(t) = t {
        p.temperature = t;
    }
    p
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn format_chat_response(content: &str) -> String {
    format!(
        "{{\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}}}}]}}",
        json_escape(content)
    )
}

fn format_completion_response(content: &str) -> String {
    format!("{{\"choices\":[{{\"text\":\"{}\"}}]}}", json_escape(content))
}

/// One SSE chunk carrying a content delta. Pure + tested.
fn format_sse_chunk(delta: &str) -> String {
    format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n", json_escape(delta))
}

fn error_json(msg: &str) -> String {
    format!("{{\"error\":\"{}\"}}", json_escape(msg))
}

// --- tests (pure functions only — no sockets) ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_line_ok() {
        let (m, p) = parse_request_line("POST /v1/chat/completions HTTP/1.1").unwrap();
        assert_eq!((m.as_str(), p.as_str()), ("POST", "/v1/chat/completions"));
        let (m, p) = parse_request_line("GET  /health  HTTP/1.0").unwrap();
        assert_eq!((m.as_str(), p.as_str()), ("GET", "/health"));
        // Query strings are stripped for routing.
        let (m, p) = parse_request_line("GET /health?x=1 HTTP/1.1").unwrap();
        assert_eq!((m.as_str(), p.as_str()), ("GET", "/health"));
    }

    #[test]
    fn request_line_rejects_bad() {
        assert!(parse_request_line("").is_none());
        assert!(parse_request_line("GET HTTP/1.1").is_none());
        assert!(parse_request_line("GET nope HTTP/1.1").is_none());
        assert!(parse_request_line("GET /health FROB/1.1").is_none());
    }

    #[test]
    fn chat_body_fields() {
        let body = r#"{"model":"x", "stream" : true, "max_tokens" : 64,
            "temperature":0.5,
            "messages":[{"role":"system","content":"Be helpful."},
                        {"role":"user","content":"Hi"}]}"#;
        let base = SamplingParams::default();
        let req = parse_chat_body(body).unwrap();
        assert!(req.stream && req.max_tokens == 64 && req.temperature == Some(0.5));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].0, "system");
        assert_eq!(req.messages[1].1, "Hi");
        // Reordered keys + extra whitespace still parse; default temp kept.
        let req2 = parse_chat_body(
            r#"{"messages" : [ {"content" : "Yo" , "role" : "user"} ] ,"stream":false}"#,
        )
        .unwrap();
        assert!(!req2.stream && req2.messages.len() == 1);
        assert_eq!(req2.effective_params(&base).temperature, base.temperature);
    }

    #[test]
    fn sse_chunk_format() {
        let c = format_sse_chunk("he\"llo\n");
        assert!(c.starts_with("data: ") && c.ends_with("\n\n"));
        assert!(c.contains(r#"content":"he\"llo\n""#));
        let r = format_chat_response("a\"b");
        assert!(r.contains(r#""role":"assistant""#) && r.contains(r#"content":"a\"b""#));
    }

    #[test]
    fn scanner_escapes_and_unicode() {
        // \u escapes incl. a surrogate pair (U+1D11E MUSICAL SYMBOL G CLEF).
        let body = r#"{"prompt":"caf\u00e9 \uD834\uDD1E \b\f\/","max_tokens":8}"#;
        let fs = object_fields(body).unwrap();
        match field(&fs, "prompt") {
            Some(JVal::Str(s)) => assert_eq!(s, "café \u{1D11E} \u{8}\u{c}/"),
            other => panic!("unexpected {other:?}"),
        }
        // Lone surrogate is malformed.
        assert!(object_fields(r#"{"prompt":"\uD834"}"#).is_none());
        // Content containing braces/quotes/keys never breaks object parsing.
        let tricky = r#"{"messages":[{"role":"user","content":"say {\"temperature\": 0} } ok"}],"max_tokens":5}"#;
        let msgs = extract_messages(tricky).unwrap();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].1.contains("temperature"));
        let fs2 = object_fields(tricky).unwrap();
        assert!(matches!(field(&fs2, "max_tokens"), Some(JVal::Num(v)) if *v == 5.0));
    }

    #[test]
    fn scanner_strict_numbers_and_bools() {
        let fs = object_fields(r#"{"a":64,"b":-1.5,"c":1e3,"d":true,"e":false,"f":null}"#).unwrap();
        assert!(matches!(field(&fs, "a"), Some(JVal::Num(v)) if *v == 64.0));
        assert!(matches!(field(&fs, "c"), Some(JVal::Num(v)) if *v == 1000.0));
        assert!(matches!(field(&fs, "d"), Some(JVal::Bool(true))));
        // Trailing garbage / bad exponent / bare words are malformed.
        assert!(object_fields(r#"{"a":64x}"#).is_none());
        assert!(object_fields(r#"{"a":1e}"#).is_none());
        assert!(object_fields(r#"{"a":truish}"#).is_none());
        assert!(object_fields(r#"{"a":}"#).is_none());
        // NaN/Infinity are not JSON numbers.
        assert!(object_fields(r#"{"temperature":NaN}"#).is_none());
    }

    #[test]
    fn validation_rejects_bad_params() {
        assert!(parse_chat_body(r#"{"messages":[]}"#).is_err());
        assert!(parse_chat_body(r#"{}"#).is_err());
        assert!(parse_chat_body(r#"not json"#).is_err());
        assert!(parse_chat_body(r#"{"messages":[{"role":"user","content":"hi"}],"temperature":NaN}"#).is_err());
        // max_tokens clamps on both paths instead of pinning the server.
        let req = parse_chat_body(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":1e18}"#).unwrap();
        assert_eq!(req.max_tokens, MAX_TOKENS_CAP);
        assert!(parse_chat_body(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":-3}"#).is_err());
        assert!(parse_completion_body(r#"{"prompt":""}"#).is_err());
        assert!(parse_completion_body(r#"{}"#).is_err());
        let (p, m, s, t) = parse_completion_body(r#"{"prompt":"hi","max_tokens":7,"stream":true,"temperature":0.2}"#).unwrap();
        assert_eq!((p.as_str(), m, s, t), ("hi", 7, true, Some(0.2)));
    }
}
