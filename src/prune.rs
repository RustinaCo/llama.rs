//! Vocabulary pruning: rewrite a GGUF file keeping only the tokens needed for
//! a Latin + Arabic/Persian (+ symbols/emoji) deployment.
//!
//! Why this helps the stated goals ("nothing extra in the final platform"):
//! * The embedding and output matrices scale with vocabulary size. Qwen3's
//!   151,936-token vocabulary at `n_embd = 1024` is ~150M parameters — over a
//!   quarter of a 0.6B model — most of it CJK/Cyrillic/other scripts.
//! * After pruning, those rows are gone from the file: smaller download,
//!   smaller mmap footprint, and a smaller final `output.weight` matmul per
//!   token (logits over ~40k instead of ~152k).
//! * The engine reads the vocabulary from the GGUF itself, so a pruned model
//!   runs with zero code changes.
//!
//! Safety properties:
//! * All control / user-defined / unknown / byte tokens are always kept, so
//!   the byte-fallback at the heart of BPE can never break: any text still
//!   encodes (unwanted scripts just encode into many small pieces).
//! * Every kept BPE merge still resolves (both parts and the merged token are
//!   kept); merges that would produce a dropped token are removed, which is
//!   exactly what [`crate::tokenizer::Tokenizer`] would skip on load anyway.
//! * Every `*_token_id` metadata key (bos/eos/pad/...) is remapped.
//!
//! Note: a pruned model still needs a short LoRA fine-tune to recover quality
//! (see `tools/data_prep.py`); pruning alone only removes capacity.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::gguf::{GgufFile, MetaValue, TensorInfo};
use crate::tokenizer::decode_token_bytes;

/// Unicode ranges belonging to scripts we do not need for a
/// Persian/English deployment. Anything not listed here is kept (digits,
/// punctuation, symbols, emoji, PUA, ...), which is the safe direction:
/// unknown symbols stay usable, only whole unwanted languages are removed.
const DROP_RANGES: &[(u32, u32)] = &[
    (0x0400, 0x052F), // Cyrillic + supplement
    (0x0530, 0x058F), // Armenian
    (0x0590, 0x05FF), // Hebrew
    (0x0700, 0x07FF), // Syriac, Thaana, NKo
    (0x0800, 0x089F), // Samaritan, Mandaic... (Arabic Ext-A at 08A0+ is kept)
    (0x0900, 0x0DFF), // Indic scripts (Devanagari .. Sinhala)
    (0x0E00, 0x0EFF), // Thai, Lao
    (0x0F00, 0x0FFF), // Tibetan
    (0x1000, 0x109F), // Myanmar
    (0x10A0, 0x10FF), // Georgian
    (0x1100, 0x11FF), // Hangul Jamo
    (0x1200, 0x139F), // Ethiopic (+ supplement)
    (0x13A0, 0x13FF), // Cherokee
    (0x1400, 0x167F), // Canadian Aboriginal syllabics
    (0x1680, 0x16FF), // Ogham, Runic
    (0x1700, 0x18FF), // Philippine scripts, Mongolian, ...
    (0x1900, 0x1DFF), // SE-Asian scripts + phonetic extensions
    (0x1F00, 0x1FFF), // Greek Extended (polytonic; modern Greek 0370-03FF kept)
    (0x2C00, 0x2C5F), // Glagolitic
    (0x2C80, 0x2CFF), // Coptic
    (0x2D00, 0x2D7F), // Georgian supplement, Tifinagh
    (0x2D80, 0x2DDF), // Ethiopic Extended
    (0x2DE0, 0x2DFF), // Cyrillic Extended-A
    (0x2E80, 0x2FFF), // CJK radicals, Kangxi, description chars
    (0x3000, 0x303F), // CJK symbols & punctuation
    (0x3040, 0x30FF), // Hiragana, Katakana
    (0x3100, 0x312F), // Bopomofo
    (0x3130, 0x318F), // Hangul Compatibility Jamo
    (0x3190, 0x31FF), // Kanbun, Bopomofo ext, CJK strokes, Katakana ext
    (0x3200, 0x33FF), // Enclosed / squared CJK
    (0x3400, 0x4DBF), // CJK Extension A
    (0x4DC0, 0x4DFF), // Yijing hexagrams
    (0x4E00, 0x9FFF), // CJK Unified Ideographs
    (0xA000, 0xA71F), // Yi, Lisu, Vai, Cyrillic ext-B, Bamum, tone mods
    (0xA800, 0xAFFF), // Phags-pa, Javanese, Cham, Tai Viet, Meetei, ...
    (0xAB00, 0xAB2F), // Ethiopic Extended-A
    (0xAB70, 0xABFF), // Cherokee supplement, Meetei extensions
    (0xAC00, 0xD7FF), // Hangul syllables + Jamo Extended-B
    (0xF900, 0xFAFF), // CJK compatibility ideographs
    (0xFE30, 0xFE4F), // CJK compatibility forms
    (0xFE50, 0xFE6F), // Small form variants (CJK punctuation)
    (0xFF00, 0xFFEF), // Fullwidth / halfwidth forms
    (0x10000, 0x1D0FF), // Ancient/historic scripts (Linear B, Brahmi, ...)
    (0x1D200, 0x1D3FF), // Ancient Greek symbols, Tai Xuan Jing
    (0x1D800, 0x1EFFF), // Sutton SignWriting (historic scripts kept out)
    (0x20000, 0x2FFFF), // CJK Extensions B-F + compatibility supplement
    (0x30000, 0x3FFFF), // CJK Extension G+
];

/// Kept sub-ranges inside the astral drop zones (notation systems, not
/// languages): musical symbols and mathematical alphanumerics.
const KEEP_ASTRAL: &[(u32, u32)] = &[(0x1D100, 0x1D1FF), (0x1D400, 0x1D7FF)];

fn char_dropped(c: char) -> bool {
    let u = c as u32;
    if KEEP_ASTRAL.iter().any(|&(a, b)| u >= a && u <= b) {
        return false;
    }
    DROP_RANGES.iter().any(|&(a, b)| u >= a && u <= b)
}

/// True if the token's real surface text uses only scripts we keep.
///
/// The GGUF string is byte-level mapped, so it is first decoded to raw bytes
/// and then interpreted as UTF-8 (lossy: lone BPE fragments become U+FFFD,
/// which is kept — fragments must survive for merges to work).
fn surface_kept(token: &str) -> bool {
    let bytes = decode_token_bytes(token);
    let s = String::from_utf8_lossy(&bytes);
    !s.chars().any(char_dropped)
}

#[derive(Debug)]
pub struct PruneStats {
    pub old_vocab: usize,
    pub new_vocab: usize,
    pub kept_special: usize,
    pub kept_byte: usize,
    pub kept_normal: usize,
    pub dropped: usize,
    pub merges_old: usize,
    pub merges_new: usize,
    pub embd_bytes_saved: u64,
    pub out_bytes: u64,
}

/// Rewrite `in_path` to `out_path`, dropping vocabulary (and the matching
/// embedding/output rows) for scripts outside Latin + Arabic/Persian.
pub fn prune_vocab<P: AsRef<Path>, Q: AsRef<Path>>(in_path: P, out_path: Q) -> io::Result<PruneStats> {
    let gguf = GgufFile::open(&in_path)?;

    let tokens: Vec<String> = gguf
        .meta("tokenizer.ggml.tokens")
        .and_then(|v| v.as_string_vec())
        .ok_or_else(|| bad("tokenizer.ggml.tokens"))?;
    let token_types: Vec<u32> = gguf
        .meta("tokenizer.ggml.token_type")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().map(|x| x.as_u32().unwrap_or(1)).collect())
        .unwrap_or_else(|| vec![1u32; tokens.len()]);
    let merges: Vec<String> = gguf
        .meta("tokenizer.ggml.merges")
        .and_then(|v| v.as_string_vec())
        .unwrap_or_default();
    if token_types.len() != tokens.len() {
        return Err(bad("tokenizer.ggml.token_type length mismatch"));
    }
    let vocab = tokens.len();

    // Token ids that must survive regardless of script.
    let mut pinned = vec![false; vocab];
    for key in gguf.metadata.keys() {
        if key.ends_with("_token_id") {
            if let Some(id) = gguf.meta(key).and_then(|v| v.as_u32()) {
                if (id as usize) < vocab {
                    pinned[id as usize] = true;
                }
            }
        }
    }

    let mut keep = vec![false; vocab];
    let mut kept_special = 0usize;
    let mut kept_byte = 0usize;
    let mut kept_normal = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        let ty = token_types[i];
        // 2 = unknown, 3 = control, 4 = user-defined, 6 = byte.
        let structural = matches!(ty, 2 | 3 | 4 | 6) || pinned[i];
        if structural {
            keep[i] = true;
            if matches!(ty, 3 | 4) {
                kept_special += 1;
            } else if ty == 6 {
                kept_byte += 1;
            } else {
                kept_normal += 1;
            }
        } else if surface_kept(t) {
            keep[i] = true;
            kept_normal += 1;
        }
    }
    let new_vocab: usize = keep.iter().filter(|&&k| k).count();
    if new_vocab == 0 {
        return Err(bad("prune would remove the entire vocabulary"));
    }

    // Old id -> new id.
    let mut new_id = vec![u32::MAX; vocab];
    let mut next = 0u32;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            new_id[i] = next;
            next += 1;
        }
    }

    // Filter merges: keep only merges fully resolvable inside the kept set
    // (mirrors Tokenizer::new, which skips the rest on load).
    let old_to_id: HashMap<&str, u32> =
        tokens.iter().enumerate().map(|(i, t)| (t.as_str(), i as u32)).collect();
    let mut merges_new = Vec::new();
    for m in &merges {
        if let Some((l, r)) = m.split_once(' ') {
            if let (Some(&li), Some(&ri)) = (old_to_id.get(l), old_to_id.get(r)) {
                let merged = format!("{l}{r}");
                if let Some(&mi) = old_to_id.get(merged.as_str()) {
                    if keep[li as usize] && keep[ri as usize] && keep[mi as usize] {
                        merges_new.push(m.clone());
                    }
                }
            }
        }
    }

    // Rebuild metadata.
    let mut meta = gguf.metadata.clone();
    let mut new_tokens = Vec::with_capacity(new_vocab);
    let mut new_types = Vec::with_capacity(new_vocab);
    for (i, t) in tokens.iter().enumerate() {
        if keep[i] {
            new_tokens.push(MetaValue::Str(t.clone()));
            new_types.push(MetaValue::I32(token_types[i] as i32));
        }
    }
    meta.insert("tokenizer.ggml.tokens".to_string(), MetaValue::Array(new_tokens));
    meta.insert("tokenizer.ggml.token_type".to_string(), MetaValue::Array(new_types));
    meta.insert(
        "tokenizer.ggml.merges".to_string(),
        MetaValue::Array(merges_new.iter().cloned().map(MetaValue::Str).collect()),
    );
    // Remap every *_token_id key.
    for key in gguf.metadata.keys() {
        if key.ends_with("_token_id") {
            if let Some(id) = gguf.meta(key).and_then(|v| v.as_u32()) {
                let ni = new_id
                    .get(id as usize)
                    .copied()
                    .filter(|&n| n != u32::MAX)
                    .ok_or_else(|| {
                        bad(&format!("special token id {id} ('{key}') was pruned"))
                    })?;
                meta.insert(key.clone(), MetaValue::U32(ni));
            }
        }
    }

    // Slice the vocabulary-sized tensors (rows = token ids).
    let mut new_infos: Vec<TensorInfo> = Vec::with_capacity(gguf.tensor_order.len());
    let mut blobs: Vec<std::borrow::Cow<'_, [u8]>> = Vec::with_capacity(gguf.tensor_order.len());
    let mut embd_saved: u64 = 0;
    for name in &gguf.tensor_order {
        let info = &gguf.tensors[name];
        let data = gguf.tensor_bytes(name)?;
        let is_vocab_sized =
            (name == "token_embd.weight" || name == "output.weight") && info.n_rows() == vocab;
        if is_vocab_sized {
            let rb = info.ty.row_size(info.n_cols());
            let mut sliced = Vec::with_capacity(new_vocab * rb);
            for (i, &k) in keep.iter().enumerate() {
                if k {
                    sliced.extend_from_slice(&data[i * rb..i * rb + rb]);
                }
            }
            embd_saved += (data.len() - sliced.len()) as u64;
            let mut ni = info.clone();
            ni.dims = vec![info.n_cols(), new_vocab];
            new_infos.push(ni);
            blobs.push(std::borrow::Cow::Owned(sliced));
        } else {
            new_infos.push(info.clone());
            blobs.push(std::borrow::Cow::Borrowed(data));
        }
    }

    let alignment = gguf.meta("general.alignment").and_then(|v| v.as_usize()).unwrap_or(32);
    write_gguf(&out_path, gguf.version, &meta, &new_infos, &blobs, alignment)?;
    let out_bytes = std::fs::metadata(&out_path)?.len();

    Ok(PruneStats {
        old_vocab: vocab,
        new_vocab,
        kept_special,
        kept_byte,
        kept_normal,
        dropped: vocab - new_vocab,
        merges_old: merges.len(),
        merges_new: merges_new_len(&meta),
        embd_bytes_saved: embd_saved,
        out_bytes,
    })
}

fn merges_new_len(meta: &HashMap<String, MetaValue>) -> usize {
    meta.get("tokenizer.ggml.merges")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("prune: {msg}"))
}

fn align_up(x: usize, a: usize) -> usize {
    if a == 0 { x } else { (x + a - 1) / a * a }
}

fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn write_value<W: Write>(w: &mut W, v: &MetaValue) -> io::Result<()> {
    match v {
        MetaValue::U8(x) => { w.write_all(&0u32.to_le_bytes())?; w.write_all(&[*x])?; }
        MetaValue::I8(x) => { w.write_all(&1u32.to_le_bytes())?; w.write_all(&[*x as u8])?; }
        MetaValue::U16(x) => { w.write_all(&2u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        MetaValue::I16(x) => { w.write_all(&3u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        MetaValue::U32(x) => { w.write_all(&4u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        MetaValue::I32(x) => { w.write_all(&5u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        MetaValue::F32(x) => { w.write_all(&6u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        MetaValue::Bool(x) => { w.write_all(&7u32.to_le_bytes())?; w.write_all(&[*x as u8])?; }
        MetaValue::Str(x) => { w.write_all(&8u32.to_le_bytes())?; write_str(w, x)?; }
        MetaValue::Array(items) => {
            w.write_all(&9u32.to_le_bytes())?;
            let et = items.first().map(elem_tag).unwrap_or(8);
            w.write_all(&et.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            for it in items {
                write_array_elem(w, it)?;
            }
        }
        MetaValue::U64(x) => { w.write_all(&10u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        MetaValue::I64(x) => { w.write_all(&11u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
        MetaValue::F64(x) => { w.write_all(&12u32.to_le_bytes())?; w.write_all(&x.to_le_bytes())?; }
    }
    Ok(())
}

/// Element tag for array items (GGUF arrays are homogeneous).
fn elem_tag(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U8(_) => 0,
        MetaValue::I8(_) => 1,
        MetaValue::U16(_) => 2,
        MetaValue::I16(_) => 3,
        MetaValue::U32(_) => 4,
        MetaValue::I32(_) => 5,
        MetaValue::F32(_) => 6,
        MetaValue::Bool(_) => 7,
        MetaValue::Str(_) => 8,
        MetaValue::Array(_) => 9,
        MetaValue::U64(_) => 10,
        MetaValue::I64(_) => 11,
        MetaValue::F64(_) => 12,
    }
}

/// Array items are written bare (no per-item tag), mirroring the reader.
fn write_array_elem<W: Write>(w: &mut W, v: &MetaValue) -> io::Result<()> {
    match v {
        MetaValue::U8(x) => w.write_all(&[*x])?,
        MetaValue::I8(x) => w.write_all(&[*x as u8])?,
        MetaValue::U16(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::I16(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::U32(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::I32(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::F32(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::Bool(x) => w.write_all(&[*x as u8])?,
        MetaValue::Str(x) => write_str(w, x)?,
        MetaValue::U64(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::I64(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::F64(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::Array(_) => {
            return Err(bad("nested arrays are not supported by the writer"));
        }
    }
    Ok(())
}

fn write_gguf<P: AsRef<Path>>(
    path: P,
    version: u32,
    meta: &HashMap<String, MetaValue>,
    infos: &[TensorInfo],
    blobs: &[std::borrow::Cow<'_, [u8]>],
    alignment: usize,
) -> io::Result<()> {
    // Tensor data offsets (each aligned), mirroring the reader's layout.
    let mut offsets = Vec::with_capacity(infos.len());
    let mut off = 0usize;
    for info in infos {
        off = align_up(off, alignment);
        offsets.push(off);
        off += info.nbytes();
    }

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(b"GGUF")?;
    w.write_all(&version.to_le_bytes())?;
    w.write_all(&(infos.len() as u64).to_le_bytes())?;
    w.write_all(&(meta.len() as u64).to_le_bytes())?;

    let mut keys: Vec<&String> = meta.keys().collect();
    keys.sort();
    for k in keys {
        write_str(&mut w, k)?;
        write_value(&mut w, &meta[k])?;
    }

    for (info, &off) in infos.iter().zip(offsets.iter()) {
        write_str(&mut w, &info.name)?;
        w.write_all(&(info.dims.len() as u32).to_le_bytes())?;
        for d in &info.dims {
            w.write_all(&(*d as u64).to_le_bytes())?;
        }
        w.write_all(&info.ty.as_u32().to_le_bytes())?;
        w.write_all(&(off as u64).to_le_bytes())?;
    }

    // Pad to the data section, then write each blob at its offset.
    use std::io::Seek;
    let mut cur = w.seek(std::io::SeekFrom::Current(0))?;
    // data_start = align_up(header_len, alignment); header_len == cur now.
    let data_start = align_up(cur as usize, alignment) as u64;
    while cur < data_start {
        w.write_all(&[0u8])?;
        cur += 1;
    }
    for (idx, blob) in blobs.iter().enumerate() {
        let want = data_start + offsets[idx] as u64;
        while cur < want {
            w.write_all(&[0u8])?;
            cur += 1;
        }
        w.write_all(blob)?;
        cur += blob.len() as u64;
    }
    w.flush()?;
    Ok(())
}
