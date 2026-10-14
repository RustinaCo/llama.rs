//! Byte-level BPE tokenizer (GPT-2 / Qwen2 flavour), built from GGUF vocab data.

use std::collections::HashMap;

/// GPT-2 `bytes_to_unicode()` mapping: byte value -> unicode scalar used in the
/// vocabulary strings.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    for b in 0x21u32..=0x7e {
        bs.push(b);
    }
    for b in 0xa1u32..=0xac {
        bs.push(b);
    }
    for b in 0xaeu32..=0xff {
        bs.push(b);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

pub struct Tokenizer {
    id_to_token: Vec<String>,
    /// (left_id, right_id) -> (rank, merged_id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_to_id: [u32; 256],
    char_to_byte: HashMap<char, u8>,
    specials: Vec<(String, u32)>, // sorted by length desc
    pub bos_id: u32,
    pub eos_id: u32,
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_numeric()
}
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

impl Tokenizer {
    pub fn new(
        tokens: Vec<String>,
        token_types: &[u32],
        merges: Vec<String>,
        bos_id: u32,
        eos_id: u32,
    ) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }

        let byte_to_char = bytes_to_unicode();
        let mut char_to_byte = HashMap::with_capacity(256);
        for (b, &c) in byte_to_char.iter().enumerate() {
            char_to_byte.insert(c, b as u8);
        }
        let mut byte_to_id = [0u32; 256];
        for b in 0..256 {
            let s: String = byte_to_char[b].to_string();
            if let Some(&id) = token_to_id.get(&s) {
                byte_to_id[b] = id;
            }
        }

        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            if let Some((l, r)) = m.split_once(' ') {
                if let (Some(&li), Some(&ri), Some(&mi)) = (
                    token_to_id.get(l),
                    token_to_id.get(r),
                    token_to_id.get(&format!("{l}{r}")),
                ) {
                    merge_map.insert((li, ri), (rank as u32, mi));
                }
            }
        }

        let mut specials: Vec<(String, u32)> = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let ty = token_types.get(i).copied().unwrap_or(1);
            if ty == 3 || ty == 4 {
                specials.push((t.clone(), i as u32));
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Tokenizer {
            id_to_token: tokens,
            merges: merge_map,
            byte_to_id,
            char_to_byte,
            specials,
            bos_id,
            eos_id,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Raw byte sequence represented by a token id (inverse of the byte-level
    /// unicode mapping). Used for streaming decode.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let mut out = Vec::new();
        let Some(s) = self.id_to_token.get(id as usize) else {
            return out;
        };
        for ch in s.chars() {
            if let Some(&b) = self.char_to_byte.get(&ch) {
                out.push(b);
            } else {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        out
    }

    /// Pre-tokenize using the Qwen2/GPT-2 regex, expressed directly:
    /// `(?:'[sS]|...)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let n = chars.len();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < n {
            let c = chars[i].1;
            let mut end = 0usize;

            // alt1: contraction
            if c == '\'' {
                if let Some(len) = self.match_contraction(&chars, i) {
                    end = i + len;
                }
            }

            // alt2: [^\r\n\p{L}\p{N}]?\p{L}+
            if end == 0 {
                if is_letter(c) {
                    let mut j = i + 1;
                    while j < n && is_letter(chars[j].1) {
                        j += 1;
                    }
                    end = j;
                } else if c != '\r' && c != '\n' && !is_number(c) && i + 1 < n && is_letter(chars[i + 1].1) {
                    let mut j = i + 1;
                    while j < n && is_letter(chars[j].1) {
                        j += 1;
                    }
                    end = j;
                }
            }

            // alt3: \p{N}
            if end == 0 && is_number(c) {
                end = i + 1;
            }

            // alt5: \s*[\r\n]+
            if end == 0 && is_space(c) {
                let mut j = i;
                while j < n && is_space(chars[j].1) && chars[j].1 != '\r' && chars[j].1 != '\n' {
                    j += 1;
                }
                let mut k = j;
                while k < n && (chars[k].1 == '\r' || chars[k].1 == '\n') {
                    k += 1;
                }
                if k > j {
                    end = k;
                } else {
                    // alt6: \s+(?!\S)  and  alt7: \s+
                    let mut run = i;
                    while run < n && is_space(chars[run].1) {
                        run += 1;
                    }
                    let len = run - i;
                    if run == n {
                        end = run;
                    } else if len >= 2 {
                        end = run - 1; // leave the final space for the next token
                    } else {
                        end = run;
                    }
                }
            }

            // alt4: ' ?[^\s\p{L}\p{N}]+[\r\n]*'
            if end == 0 && !is_letter(c) && !is_number(c) && !is_space(c) {
                let mut j = i;
                if chars[j].1 == ' ' {
                    j += 1;
                }
                let mid = j;
                while j < n
                    && !is_space(chars[j].1)
                    && !is_letter(chars[j].1)
                    && !is_number(chars[j].1)
                {
                    j += 1;
                }
                if j > mid {
                    while j < n && (chars[j].1 == '\r' || chars[j].1 == '\n') {
                        j += 1;
                    }
                    end = j;
                }
            }

            if end <= i {
                end = i + 1; // guaranteed progress
            }
            let b0 = chars[i].0;
            let b1 = if end >= n { text.len() } else { chars[end].0 };
            out.push(&text[b0..b1]);
            i = end;
        }
        out
    }

    fn match_contraction(&self, chars: &[(usize, char)], i: usize) -> Option<usize> {
        let n = chars.len();
        if i + 1 >= n {
            return None;
        }
        let c1 = chars[i + 1].1;
        match c1 {
            's' | 'S' | 't' | 'T' | 'm' | 'M' | 'd' | 'D' => Some(2),
            'r' | 'R' | 'v' | 'V' | 'l' | 'L' => {
                if i + 2 < n {
                    let c2 = chars[i + 2].1;
                    let ok = matches!(c2, 'e' | 'E' | 'l' | 'L');
                    if ok {
                        return Some(3);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn bpe(&self, bytes: &[u8]) -> Vec<u32> {
        let mut syms: Vec<u32> = bytes.iter().map(|&b| self.byte_to_id[b as usize]).collect();
        if syms.len() < 2 {
            return syms;
        }
        loop {
            let mut best: Option<(usize, u32, u32)> = None;
            for i in 0..syms.len() - 1 {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1])) {
                    if best.map_or(true, |(_, br, _)| rank < br) {
                        best = Some((i, rank, merged));
                    }
                }
            }
            match best {
                Some((i, _, merged)) => {
                    syms[i] = merged;
                    syms.remove(i + 1);
                    if syms.len() < 2 {
                        break;
                    }
                }
                None => break,
            }
        }
        syms
    }

    /// Encode text, recognizing special tokens verbatim.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut pos = 0usize;
        while pos < text.len() {
            let mut found: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(rel) = text[pos..].find(s.as_str()) {
                    let start = pos + rel;
                    if found.map_or(true, |(fs, _, _)| start < fs) {
                        found = Some((start, start + s.len(), *id));
                    }
                }
            }
            match found {
                Some((start, end, id)) => {
                    if start > pos {
                        self.encode_normal(&text[pos..start], &mut ids);
                    }
                    ids.push(id);
                    pos = end;
                }
                None => {
                    self.encode_normal(&text[pos..], &mut ids);
                    break;
                }
            }
        }
        ids
    }

    fn encode_normal(&self, text: &str, ids: &mut Vec<u32>) {
        for piece in self.pretokenize(text) {
            ids.extend(self.bpe(piece.as_bytes()));
        }
    }

    /// Decode ids back to text (lossy UTF-8).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let s = match self.id_to_token.get(id as usize) {
                Some(s) => s,
                None => continue,
            };
            for ch in s.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Decode a GGUF vocabulary string back to the raw bytes it represents.
///
/// GPT-2/Qwen2 vocabularies store non-printable bytes through the
/// `bytes_to_unicode` mapping; this inverts that mapping (mirroring
/// [`Tokenizer::token_bytes`]) so caller code can inspect the true surface
/// text of a token for script classification.
pub fn decode_token_bytes(s: &str) -> Vec<u8> {
    use std::sync::OnceLock;
    static MAP: OnceLock<std::collections::HashMap<char, u8>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let t = bytes_to_unicode();
        let mut m = std::collections::HashMap::with_capacity(256);
        for (b, &c) in t.iter().enumerate() {
            m.insert(c, b as u8);
        }
        m
    });
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        if let Some(&b) = map.get(&ch) {
            out.push(b);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

/// Minimal Qwen3 chat prompt formatter.
pub fn format_qwen_chat(prompt: &str, system: Option<&str>, no_think: bool) -> String {
    let mut s = String::new();
    if let Some(sys) = system {
        s.push_str("<|im_start|>system\n");
        s.push_str(sys);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(prompt);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if no_think {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s
}
