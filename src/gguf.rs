//! Minimal GGUF v2/v3 reader.
//!
//! The file is memory-mapped, so tensor bytes are paged in lazily by the OS and
//! never copied into a heap buffer. All metadata is parsed eagerly into a map.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;

use crate::quant::GgmlType;

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::U32(v) => Some(*v),
            MetaValue::U64(v) => Some(*v as u32),
            MetaValue::I32(v) => Some(*v as u32),
            MetaValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u32().map(|v| v as usize)
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Materialize a `Vec<String>` from an array of strings.
    pub fn as_string_vec(&self) -> Option<Vec<String>> {
        let arr = self.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            out.push(v.as_str()?.to_string());
        }
        Some(out)
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGML dimensions: `dims[0]` is the fastest-varying (row) dimension.
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        let bs = self.ty.block_size();
        (self.n_elements() / bs) * self.ty.type_size()
    }

    /// Number of rows (product of all dims except the fastest-varying one).
    pub fn n_rows(&self) -> usize {
        if self.dims.len() <= 1 {
            1
        } else {
            self.dims[1..].iter().product()
        }
    }

    /// Length of one row (the fastest-varying dimension).
    pub fn n_cols(&self) -> usize {
        self.dims.first().copied().unwrap_or(1)
    }
}

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: HashMap<String, TensorInfo>,
    pub tensor_order: Vec<String>,
    data_offset: usize,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        // checked_add: a malicious length (e.g. u64::MAX) must error, never
        // wrap around the bounds check and panic the slicer below.
        let end = self.pos.checked_add(n).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "gguf: length overflow")
        })?;
        if end > self.buf.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "gguf: unexpected eof"));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> io::Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> io::Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn i16(&mut self) -> io::Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> io::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> io::Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> io::Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn i64(&mut self) -> io::Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn bool(&mut self) -> io::Result<bool> {
        Ok(self.u8()? != 0)
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    fn value(&mut self, ty: u32) -> io::Result<MetaValue> {
        Ok(match ty {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.i8()?),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.i16()?),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.i32()?),
            6 => MetaValue::F32(self.f32()?),
            7 => MetaValue::Bool(self.bool()?),
            8 => MetaValue::Str(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::new();
                for _ in 0..n {
                    v.push(self.value(elem_ty)?);
                }
                MetaValue::Array(v)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.i64()?),
            12 => MetaValue::F64(self.f64()?),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("gguf: unknown metadata value type {other}"),
                ));
            }
        })
    }
}

fn align_up(x: usize, a: usize) -> usize {
    if a == 0 {
        x
    } else {
        (x + a - 1) / a * a
    }
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(&path)?;
        // Safety: the file is only read, never mutated while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> io::Result<Self> {
        let mut cur = Cursor::new(&mmap);
        if cur.take(4)? != b"GGUF" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "gguf: bad magic"));
        }
        let version = cur.u32()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gguf: unsupported version {version}"),
            ));
        }
        let n_tensors = cur.u64()? as usize;
        let n_kv = cur.u64()? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = cur.string()?;
            let ty = cur.u32()?;
            let val = cur.value(ty)?;
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::new();
        let mut tensor_order = Vec::new();
        for _ in 0..n_tensors {
            let name = cur.string()?;
            let n_dims = cur.u32()? as usize;
            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(cur.u64()? as usize);
            }
            let raw_ty = cur.u32()?;
            let offset = cur.u64()? as usize;
            let ty = GgmlType::from_u32(raw_ty).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("gguf: unsupported tensor type {raw_ty} for '{name}'"),
                )
            })?;
            tensor_order.push(name.clone());
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ty,
                    offset,
                },
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_usize())
            .unwrap_or(32);
        let data_offset = align_up(cur.pos, alignment);

        Ok(GgufFile {
            mmap,
            version,
            metadata,
            tensors,
            tensor_order,
            data_offset,
        })
    }

    /// Raw, potentially-quantized bytes of a tensor.
    pub fn tensor_bytes(&self, name: &str) -> io::Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor '{name}'")))?;
        let start = self.data_offset + info.offset;
        let end = start + info.nbytes();
        self.mmap
            .get(start..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, format!("tensor '{name}' out of range")))
    }

    /// Raw tensor bytes for an already-resolved [`TensorInfo`].
    ///
    /// This skips the hash-map lookup done by [`GgufFile::tensor_bytes`], which
    /// matters on the hot path where the same tensors are read every token.
    pub fn tensor_bytes_info(&self, info: &TensorInfo) -> Option<&[u8]> {
        let start = self.data_offset + info.offset;
        let end = start + info.nbytes();
        self.mmap.get(start..end)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rust_llama_gguf_{tag}_{}.bin", std::process::id()))
    }

    #[test]
    fn malformed_inputs_fail_without_abort() {
        // Empty file: bad magic, not a hang or panic.
        let p0 = tmp_path("empty");
        std::fs::write(&p0, b"").unwrap();
        assert!(GgufFile::open(&p0).is_err());
        // Absurd counts must fail fast (previously: capacity-overflow abort
        // inside `with_capacity` before any bounds check ran).
        let p1 = tmp_path("huge");
        let mut f = std::fs::File::create(&p1).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap(); // n_tensors
        f.write_all(&u64::MAX.to_le_bytes()).unwrap(); // n_kv
        drop(f);
        let t0 = std::time::Instant::now();
        assert!(GgufFile::open(&p1).is_err());
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "must fail fast");
        // Truncated string length likewise fails, not hangs.
        let p2 = tmp_path("trunc");
        let mut g = std::fs::File::create(&p2).unwrap();
        g.write_all(b"GGUF").unwrap();
        g.write_all(&3u32.to_le_bytes()).unwrap();
        g.write_all(&0u64.to_le_bytes()).unwrap();
        g.write_all(&1u64.to_le_bytes()).unwrap();
        g.write_all(&u64::MAX.to_le_bytes()).unwrap(); // key len: absurd
        drop(g);
        assert!(GgufFile::open(&p2).is_err());
        std::fs::remove_file(&p0).ok();
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }
}
