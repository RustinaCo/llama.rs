//! A view over one GGUF tensor: rows of quantized bytes with row-wise access.

use crate::gguf::TensorInfo;
use crate::quant::{self, GgmlType};

pub struct Tensor<'a> {
    pub info: &'a TensorInfo,
    pub data: &'a [u8],
}

impl<'a> Tensor<'a> {
    pub fn new(info: &'a TensorInfo, data: &'a [u8]) -> Self {
        Tensor { info, data }
    }

    pub fn ty(&self) -> GgmlType {
        self.info.ty
    }

    pub fn n_rows(&self) -> usize {
        self.info.n_rows()
    }

    pub fn n_cols(&self) -> usize {
        self.info.n_cols()
    }

    pub fn row_bytes(&self) -> usize {
        self.info.ty.row_size(self.n_cols())
    }

    pub fn row_slice(&self, row: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.data[row * rb..row * rb + rb]
    }

    /// Zero-allocation view of an F32 row (only if this tensor is plain F32).
    pub fn f32_slice(&self) -> Option<&'a [f32]> {
        if self.info.ty != GgmlType::F32 {
            return None;
        }
        let n = self.info.n_elements();
        // Alignment is guaranteed by GGUF alignment (>= 32) in practice; guard anyway.
        let ptr = self.data.as_ptr();
        if (ptr as usize) % std::mem::align_of::<f32>() != 0 {
            return None;
        }
        // Safety: we checked the type (4 bytes/elem), length, and alignment.
        Some(unsafe { std::slice::from_raw_parts(ptr as *const f32, n) })
    }

    pub fn f32_row(&self, row: usize) -> Option<&'a [f32]> {
        let all = self.f32_slice()?;
        let c = self.n_cols();
        Some(&all[row * c..row * c + c])
    }

    /// Dequantize row `row` into `out` (`out.len()` must equal `n_cols`).
    pub fn dequant_row(&self, row: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.n_cols());
        quant::dequant_row(self.info.ty, self.row_slice(row), out);
    }

    /// Fully materialize the tensor as `f32` (used for small tensors like norms).
    pub fn to_f32(&self) -> Vec<f32> {
        if let Some(s) = self.f32_slice() {
            return s.to_vec();
        }
        let n = self.info.n_elements();
        let mut out = vec![0f32; n];
        let c = self.n_cols();
        let rows = self.n_rows();
        for r in 0..rows {
            self.dequant_row(r, &mut out[r * c..r * c + c]);
        }
        out
    }
}
