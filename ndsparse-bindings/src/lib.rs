//! Basic structures to experiment with other languages. This crate is intended to be used
//! as a template for your own custom needs.

#![allow(incomplete_features)]
#![feature(const_generics)]

#[cfg(feature = "with_pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "with_wasm_bindgen")]
use wasm_bindgen::prelude::*;

macro_rules! create_csl {
  (
    $struct_name:ident,
    $data_ty:ty,
    $data_storage:ty,
    $indcs_storage:ty,
    $offs_storage:ty,
    $dims:literal
  ) => {
    #[cfg_attr(feature = "with_pyo3", pyclass)]
    #[cfg_attr(feature = "with_wasm_bindgen", wasm_bindgen)]
    #[derive(Debug)]
    pub struct $struct_name {
      csl: ndsparse::csl::Csl<$data_ty, $data_storage, $indcs_storage, $offs_storage, $dims>,
    }

    //#[cfg_attr(feature = "with_pyo3", pymethods)] -> https://github.com/PyO3/pyo3/issues/780
    impl $struct_name {
      //#[cfg_attr(feature = "with_pyo3", new)] -> https://github.com/PyO3/pyo3/issues/780
      pub fn new(
        dims: [usize; $dims],
        data: $data_storage,
        indcs: $indcs_storage,
        offs: $offs_storage,
      ) -> Self {
        Self { csl: ndsparse::csl::Csl::new(dims, data, indcs, offs) }
      }

      pub fn clear(&mut self) {
        self.csl.clear()
      }

      pub fn dims(&self) -> [usize; $dims] {
        *self.csl.dims()
      }

      pub fn nnz(&self) -> usize {
        self.csl.nnz()
      }

      pub fn truncate(&mut self, dims: [usize; $dims]) {
        self.csl.truncate(dims)
      }
    }

    #[cfg(feature = "with_pyo3")]
    #[pymethods]
    impl $struct_name {
      pub fn data_py(&self) -> PyResult<Vec<$data_ty>> {
        Ok(self.csl.data().to_vec())
      }

      pub fn indcs_py(&self) -> PyResult<Vec<usize>> {
        Ok(self.csl.indcs().to_vec())
      }

      pub fn offs_py(&self) -> PyResult<Vec<usize>> {
        Ok(self.csl.offs().to_vec())
      }
    }

    #[cfg_attr(feature = "with_wasm_bindgen", wasm_bindgen)]
    impl $struct_name {
      pub fn data(&self) -> Vec<$data_ty> {
        self.csl.data().to_vec()
      }

      pub fn indcs(&self) -> Vec<usize> {
        self.csl.indcs().to_vec()
      }

      pub fn offs(&self) -> Vec<usize> {
        self.csl.offs().to_vec()
      }
    }
  };
}

create_csl!(Csl0VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 0);
create_csl!(Csl1VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 1);
create_csl!(Csl2VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 2);
create_csl!(Csl3VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 3);
create_csl!(Csl4VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 4);
create_csl!(Csl5VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 5);
create_csl!(Csl6VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 6);
create_csl!(Csl7VecI32, i32, Vec<i32>, Vec<usize>, Vec<usize>, 7);

create_csl!(Csl0VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 0);
create_csl!(Csl1VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 1);
create_csl!(Csl2VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 2);
create_csl!(Csl3VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 3);
create_csl!(Csl4VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 4);
create_csl!(Csl5VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 5);
create_csl!(Csl6VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 6);
create_csl!(Csl7VecF64, f64, Vec<f64>, Vec<usize>, Vec<usize>, 7);
