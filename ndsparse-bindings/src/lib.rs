//! Basic structures to experiment with other languages. This crate is intended to be used
//! as a template for your own custom needs.
//!
//! The array support of these third-parties dependencies is minimum to non-existent, threfore,
//! the overhead of heap allocating.

use ndsparse::csl::Csl;
#[cfg(feature = "with-pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "with-wasm-bindgen")]
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
    #[cfg_attr(feature = "with-pyo3", pyclass)]
    #[cfg_attr(feature = "with-wasm-bindgen", wasm_bindgen)]
    #[derive(Debug)]
    pub struct $struct_name {
      csl: Csl<[usize; $dims], $data_storage, $indcs_storage, $offs_storage>,
    }

    // Generic

    #[cfg_attr(feature = "with-pyo3", pymethods)]
    #[cfg_attr(feature = "with-wasm-bindgen", wasm_bindgen)]
    impl $struct_name {
      pub fn clear(&mut self) {
        self.csl.clear()
      }

      pub fn data_vec(&self) -> Vec<$data_ty> {
        self.csl.data().to_vec()
      }

      pub fn indcs_vec(&self) -> Vec<usize> {
        self.csl.indcs().to_vec()
      }

      pub fn offs_vec(&self) -> Vec<usize> {
        self.csl.offs().to_vec()
      }

      pub fn nnz(&self) -> usize {
        self.csl.nnz()
      }
    }

    // PyO3

    #[cfg(feature = "with-pyo3")]
    #[pymethods]
    impl $struct_name {
      #[new]
      pub fn new(
        dims: [usize; $dims],
        data: $data_storage,
        indcs: $indcs_storage,
        offs: $offs_storage,
      ) -> Self {
        Self { csl: ndsparse::csl::Csl::new(dims, data, indcs, offs) }
      }

      pub fn truncate(&mut self, dims: [usize; $dims]) {
        self.csl.truncate(dims)
      }

      pub fn value(&self, dims: [usize; $dims]) -> Option<$data_ty> {
        self.csl.value(dims).copied()
      }
    }

    // wasm-bindgen

    #[cfg(feature = "with-wasm-bindgen")]
    #[wasm_bindgen]
    impl $struct_name {
      #[wasm_bindgen(constructor)]
      pub fn new_vec(
        dims_vec: Vec<usize>,
        data: $data_storage,
        indcs: $indcs_storage,
        offs: $offs_storage,
      ) -> Self {
        let dims: [usize; $dims] = from_vec_to_array(dims_vec);
        Self { csl: ndsparse::csl::Csl::new(dims, data, indcs, offs) }
      }

      pub fn dims_vec(&self) -> Vec<usize> {
        self.csl.dims().to_vec()
      }

      pub fn truncate_vec(&mut self, dims_vec: Vec<usize>) {
        self.csl.truncate(from_vec_to_array(dims_vec))
      }

      pub fn value_vec(&self, dims_vec: Vec<usize>) -> Option<$data_ty> {
        self.csl.value(from_vec_to_array(dims_vec)).copied()
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

#[cfg(feature = "with-wasm-bindgen")]
fn from_vec_to_array<A>(vec: Vec<A::Item>) -> A
where
  A: cl_traits::Array,
{
  assert!(vec.len() >= A::CAPACITY);
  let mut iter = vec.into_iter();
  cl_traits::create_array(|_| iter.next().unwrap())
}
