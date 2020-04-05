#![cfg_attr(feature = "const-generics", allow(incomplete_features))]
#![cfg_attr(feature = "const-generics", feature(const_generics))]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(rust_2018_idioms)]
#![doc(test(attr(forbid(
  unused_variables,
  unused_assignments,
  unused_mut,
  unused_attributes,
  dead_code
))))]
#![forbid(missing_debug_implementations, missing_docs)]

//! # ndsparse
//!
//! This crate provides structures to store and retrieve N-dimensional sparse data.

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod coo;
pub mod csl;
mod dims;
pub mod doc_tests;
mod utils;

pub use cl_traits::ArrayWrapper;
pub use dims::*;
#[cfg(feature = "with-rayon")]
pub use utils::{ParallelIteratorWrapper, ParallelProducerWrapper};
