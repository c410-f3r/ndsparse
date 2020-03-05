#![allow(incomplete_features)]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(rust_2018_idioms)]
#![doc(test(attr(forbid(
  unused_variables,
  unused_assignments,
  unused_mut,
  unused_attributes,
  dead_code
))))]
#![feature(const_generics)]
#![forbid(missing_debug_implementations, missing_docs)]

//! # ndsparse
//!
//! This crate provides structures to store and retrieve N-dimensional sparse data.

#[cfg(any(feature = "alloc", feature = "std"))]
extern crate alloc;

pub mod coo;
pub mod csl;
pub mod doc_tests;
mod utils;

pub use cl_traits::ArrayWrapper;
#[cfg(feature = "with_rayon")]
pub use utils::{ParallelIteratorWrapper, ParallelProducerWrapper};
