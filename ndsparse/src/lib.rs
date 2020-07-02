//! # ndsparse
//!
//! This crate provides structures to store and retrieve N-dimensional sparse data.

#![cfg_attr(feature = "const-generics", allow(incomplete_features))]
#![cfg_attr(feature = "const-generics", feature(const_generics))]
#![cfg_attr(not(feature = "std"), no_std)]
#![doc(test(attr(deny(warnings))))]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod coo;
pub mod csl;
mod dims;
pub mod doc_tests;
mod error;
mod utils;

/// Shorcut of core::result::Result<T, ndsparse::Error>;
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(feature = "with-rayon")]
pub use utils::{ParallelIteratorWrapper, ParallelProducerWrapper};
pub use {cl_traits::ArrayWrapper, dims::*, error::*};
