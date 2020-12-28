//! # ndsparse
//!
//! This crate provides structures to store and retrieve N-dimensional sparse data.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod coo;
pub mod csl;
pub mod doc_tests;
mod error;
mod utils;

/// Shorcut of core::result::Result<T, ndsparse::Error>;
pub type Result<T> = core::result::Result<T, Error>;

pub use error::*;
#[cfg(feature = "with-rayon")]
pub use utils::{ParallelIteratorWrapper, ParallelProducerWrapper};
