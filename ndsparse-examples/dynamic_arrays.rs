//! Dynamic arrays

#![allow(
  // Run-time logic
  clippy::panic
)]

use ndsparse::csl::{Csl, CslRef};

/// CslArrayVec
pub type CslArrayVec<DATA, const D: usize, const NNZ: usize, const OFFS: usize> = Csl<
  arrayvec::ArrayVec<DATA, NNZ>,
  arrayvec::ArrayVec<usize, NNZ>,
  arrayvec::ArrayVec<usize, OFFS>,
  D,
>;

/// CslSmallVec
pub type CslSmallVec<DTA, IA, OA, const D: usize> =
  Csl<smallvec::SmallVec<DTA>, smallvec::SmallVec<IA>, smallvec::SmallVec<OA>, D>;

fn main() -> ndsparse::Result<()> {
  let mut array_vec = CslArrayVec::<i32, 2, 5, 26>::default();
  let mut small_vec = CslSmallVec::<[i32; 5], [usize; 5], [usize; 32], 2>::default();
  array_vec.constructor()?.next_outermost_dim(5)?.push_line([(0, 1), (3, 2)].iter().copied())?;
  small_vec.constructor()?.next_outermost_dim(5)?.push_line([(0, 1), (3, 2)].iter().copied())?;
  assert!(array_vec.line([0, 0]) == Some(CslRef::new([5], &[1, 2][..], &[0, 3][..], &[0, 2][..])?));
  assert!(array_vec.line([0, 0]) == small_vec.line([0, 0]));
  Ok(())
}
