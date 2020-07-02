//! Dynamic arrays

#![allow(clippy::panic)]

use ndsparse::csl::{Csl, CslRef};

/// CslArrayVec
pub type CslArrayVec<DA, DTA, IA, OA> = Csl<
  DA,
  arrayvec::ArrayVec<cl_traits::ArrayWrapper<DTA>>,
  arrayvec::ArrayVec<cl_traits::ArrayWrapper<IA>>,
  arrayvec::ArrayVec<cl_traits::ArrayWrapper<OA>>,
>;

/// CslSmallVec
pub type CslSmallVec<DA, DTA, IA, OA> = Csl<
  DA,
  smallvec::SmallVec<cl_traits::ArrayWrapper<DTA>>,
  smallvec::SmallVec<cl_traits::ArrayWrapper<IA>>,
  smallvec::SmallVec<cl_traits::ArrayWrapper<OA>>,
>;

fn main() -> ndsparse::Result<()> {
  let mut array_vec = CslArrayVec::<[usize; 2], [i32; 5], [usize; 5], [usize; 26]>::default();
  let mut small_vec = CslSmallVec::<[usize; 2], [i32; 5], [usize; 5], [usize; 26]>::default();
  array_vec.constructor()?.next_outermost_dim(5)?.push_line([(0, 1), (3, 2)].iter().copied())?;
  small_vec.constructor()?.next_outermost_dim(5)?.push_line([(0, 1), (3, 2)].iter().copied())?;
  assert!(array_vec.line([0, 0]) == Some(CslRef::new([5], &[1, 2][..], &[0, 3][..], &[0, 2][..])?));
  assert!(array_vec.line([0, 0]) == small_vec.line([0, 0]));
  Ok(())
}
