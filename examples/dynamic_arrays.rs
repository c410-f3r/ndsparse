use ndsparse::csl::{CslArrayVec, CslRef, CslSmallVec, CslStaticVec};

fn main() {
  let mut array_vec = CslArrayVec::<[usize; 2], [i32; 5], [usize; 5], [usize; 26]>::default();
  let mut small_vec = CslSmallVec::<[usize; 2], [i32; 5], [usize; 5], [usize; 26]>::default();
  let mut static_vec = CslStaticVec::<i32, 2, 5, 26>::default();
  array_vec.constructor().next_outermost_dim(5).push_line(&[1, 2], &[0, 3]);
  small_vec.constructor().next_outermost_dim(5).push_line(&[1, 2], &[0, 3]);
  static_vec.constructor().next_outermost_dim(5).push_line(&[1, 2], &[0, 3]);
  assert!(array_vec.line([0, 0]) == Some(CslRef::new([5], &[1, 2][..], &[0, 3][..], &[0, 2][..])));
  assert!(array_vec.line([0, 0]) == small_vec.line([0, 0]));
  assert!(array_vec.line([0, 0]) == static_vec.line([0, 0]));
  assert!(small_vec.line([0, 0]) == static_vec.line([0, 0]));
}
