//! CSL (Compressed Sparse Line).
//!
//! A generalization of the [`CSC`]/[`CSR`] structures for N dimensions. Beware that this structure
//! doesn't make any distinction of what is a `column` or a `row` because the order of the elements
//! is up to the caller.
//!
//! [`CSC`]: en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)
//! [`CSR`]: en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

mod csl_iter;
mod csl_line_constructor;
#[cfg(all(test, feature = "with_rand"))]
mod csl_quickcheck;
#[cfg(feature = "with_rayon")]
mod csl_rayon;
mod csl_utils;

#[cfg(feature = "with_rand")]
mod csl_rnd;
use crate::utils::{are_in_ascending_order, are_in_upper_bound, does_not_have_duplicates};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use cl_traits::{ArrayWrapper, Clear, Push, Truncate};
use core::{marker::PhantomData, ops::Range};
pub use csl_iter::*;
pub use csl_line_constructor::*;
#[cfg(feature = "with_rayon")]
pub use csl_rayon::*;
use csl_utils::*;

/// CSL backed by a static array.
pub type CslArray<DATA, const DIMS: usize, const NNZ: usize, const OFFS: usize> =
  Csl<DATA, ArrayWrapper<DATA, NNZ>, ArrayWrapper<usize, NNZ>, ArrayWrapper<usize, OFFS>, DIMS>;
#[cfg(feature = "with_arrayvec")]
/// CSL backed by the `ArrayVec` dependency.
pub type CslArrayVec<DATA, const DIMS: usize, const NNZ: usize, const OFFS: usize> = Csl<
  DATA,
  cl_traits::ArrayVecArrayWrapper<DATA, NNZ>,
  cl_traits::ArrayVecArrayWrapper<usize, NNZ>,
  cl_traits::ArrayVecArrayWrapper<usize, OFFS>,
  DIMS,
>;
/// Mutable CSL reference.
pub type CslMut<'a, DATA, const DIMS: usize> =
  Csl<DATA, &'a mut [DATA], &'a [usize], &'a [usize], DIMS>;
/// Immutable CSL reference.
pub type CslRef<'a, DATA, const DIMS: usize> =
  Csl<DATA, &'a [DATA], &'a [usize], &'a [usize], DIMS>;
#[cfg(feature = "with_smallvec")]
/// CSL backed by the `SmallVec` dependency.
pub type CslSmallVec<DATA, const DIMS: usize, const NNZ: usize, const OFFS: usize> = Csl<
  DATA,
  cl_traits::SmallVecArrayWrapper<DATA, NNZ>,
  cl_traits::SmallVecArrayWrapper<usize, NNZ>,
  cl_traits::SmallVecArrayWrapper<usize, OFFS>,
  DIMS,
>;
#[cfg(feature = "with_staticvec")]
/// CSL backed by the `StaticVec` dependency
pub type CslStaticVec<DATA, const DIMS: usize, const NNZ: usize, const OFFS: usize> = Csl<
  DATA,
  staticvec::StaticVec<DATA, NNZ>,
  staticvec::StaticVec<usize, NNZ>,
  staticvec::StaticVec<usize, OFFS>,
  DIMS,
>;
#[cfg(feature = "alloc")]
/// CSL backed by a dynamic vector.
pub type CslVec<DATA, const DIMS: usize> = Csl<DATA, Vec<DATA>, Vec<usize>, Vec<usize>, DIMS>;

/// Base structure for all CSL* variants.
///
/// It is possible to define your own fancy CSL, e.g.,
/// `Csl<BigNum, [BigNum; 32], ArrayVec<[usize; 32]>, StaticVec<usize, 123>, 321>`.
#[cfg_attr(feature = "with_serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Csl<DATA, DS, IS, OS, const DIMS: usize> {
  // DS (Data Storage): Container that stores DATA
  pub(crate) data: DS,
  pub(crate) dims: ArrayWrapper<usize, DIMS>,
  // IS (Indices Storage): Container that stores CSL indices
  pub(crate) indcs: IS,
  // OS (Offsets Storage): Container that stores CSL offsets
  pub(crate) offs: OS,
  // The compiler doesn't infer DATA when DS = [DATA; NNZ], threfore, this field
  pub(crate) phantom: PhantomData<DATA>,
}

impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS> {
  /// The definitions of all dimensions.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(csl_array_4().dims(), &[2, 3, 4, 5]);
  /// ```
  #[inline]
  pub fn dims(&self) -> &[usize; DIMS] {
    &*self.dims
  }
}

impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS>
where
  DS: AsRef<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  /// Creates a valid CSL instance.
  ///
  /// The compressed fields are a bit complex and unless you really know what you are doing, this
  /// method shouldn't probably be used directly. Please, try to consider using [`#constructor`]
  /// instead.
  ///
  /// # Arguments
  ///
  /// * `into_dims`: Array of dimensions
  /// * `into_data`: Data collection
  /// * `into_indcs`: Indices of each data item
  /// * `into_offs`: Offset of each innermost line
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::csl::{Csl, CslVec};
  /// // A sparse array ([8, _, _, _, _, 9, _, _, _, _])
  /// let mut _sparse_array: Csl<f64, [f64; 2], [usize; 2], [usize; 2], 1>;
  /// _sparse_array = Csl::new([10], [8.0, 9.0], [0, 5], [0, 2]);
  /// // A bunch of nothing for your overflow needs
  /// let mut _over_nine: CslVec<(), 9001>;
  /// _over_nine = CslVec::new([0; 9001], vec![], vec![], vec![]);
  /// ```
  ///
  /// # Assertions
  ///
  /// * Innermost dimensions length must be greater than zero
  /// ```rust,should_panic
  /// use ndsparse::csl::CslVec;
  /// let _: CslVec<i32, 7> = CslVec::new([1, 2, 3, 4, 5, 0, 7], vec![], vec![], vec![]);
  /// ```
  ///
  /// * The data length must equal the indices length
  /// ```rust,should_panic
  /// use ndsparse::csl::{ CslVec};
  /// let _ = CslVec::new([10], vec![8, 9], vec![0], vec![0, 2]);
  /// ```
  ///
  /// * Offsets must be in ascending order
  /// ```rust,should_panic
  /// use ndsparse::csl::CslArray;
  /// let _ = CslArray::new([10], [8, 9], [0, 5], [2, 0]);
  /// ```
  ///
  /// * Offsets length must equal the dimensions product (without the innermost dimension) plus one
  /// ```rust,should_panic
  /// use ndsparse::csl::CslVec;
  /// let _ = CslVec::new([10], vec![8, 9], vec![0, 5], vec![0, 2, 4]);
  /// ```
  ///
  /// * Indices of a line must be unique
  /// ```rust,should_panic
  /// use ndsparse::csl::CslArray;
  /// let _ = CslArray::new([10], [8, 9], [0, 0], [0, 2]);
  /// ```
  ///
  /// * The data and indices length must be equal or less than the product of all dimensions length
  /// ```rust,should_panic
  /// use ndsparse::csl::CslVec;
  /// let _ = CslVec::new([10], vec![8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], vec![0, 5], vec![0, 2]);
  /// ```
  ///
  /// * Last offset must equal the nnz
  /// ```rust,should_panic
  /// use ndsparse::csl::CslArray;
  /// let _ = CslArray::new([10], [8, 9], [0, 5], [0, 4]);
  /// ```
  ///
  /// * The indices must be less than the innermost dimension length
  /// ```rust,should_panic
  /// use ndsparse::csl::CslArray;
  /// let _ = CslArray::new([10], [8, 9], [0, 10], [0, 2]);
  /// ```
  pub fn new<ID, IDS, IIS, IOS>(
    into_dims: ID,
    into_data: IDS,
    into_indcs: IIS,
    into_offs: IOS,
  ) -> Self
  where
    ID: Into<ArrayWrapper<usize, DIMS>>,
    IDS: Into<DS>,
    IIS: Into<IS>,
    IOS: Into<OS>,
  {
    let data = into_data.into();
    let dims = into_dims.into();
    let indcs = into_indcs.into();
    let offs = into_offs.into();
    let data_ref = data.as_ref();
    let indcs_ref = indcs.as_ref();
    let offs_ref = offs.as_ref();
    assert!(
      {
        let mut is_valid = true;
        if let Some(idx) = dims.iter().position(|dim| *dim != 0) {
          is_valid = dims[idx..].iter().all(|dim| *dim != 0);
        }
        is_valid
      },
      "Innermost dimensions length must be greater than zero"
    );
    assert!(data_ref.len() == indcs_ref.len(), "The data length must equal the indices length");
    assert!(are_in_ascending_order(&offs_ref, |a, b| [a, b]), "Offsets must be in ascending order");
    assert!(
      {
        let max_nnz = max_nnz(&dims);
        data_ref.len() <= max_nnz && indcs_ref.len() <= max_nnz
      },
      "The data and indices length must be equal or less than the product of all dimensions length"
    );
    if let Some(first) = offs_ref.get(0) {
      assert!(
        offs_ref.windows(2).all(|x| {
          let range = x[0] - first..x[1] - first;
          does_not_have_duplicates(&indcs_ref[range])
        }),
        "Indices of a line must be unique"
      );
    }
    if let Some(last_ref) = offs_ref.last() {
      let last = last_ref - offs_ref[0];
      assert!(last == data_ref.len() && last == indcs_ref.len(), "Last offset must equal the nnz");
    }
    if let Some(last) = dims.last() {
      let are_in_upper_bound = are_in_upper_bound(indcs_ref, last);
      assert!(are_in_upper_bound, "The indices must be less than the innermost dimension length");
      assert!(
        offs_ref.len() == offs_len(&dims),
        "Non-empty offsets length must equal the dimensions product (without the innermost \
         dimension) plus one"
      );
    }
    Self { data, dims, indcs, offs, phantom: PhantomData }
  }

  /// The data that is being stored.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(csl_array_4().data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
  /// ```
  pub fn data(&self) -> &[DATA] {
    self.data.as_ref()
  }

  /// Indices (indcs) of a line, i.e., indices of the innermost dimension.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(csl_array_4().indcs(), &[0, 3, 1, 3, 4, 2, 2, 4, 2]);
  /// ```
  pub fn indcs(&self) -> &[usize] {
    self.indcs.as_ref()
  }

  /// Any immutable line reference determined by `indcs`. The innermost dimension is ignored.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use ndsparse::{csl::CslRef, doc_tests::csl_array_4};
  /// let csl = csl_array_4();
  /// assert_eq!(csl.line([0, 0, 2, 0]), Some(CslRef::new([5], &[][..], &[][..], &[3, 3][..])));
  /// assert_eq!(csl.line([0, 1, 0, 0]), Some(CslRef::new([5], &[6][..], &[2][..], &[5, 6][..])));
  /// ```
  pub fn line(&self, indcs: [usize; DIMS]) -> Option<CslRef<'_, DATA, 1>> {
    line(self, indcs)
  }

  /// Number of NonZero elements.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(csl_array_4().nnz(), 9);
  /// ```
  #[inline]
  pub fn nnz(&self) -> usize {
    self.data.as_ref().len()
  }

  /// The joining of two consecutives offsets (offs) represent the starting and ending points of a
  /// line in the `data` and `indcs` slices.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(
  ///   csl_array_4().offs(),
  ///   &[0, 2, 3, 3, 5, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
  /// );
  /// ```
  pub fn offs(&self) -> &[usize] {
    self.offs.as_ref()
  }

  /// Iterator that returns immutable references of the outermost dimension
  ///
  /// # Examples
  ///
  /// ```rust
  /// use ndsparse::{csl::CslRef, doc_tests::csl_array_4};
  /// let csl = csl_array_4();
  /// let sub_csl = csl.sub_dim(0..3);
  /// let mut iter = sub_csl.outermost_iter();
  /// assert_eq!(
  ///   iter.next().unwrap(),
  ///   CslRef::new([1, 4, 5], &[1, 2, 3, 4, 5][..], &[0, 3, 1, 3, 4][..], &[0, 2, 3, 3, 5][..])
  /// );
  /// assert_eq!(
  ///   iter.next().unwrap(),
  ///   CslRef::new([1, 4, 5], &[6][..], &[2][..], &[5, 6, 6, 6, 6][..])
  /// );
  /// assert_eq!(
  ///   iter.next().unwrap(),
  ///   CslRef::new([1, 4, 5], &[7, 8][..], &[2, 4][..], &[6, 7, 8, 8, 8][..])
  /// );
  /// assert_eq!(iter.next(), None);
  /// ```
  ///
  /// # Assertions
  ///
  /// * `DIMS` must be greater than 1
  /// ```rust,should_panic
  /// use ndsparse::csl::CslVec;
  /// let _ = CslVec::<i32, 1>::default().outermost_iter();
  /// ```
  pub fn outermost_iter(&self) -> CsIterRef<'_, DATA, DIMS> {
    CsIterRef::new(&self.dims, self.data.as_ref().as_ptr(), self.indcs.as_ref(), self.offs.as_ref())
  }

  /// Parallel iterator that returns all immutable references of the current dimension
  /// using `rayon`.
  ///
  /// # Examples
  ///
  /// ```rust,
  /// use ndsparse::doc_tests::csl_array_4;
  /// use rayon::prelude::*;
  /// let csl = csl_array_4();
  /// let outermost_rayon_iter = csl.outermost_rayon_iter();
  /// outermost_rayon_iter.enumerate().for_each(|(idx, csl_ref)| {
  ///   assert_eq!(csl_ref, csl.outermost_iter().nth(idx).unwrap());
  /// });
  /// ```
  ///
  /// # Assertions
  ///
  /// * `DIMS` must be greater than 1
  /// ```rust,should_panic
  /// use ndsparse::csl::CslVec;
  /// let _ = CslVec::<i32, 1>::default().outermost_rayon_iter();
  /// ```
  #[cfg(feature = "with_rayon")]
  pub fn outermost_rayon_iter(&self) -> crate::ParallelIteratorWrapper<CsIterRef<'_, DATA, DIMS>> {
    crate::ParallelIteratorWrapper(self.outermost_iter())
  }

  /// Retrieves an immutable reference of any sub dimension.
  ///
  /// # Arguments
  ///
  /// * `const N`: Desired dimension
  /// * `range`: Starting and ending of the desired dimension
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::{csl::CslRef, doc_tests::csl_array_4};
  /// let csl = csl_array_4();
  /// // The last cuboid
  /// assert_eq!(
  ///   csl.sub_dim(1..2),
  ///   CslRef::new([1, 3, 4, 5], &[9][..], &[2][..], &[8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9][..])
  /// );
  /// // The last 2 matrices of the first cuboid;
  /// assert_eq!(
  ///   csl.sub_dim(1..3),
  ///   CslRef::new([2, 4, 5], &[6, 7, 8][..], &[2, 2, 4][..], &[5, 6, 6, 6, 6, 7, 8, 8, 8][..])
  /// );
  /// ```
  pub fn sub_dim<const N: usize>(&self, range: Range<usize>) -> CslRef<'_, DATA, N> {
    assert!(N <= DIMS);
    sub_dim(self, range)
  }

  /// Retrieves an immutable reference of a single data value.
  ///
  /// # Arguments
  ///
  /// * `indcs`: Indices of all dimensions
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(csl_array_4().value([1, 0, 2, 2]), Some(&9));
  /// ```
  /// 
  /// # Assertions
  ///
  /// * `indcs` must be within dimensions bounds
  /// ```rust,should_panic
  /// use ndsparse::doc_tests::csl_array_4;
  /// let _ = csl_array_4().value([9, 9, 9, 9]);
  /// ```
  pub fn value(&self, indcs: [usize; DIMS]) -> Option<&DATA> {
    data_idx(self, indcs).map(|idx| &self.data.as_ref()[idx])
  }
}

impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS>
where
  DS: AsMut<[DATA]> + AsRef<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  /// Mutable version of [`data`](#method.data).
  pub fn data_mut(&mut self) -> &mut [DATA] {
    self.data.as_mut()
  }

  /// Mutable version of [`line`](#method.line).
  pub fn line_mut(&mut self, indcs: [usize; DIMS]) -> Option<CslMut<'_, DATA, 1>> {
    line_mut(self, indcs)
  }

  /// Mutable version of [`outermost_iter`](#method.outermost_iter).
  pub fn outermost_iter_mut(&mut self) -> CslIterMut<'_, DATA, DIMS> {
    CslIterMut::new(
      &self.dims,
      self.data.as_mut().as_mut_ptr(),
      self.indcs.as_ref(),
      self.offs.as_ref(),
    )
  }

  /// Mutable version of [`outermost_rayon_iter`](#method.outermost_rayon_iter).
  #[cfg(feature = "with_rayon")]
  pub fn outermost_rayon_iter_mut(
    &mut self,
  ) -> crate::ParallelIteratorWrapper<CslIterMut<'_, DATA, DIMS>> {
    crate::ParallelIteratorWrapper(self.outermost_iter_mut())
  }

  /// Mutable version of [`sub_dim`](#method.sub_dim).
  pub fn sub_dim_mut<const N: usize>(&mut self, range: Range<usize>) -> CslMut<'_, DATA, N> {
    sub_dim_mut(self, range)
  }

  /// Mutable version of [`value`](#method.value).
  pub fn value_mut(&mut self, indcs: [usize; DIMS]) -> Option<&mut DATA> {
    data_idx(self, indcs).map(move |idx| &mut self.data.as_mut()[idx])
  }
}

impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS>
where
  DS: AsRef<[DATA]> + Push<Input = DATA>,
  IS: AsRef<[usize]> + Push<Input = usize>,
  OS: AsRef<[usize]> + Push<Input = usize>,
{
  /// See [`CslLineConstructor`](CslLineConstructor) for more information.
  pub fn constructor(&mut self) -> CslLineConstructor<'_, DATA, DS, IS, OS, DIMS> {
    CslLineConstructor::new(self)
  }
}

#[cfg(feature = "with_rand")]
impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS>
where
  DATA: Default,
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Default + Push<Input = DATA>,
  IS: AsMut<[usize]> + AsRef<[usize]> + Default + Push<Input = usize>,
  OS: AsMut<[usize]> + AsRef<[usize]> + Default + Push<Input = usize>,
{
  /// Creates a new random and valid instance delimited by the passed arguments.
  ///
  /// # Arguments
  ///
  /// * `into_dims`: Array of dimensions
  /// * `nnz`: Number of Non-Zero elements
  /// * `rng`: `rand::Rng` trait
  /// * `cb`: Callback to control data creation
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::csl::CslVec;
  /// use rand::{thread_rng, Rng};
  /// let mut _random: CslVec<u8, 8>;
  /// let mut rng = thread_rng();
  /// _random = CslVec::new_random_with_rand([1, 2, 3, 4, 5, 6, 7, 8], 9, &mut rng, |r, _| r.gen());
  /// ```
  pub fn new_random_with_rand<F, ID, R>(into_dims: ID, nnz: usize, rng: &mut R, cb: F) -> Self
  where
    F: FnMut(&mut R, [usize; DIMS]) -> DATA,
    ID: Into<ArrayWrapper<usize, DIMS>>,
    R: rand::Rng,
  {
    let dims = into_dims.into();
    let mut csl = Self::default();
    csl.dims = dims;
    csl_rnd::CslRnd::new(&mut csl, nnz, rng).fill(cb);
    Csl::new(csl.dims, csl.data, csl.indcs, csl.offs)
  }
}

impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS>
where
  DS: Clear,
  IS: Clear,
  OS: Clear,
{
  /// Clears all values and dimensions.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::{csl::CslVec, doc_tests::csl_vec_4};
  /// let mut csl = csl_vec_4();
  /// csl.clear();
  /// assert_eq!(csl, CslVec::default());
  /// ```
  pub fn clear(&mut self) {
    self.dims = Default::default();
    self.data.clear();
    self.indcs.clear();
    self.offs.clear();
  }
}

impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS>
where
  DS: AsMut<[DATA]> + AsRef<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  /// Intra-swap a single data value.
  ///
  /// # Arguments
  ///
  /// * `a`: First set of indices
  /// * `b`: SEcodn set of indices
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_vec_4;
  /// let mut csl = csl_vec_4();
  /// csl.swap_value([0, 0, 0, 0], [1, 0, 2, 2]);
  /// assert_eq!(csl.data(), &[9, 2, 3, 4, 5, 6, 7, 8, 1]);
  /// ```
  ///
  /// # Assertions
  ///
  /// Uses the same assertions of [`value`](#method.value).
  pub fn swap_value(&mut self, a: [usize; DIMS], b: [usize; DIMS]) -> bool {
    assert!(a[..] < self.dims[..] && b[..] < self.dims[..]);
    if let Some(a_idx) = data_idx(self, a) {
      if let Some(b_idx) = data_idx(self, b) {
        self.data.as_mut().swap(a_idx, b_idx);
        return true;
      }
    }
    false
  }
}

impl<DATA, DS, IS, OS, const DIMS: usize> Csl<DATA, DS, IS, OS, DIMS>
where
  DS: Truncate<Input = usize>,
  IS: Truncate<Input = usize>,
  OS: AsRef<[usize]> + Push<Input = usize> + Truncate<Input = usize>,
{
  /// Truncates all values in the exactly exclusive point defined by `indcs`.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::{csl::CslVec, doc_tests::csl_vec_4};
  /// let mut csl = csl_vec_4();
  /// csl.truncate([0, 0, 3, 4]);
  /// assert_eq!(
  ///   csl,
  ///   CslVec::new([0, 0, 4, 5], vec![1, 2, 3, 4], vec![0, 3, 1, 3], vec![0, 2, 3, 3, 4])
  /// );
  /// ```
  pub fn truncate(&mut self, indcs: [usize; DIMS]) {
    if let Some([offs_indcs, values]) = line_offs(&self.dims, &indcs, self.offs.as_ref()) {
      let cut_point = values.start + 1;
      self.data.truncate(cut_point);
      self.indcs.truncate(cut_point);
      self.offs.truncate(offs_indcs.start + 1);
      self.offs.push(*indcs.last().unwrap());
      indcs.iter().zip(self.dims.iter_mut()).filter(|(a, _)| **a == 0).for_each(|(_, b)| *b = 0);
    }
  }
}
