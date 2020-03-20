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
#[cfg(all(test, feature = "alloc", feature = "with_rand"))]
mod csl_quickcheck;
#[cfg(feature = "with_rayon")]
mod csl_rayon;
mod csl_utils;

#[cfg(feature = "with_rand")]
mod csl_rnd;
use crate::{
  utils::{are_in_ascending_order, are_in_upper_bound, does_not_have_duplicates, max_nnz},
  Dims,
};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use cl_traits::{ArrayWrapper, Clear, Push, Storage, Truncate, WithCapacity};
use core::ops::Range;
pub use csl_iter::*;
pub use csl_line_constructor::*;
#[cfg(feature = "with_rayon")]
pub use csl_rayon::*;
use csl_utils::*;

/// CSL backed by a static array.
///
/// * Types
///
/// * `DA`: Dimensions Array
/// * `DTA` DaTa Array
/// * `IA`: Indices Array
/// * `OA`: Offsets Array
pub type CslArray<DA, DTA, IA, OA> = Csl<DA, ArrayWrapper<DTA>, ArrayWrapper<IA>, ArrayWrapper<OA>>;
#[cfg(feature = "with_arrayvec")]
/// CSL backed by the `ArrayVec` dependency.
pub type CslArrayVec<DA, DTA, IA, OA> = Csl<
  DA,
  cl_traits::ArrayVecArrayWrapper<DTA>,
  cl_traits::ArrayVecArrayWrapper<IA>,
  cl_traits::ArrayVecArrayWrapper<OA>,
>;
/// Mutable CSL reference.
pub type CslMut<'a, DA, DATA> = Csl<DA, &'a mut [DATA], &'a [usize], &'a [usize]>;
/// Immutable CSL reference.
pub type CslRef<'a, DA, DATA> = Csl<DA, &'a [DATA], &'a [usize], &'a [usize]>;
#[cfg(feature = "with_smallvec")]
/// CSL backed by the `SmallVec` dependency.
///
///
/// * Types
///
/// * `DA`: Dimensions Array
/// * `DTA` DaTa Array
/// * `IA`: Indices Array
/// * `OA`: Offsets Array
pub type CslSmallVec<DA, DTA, IA, OA> = Csl<
  DA,
  cl_traits::SmallVecArrayWrapper<DTA>,
  cl_traits::SmallVecArrayWrapper<IA>,
  cl_traits::SmallVecArrayWrapper<OA>,
>;
#[cfg(feature = "with_staticvec")]
/// CSL backed by the `StaticVec` dependency
pub type CslStaticVec<DATA, const DIMS: usize, const NNZ: usize, const OFFS: usize> = Csl<
  [usize; DIMS],
  staticvec::StaticVec<DATA, NNZ>,
  staticvec::StaticVec<usize, NNZ>,
  staticvec::StaticVec<usize, OFFS>,
>;
#[cfg(feature = "alloc")]
/// CSL backed by a dynamic vector.
pub type CslVec<DA, DATA> = Csl<DA, Vec<DATA>, Vec<usize>, Vec<usize>>;

/// Base structure for all CSL* variants.
///
/// It is possible to define your own fancy CSL, e.g.,
/// `Csl<[BigNum; 32], ArrayVec<[usize; 32]>, StaticVec<usize, 123>, 321>`.
///
/// # Types
///
/// * `DS`: Data Storage
/// * `IS`: Indices Storage
/// * `OS`: Offsets Storage
/// * `const DIMS: usize`: Dimensions length
#[cfg_attr(feature = "with_serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Csl<DA, DS, IS, OS>
where
  DA: Dims,
{
  pub(crate) data: DS,
  pub(crate) dims: ArrayWrapper<DA>,
  pub(crate) indcs: IS,
  pub(crate) offs: OS,
}

impl<DA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
  DS: WithCapacity<Input = usize>,
  IS: WithCapacity<Input = usize>,
  OS: WithCapacity<Input = usize>,
{
  /// Creates an empty instance with initial capacity.
  ///
  /// For storages involving solely arrays, all arguments will be discarted.
  ///
  /// # Arguments
  ///
  /// * `nnz`: Number of Non-Zero elements
  /// * `nolp1`: Number Of Lines Plus 1, i.e., the dimensions product
  /// (without the innermost dimension) plus 1
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// let dims = [11, 10, 1];
  /// let nolp1 = dims.iter().rev().skip(1).product::<usize>() + 1;
  /// let nnz = 2;
  /// let _ = CslVec::<[usize; 3], i32>::with_capacity(nnz, nolp1);
  /// ```
  pub fn with_capacity(nnz: usize, nolp1: usize) -> Self {
    Self {
      data: DS::with_capacity(nnz),
      dims: Default::default(),
      indcs: IS::with_capacity(nnz),
      offs: OS::with_capacity(nolp1),
    }
  }
}

impl<DA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
{
  /// The definitions of all dimensions.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(csl_array_4().dims(), &[2, 3, 4, 5]);
  /// ```
  #[inline]
  pub fn dims(&self) -> &DA {
    &self.dims
  }
}

impl<DA, DATA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
  DS: AsRef<[DATA]> + Storage<Item = DATA>,
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
  #[cfg_attr(all(feature = "alloc", feature = "const_generics"), doc = "```rust")]
  #[cfg_attr(not(all(feature = "alloc", feature = "const_generics")), doc = "```ignore")]
  /// use ndsparse::csl::{CslArray, CslVec};
  /// // Sparse array ([8, _, _, _, _, 9, _, _, _, _])
  /// let mut _sparse_array = CslArray::new([10], [8.0, 9.0], [0, 5], [0, 2]);
  /// // A bunch of nothing for your overflow needs
  /// let mut _over_nine: CslVec<[usize; 9001], ()>;
  /// _over_nine = CslVec::new([0; 9001], vec![], vec![], vec![]);
  /// ```
  ///
  /// # Assertions
  ///
  /// * Innermost dimensions length must be greater than zero
  #[cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// let _: CslVec<[usize; 7], i32> = CslVec::new([1, 2, 3, 4, 5, 0, 7], vec![], vec![], vec![]);
  /// ```
  ///
  /// * The data length must equal the indices length
  #[cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
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
  #[cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
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
  #[cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
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
    ID: Into<ArrayWrapper<DA>>,
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
        if let Some(idx) = dims.slice().iter().position(|dim| *dim != 0) {
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
    if let Some(last) = dims.slice().last() {
      let are_in_upper_bound = are_in_upper_bound(indcs_ref, last);
      assert!(are_in_upper_bound, "The indices must be less than the innermost dimension length");
      assert!(
        offs_ref.len() == offs_len(&dims),
        "Non-empty offsets length must equal the dimensions product (without the innermost \
         dimension) plus one"
      );
    }
    Self { data, dims, indcs, offs }
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
  pub fn line(&self, indcs: DA) -> Option<CslRef<'_, [usize; 1], DATA>> {
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
  /// use ndsparse::csl::CslArray;
  /// let _ = CslArray::<[usize; 0], [i32; 1], [usize; 1], [usize; 2]>::default().outermost_iter();
  /// ```
  pub fn outermost_iter(&self) -> CsIterRef<'_, DA, DATA> {
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
  /// use ndsparse::csl::CslArray;
  /// let _csl = CslArray::<[usize; 0], [i32; 1], [usize; 1], [usize; 2]>::default();
  /// _csl.outermost_rayon_iter();
  /// ```
  #[cfg(feature = "with_rayon")]
  pub fn outermost_rayon_iter(&self) -> crate::ParallelIteratorWrapper<CsIterRef<'_, DA, DATA>> {
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
  pub fn sub_dim<TODA>(&self, range: Range<usize>) -> CslRef<'_, TODA, DATA>
  where
    TODA: Dims,
  {
    assert!(TODA::CAPACITY <= DA::CAPACITY);
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
  pub fn value(&self, indcs: DA) -> Option<&DATA> {
    data_idx(self, indcs).map(|idx| &self.data.as_ref()[idx])
  }
}

impl<DA, DATA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Storage<Item = DATA>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  /// Mutable version of [`data`](#method.data).
  pub fn data_mut(&mut self) -> &mut [DATA] {
    self.data.as_mut()
  }

  /// Mutable version of [`line`](#method.line).
  pub fn line_mut(&mut self, indcs: DA) -> Option<CslMut<'_, [usize; 1], DATA>> {
    line_mut(self, indcs)
  }

  /// Mutable version of [`outermost_iter`](#method.outermost_iter).
  pub fn outermost_iter_mut(&mut self) -> CslIterMut<'_, DA, DATA> {
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
  ) -> crate::ParallelIteratorWrapper<CslIterMut<'_, DA, DATA>> {
    crate::ParallelIteratorWrapper(self.outermost_iter_mut())
  }

  /// Mutable version of [`sub_dim`](#method.sub_dim).
  pub fn sub_dim_mut<TODA>(&mut self, range: Range<usize>) -> CslMut<'_, TODA, DATA>
  where
    TODA: Dims,
  {
    sub_dim_mut(self, range)
  }

  /// Mutable version of [`value`](#method.value).
  pub fn value_mut(&mut self, indcs: DA) -> Option<&mut DATA> {
    data_idx(self, indcs).map(move |idx| &mut self.data.as_mut()[idx])
  }
}

impl<DA, DATA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
  DS: AsRef<[DATA]> + Push<Input = DATA> + Storage<Item = DATA>,
  IS: AsRef<[usize]> + Push<Input = usize>,
  OS: AsRef<[usize]> + Push<Input = usize>,
{
  /// See [`CslLineConstructor`](CslLineConstructor) for more information.
  pub fn constructor(&mut self) -> CslLineConstructor<'_, DA, DS, IS, OS> {
    CslLineConstructor::new(self)
  }
}

#[cfg(feature = "with_rand")]
impl<DA, DATA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Default + Dims,
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Default + Push<Input = DATA> + Storage<Item = DATA>,
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
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// use rand::{thread_rng, Rng};
  /// let mut rng = thread_rng();
  /// let dims = [1, 2, 3, 4, 5, 6, 7, 8];
  /// let mut _random: CslVec<[usize; 8], u8>;
  /// _random = CslVec::new_controlled_random_with_rand(dims, 9, &mut rng, |r, _| r.gen());
  /// ```
  ///
  /// # Assertions
  ///
  /// * `nnz` must be equal or less than the maximum number of non-zero elements
  #[cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// use rand::{thread_rng, Rng};
  /// let mut rng = thread_rng();
  /// let dims = [1, 2, 3]; // Max of 6 elements (1 * 2 * 3)
  /// let mut _random: CslVec<[usize; 3], u8>;
  /// _random = CslVec::new_controlled_random_with_rand(dims, 7, &mut rng, |r, _| r.gen());
  /// ```
  pub fn new_controlled_random_with_rand<F, ID, R>(
    into_dims: ID,
    nnz: usize,
    rng: &mut R,
    cb: F,
  ) -> Self
  where
    F: FnMut(&mut R, DA) -> DATA,
    ID: Into<ArrayWrapper<DA>>,
    R: rand::Rng,
  {
    let dims = into_dims.into();
    assert!(
      nnz <= max_nnz(&dims),
      "`nnz` must be equal or less than the maximum number of non-zero elements"
    );
    let mut csl = Self::default();
    csl.dims = dims;
    csl_rnd::CslRnd::new(&mut csl, nnz, rng).fill(cb);
    Self::new(csl.dims, csl.data, csl.indcs, csl.offs)
  }

  /// Creates a new random and valid instance.
  ///
  /// # Arguments
  ///
  /// * `rng`: `rand::Rng` trait
  /// * `upper_bound`: The maximum allowed exclusive dimension
  ///
  /// # Example
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// use rand::{seq::SliceRandom, thread_rng};
  /// let mut rng = thread_rng();
  /// let upper_bound = 5;
  /// let random: CslVec<[usize; 8], u8> = CslVec::new_random_with_rand(&mut rng, upper_bound);
  /// assert!(random.dims().choose(&mut rng).unwrap() < &upper_bound);
  /// ```
  pub fn new_random_with_rand<R>(rng: &mut R, upper_bound: usize) -> Self
  where
    R: rand::Rng,
    rand::distributions::Standard: rand::distributions::Distribution<DATA>,
  {
    let dims = crate::utils::valid_random_dims(rng, upper_bound);
    let max_nnz = max_nnz(&dims);
    let nnz = if max_nnz == 0 { 0 } else { rng.gen_range(0, max_nnz) };
    Self::new_controlled_random_with_rand(dims, nnz, rng, |rng, _| rng.gen())
  }
}

impl<DA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
  DS: Clear,
  IS: Clear,
  OS: Clear,
{
  /// Clears all values and dimensions.
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
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

impl<DATA, DA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Storage<Item = DATA>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  /// Intra-swap a single data value.
  ///
  /// # Arguments
  ///
  /// * `a`: First set of indices
  /// * `b`: Second set of indices
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::doc_tests::csl_vec_4;
  /// let mut csl = csl_vec_4();
  /// csl.swap_value([0, 0, 0, 0], [1, 0, 2, 2]);
  /// assert_eq!(csl.data(), &[9, 2, 3, 4, 5, 6, 7, 8, 1]);
  /// ```
  ///
  /// # Assertions
  ///
  /// Uses the same assertions of [`value`](#method.value).
  pub fn swap_value(&mut self, a: DA, b: DA) -> bool {
    assert!(a.slice()[..] < self.dims[..] && b.slice()[..] < self.dims[..]);
    if let Some(a_idx) = data_idx(self, a) {
      if let Some(b_idx) = data_idx(self, b) {
        self.data.as_mut().swap(a_idx, b_idx);
        return true;
      }
    }
    false
  }
}

impl<DA, DS, IS, OS> Csl<DA, DS, IS, OS>
where
  DA: Dims,
  DS: Truncate<Input = usize>,
  IS: Truncate<Input = usize>,
  OS: AsRef<[usize]> + Push<Input = usize> + Truncate<Input = usize>,
{
  /// Truncates all values in the exactly exclusive point defined by `indcs`.
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::{csl::CslVec, doc_tests::csl_vec_4};
  /// let mut csl = csl_vec_4();
  /// csl.truncate([0, 0, 3, 4]);
  /// assert_eq!(
  ///   csl,
  ///   CslVec::new([0, 0, 4, 5], vec![1, 2, 3, 4], vec![0, 3, 1, 3], vec![0, 2, 3, 3, 4])
  /// );
  /// ```
  pub fn truncate(&mut self, indcs: DA) {
    if let Some([offs_indcs, values]) = line_offs(&self.dims, &indcs, self.offs.as_ref()) {
      let cut_point = values.start + 1;
      self.data.truncate(cut_point);
      self.indcs.truncate(cut_point);
      self.offs.truncate(offs_indcs.start + 1);
      self.offs.push(*indcs.slice().last().unwrap());
      indcs
        .slice()
        .iter()
        .zip(self.dims.slice_mut().iter_mut())
        .filter(|(a, _)| **a == 0)
        .for_each(|(_, b)| *b = 0);
    }
  }
}
