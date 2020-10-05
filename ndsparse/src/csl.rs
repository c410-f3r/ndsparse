//! CSL (Compressed Sparse Line).
//!
//! A generalization of the [`CSC`]/[`CSR`] structures for N dimensions. Beware that this structure
//! doesn't make any distinction of what is a `column` or a `row` because the order of the elements
//! is up to the caller.
//!
//! [`CSC`]: en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)
//! [`CSR`]: en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

mod csl_error;
mod csl_line_constructor;
mod csl_line_iter;
#[cfg(feature = "with-rayon")]
mod csl_rayon;
#[cfg(feature = "with-rand")]
mod csl_rnd;
mod csl_utils;

use crate::utils::{are_in_ascending_order, are_in_upper_bound, has_duplicates, max_nnz, windows2};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use cl_traits::{Clear, Push, Storage, Truncate, WithCapacity};
use core::ops::Range;
#[cfg(feature = "with-rayon")]
pub use csl_rayon::*;
use csl_utils::*;
pub use {csl_error::*, csl_line_constructor::*, csl_line_iter::*};

/// CSL backed by a static array.
pub type CslArray<DATA, const D: usize, const N: usize, const O: usize> =
  Csl<[DATA; N], [usize; N], [usize; O], D>;

/// CSL backed by a mutable slice
pub type CslMut<'a, DATA, const D: usize> = Csl<&'a mut [DATA], &'a [usize], &'a [usize], D>;

/// CSL backed by a slice
pub type CslRef<'a, DATA, const D: usize> = Csl<&'a [DATA], &'a [usize], &'a [usize], D>;

/// CSL backed by a dynamic vector.
#[cfg(feature = "alloc")]
pub type CslVec<DATA, const D: usize> = Csl<Vec<DATA>, Vec<usize>, Vec<usize>, D>;

/// Base structure for all CSL* variants.
///
/// It is possible to define your own fancy CSL, e.g., `Csl<
///   staticvec::StaticVec<num_bigint::BigNum, 32>,
///   arrayvec::ArrayVec<[usize; 32]>,
///   smallvec::SmallVec<[usize; 321]>,
///   123
/// >`.
///
/// # Types
///
/// * `DA`: Dimensions Array
/// * `DS`: Data Storage
/// * `IS`: Indices Storage
/// * `OS`: Offsets Storage
#[cfg_attr(feature = "with-serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Csl<DS, IS, OS, const D: usize> {
  pub(crate) data: DS,
  pub(crate) dims: [usize; D],
  pub(crate) indcs: IS,
  pub(crate) offs: OS,
}

impl<DS, IS, OS, const D: usize> Csl<DS, IS, OS, D>
where
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
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// let dims = [11, 10, 1];
  /// let nolp1 = dims.iter().rev().skip(1).product::<usize>() + 1;
  /// let nnz = 2;
  /// let _ = CslVec::<i32, 3>::with_capacity(nnz, nolp1);
  /// ```
  pub fn with_capacity(nnz: usize, nolp1: usize) -> Self {
    Self {
      data: DS::with_capacity(nnz),
      dims: crate::utils::default_array(),
      indcs: IS::with_capacity(nnz),
      offs: OS::with_capacity(nolp1),
    }
  }
}

impl<DS, IS, OS, const D: usize> Csl<DS, IS, OS, D> {
  /// The definitions of all dimensions.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::csl_array_4;
  /// assert_eq!(csl_array_4().dims(), &[2, 3, 4, 5]);
  /// ```
  #[inline]
  pub fn dims(&self) -> &[usize; D] {
    &self.dims
  }
}

impl<DATA, DS, IS, OS, const D: usize> Csl<DS, IS, OS, D>
where
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
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslArray, CslVec};
  /// // Sparse array ([8, _, _, _, _, 9, _, _, _, _])
  /// let mut _sparse_array = CslArray::new([10], [8.0, 9.0], [0, 5], [0, 2]);
  /// // A bunch of nothing for your overflow needs
  /// let mut _over_nine: ndsparse::Result<CslVec<(), 9001>>;
  /// _over_nine = CslVec::new([0; 9001], vec![], vec![], vec![]);
  /// ```
  pub fn new<IDS, IIS, IOS>(
    dims: [usize; D],
    into_data: IDS,
    into_indcs: IIS,
    into_offs: IOS,
  ) -> crate::Result<Self>
  where
    IDS: Into<DS>,
    IIS: Into<IS>,
    IOS: Into<OS>,
  {
    let data = into_data.into();
    let indcs = into_indcs.into();
    let offs = into_offs.into();
    let data_ref = data.as_ref();
    let indcs_ref = indcs.as_ref();
    let offs_ref = offs.as_ref();

    let innermost_dim_is_zero = {
      let mut iter = dims.iter().copied();
      while let Some(dim) = iter.next() {
        if dim != 0 {
          break;
        }
      }
      iter.any(|v| v == 0)
    };
    if innermost_dim_is_zero {
      return Err(CslError::InnermostDimsZero.into());
    }

    if data_ref.len() != indcs_ref.len() {
      return Err(CslError::DiffDataIndcsLength.into());
    }

    if !are_in_ascending_order(&offs_ref, |a, b| [a, b]) {
      return Err(CslError::InvalidOffsetsOrder.into());
    }

    let data_indcs_length_greater_than_dims_length = {
      let max_nnz = max_nnz(&dims);
      data_ref.len() > max_nnz || indcs_ref.len() > max_nnz
    };
    if data_indcs_length_greater_than_dims_length {
      return Err(CslError::DataIndcsLengthGreaterThanDimsLength.into());
    }

    if let Some(last) = dims.last() {
      let are_in_upper_bound = are_in_upper_bound(indcs_ref, last);
      if !are_in_upper_bound {
        return Err(CslError::IndcsGreaterThanEqualDimLength.into());
      }
      if offs_ref.len() != correct_offs_len(&dims)? {
        return Err(CslError::InvalidOffsetsLength.into());
      }
    }

    let first_off = if let Some(r) = offs_ref.first() {
      r
    } else {
      return Ok(Self { data, dims, indcs, offs });
    };

    if let Some(last_ref) = offs_ref.last() {
      let last = last_ref - first_off;
      if last != data_ref.len() || last != indcs_ref.len() {
        return Err(CslError::LastOffsetDifferentNnz.into());
      }
    }

    let has_duplicated_indices = windows2(offs_ref).any(|[a, b]| {
      if let Some(indcs) = indcs_ref.get(a - first_off..b - first_off) {
        has_duplicates(indcs)
      } else {
        false
      }
    });
    if has_duplicated_indices {
      return Err(CslError::DuplicatedIndices.into());
    }

    Ok(Self { data, dims, indcs, offs })
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
  /// assert_eq!(csl.line([0, 0, 2, 0]), CslRef::new([5], &[][..], &[][..], &[3, 3][..]).ok());
  /// assert_eq!(csl.line([0, 1, 0, 0]), CslRef::new([5], &[6][..], &[2][..], &[5, 6][..]).ok());
  /// ```
  pub fn line(&self, indcs: [usize; D]) -> Option<CslRef<'_, DATA, 1>> {
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

  /// Iterator that returns immutable line references of the outermost dimension
  ///
  /// # Examples
  ///
  /// ```rust
  /// # fn main() -> ndsparse::Result<()> {
  /// use ndsparse::{csl::CslRef, doc_tests::csl_array_4};
  /// let csl = csl_array_4();
  /// let sub_csl = csl.sub_dim(0..3).unwrap();
  /// let mut iter = sub_csl.outermost_line_iter()?;
  /// assert_eq!(
  ///   iter.next(),
  ///   CslRef::new([1, 4, 5], &[1, 2, 3, 4, 5][..], &[0, 3, 1, 3, 4][..], &[0, 2, 3, 3, 5][..]).ok()
  /// );
  /// assert_eq!(iter.next(), CslRef::new([1, 4, 5], &[6][..], &[2][..], &[5, 6, 6, 6, 6][..]).ok());
  /// assert_eq!(
  ///   iter.next(),
  ///   CslRef::new([1, 4, 5], &[7, 8][..], &[2, 4][..], &[6, 7, 8, 8, 8][..]).ok()
  /// );
  /// assert_eq!(iter.next(), None);
  /// # Ok(()) }
  pub fn outermost_line_iter(&self) -> crate::Result<CslLineIterRef<'_, DATA, D>> {
    CslLineIterRef::new(self.dims, self.data.as_ref(), self.indcs.as_ref(), self.offs.as_ref())
  }

  /// Parallel iterator that returns all immutable line references of the current dimension
  /// using `rayon`.
  ///
  /// # Examples
  #[cfg_attr(all(feature = "alloc", feature = "with-rayon"), doc = "```rust")]
  #[cfg_attr(not(all(feature = "alloc", feature = "with-rayon")), doc = "```ignore")]
  /// # fn main() -> ndsparse::Result<()> {
  /// use ndsparse::doc_tests::csl_array_4;
  /// use rayon::prelude::*;
  /// let csl = csl_array_4();
  /// let outermost_rayon_iter = csl.outermost_line_rayon_iter()?;
  /// outermost_rayon_iter.enumerate().for_each(|(idx, csl_ref)| {
  ///   assert_eq!(csl_ref, csl.outermost_line_iter().unwrap().nth(idx).unwrap());
  /// });
  /// # Ok(()) }
  /// ```
  #[cfg(feature = "with-rayon")]
  pub fn outermost_line_rayon_iter(
    &self,
  ) -> crate::Result<crate::ParallelIteratorWrapper<CslLineIterRef<'_, DATA, D>>> {
    Ok(crate::ParallelIteratorWrapper(self.outermost_line_iter()?))
  }

  /// Retrieves an immutable reference of any sub dimension.
  ///
  /// # Arguments
  ///
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
  ///     .ok()
  /// );
  /// // The last 2 matrices of the first cuboid;
  /// assert_eq!(
  ///   csl.sub_dim(1..3),
  ///   CslRef::new([2, 4, 5], &[6, 7, 8][..], &[2, 2, 4][..], &[5, 6, 6, 6, 6, 7, 8, 8, 8][..]).ok()
  /// );
  /// ```
  pub fn sub_dim<const TD: usize>(&self, range: Range<usize>) -> Option<CslRef<'_, DATA, TD>> {
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
  /// let csl = csl_array_4();
  /// assert_eq!(csl.value([1, 0, 2, 2]), Some(&9));
  /// let line = csl.line([0, 0, 3, 0]).unwrap();
  /// assert_eq!(line.value([3]), Some(&4));
  /// ```
  pub fn value(&self, indcs: [usize; D]) -> Option<&DATA> {
    let idx = data_idx(self, indcs)?;
    self.data.as_ref().get(idx)
  }
}

impl<DATA, DS, IS, OS, const D: usize> Csl<DS, IS, OS, D>
where
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Storage<Item = DATA>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  /// Clears all values and dimensions.
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::{csl::CslVec, doc_tests::csl_vec_4};
  /// let mut csl = csl_vec_4();
  /// csl.clear();
  /// assert_eq!(csl, CslVec::default());
  /// ```
  pub fn clear(&mut self)
  where
    DS: Clear,
    IS: Clear,
    OS: Clear + Push<Input = usize>,
  {
    self.dims = crate::utils::default_array();
    self.data.clear();
    self.indcs.clear();
    self.offs.clear();
    self.offs.push(0);
  }

  /// See [`CslLineConstructor`](CslLineConstructor) for more information.
  pub fn constructor(&mut self) -> crate::Result<CslLineConstructor<'_, DS, IS, OS, D>>
  where
    DS: Push<Input = DATA>,
    IS: Push<Input = usize>,
    OS: Push<Input = usize>,
  {
    CslLineConstructor::new(self)
  }

  /// Mutable version of [`data`](#method.data).
  pub fn data_mut(&mut self) -> &mut [DATA] {
    self.data.as_mut()
  }

  /// Mutable version of [`line`](#method.line).
  pub fn line_mut(&mut self, indcs: [usize; D]) -> Option<CslMut<'_, DATA, 1>> {
    line_mut(self, indcs)
  }

  /// Mutable version of [`outermost_line_iter`](#method.outermost_line_iter).
  pub fn outermost_line_iter_mut(&mut self) -> crate::Result<CslLineIterMut<'_, DATA, D>> {
    CslLineIterMut::new(self.dims, self.data.as_mut(), self.indcs.as_ref(), self.offs.as_ref())
  }

  /// Mutable version of [`outermost_line_rayon_iter`](#method.outermost_line_rayon_iter).
  #[cfg(feature = "with-rayon")]
  pub fn outermost_line_rayon_iter_mut(
    &mut self,
  ) -> crate::Result<crate::ParallelIteratorWrapper<CslLineIterMut<'_, DATA, D>>> {
    Ok(crate::ParallelIteratorWrapper(self.outermost_line_iter_mut()?))
  }

  /// Mutable version of [`sub_dim`](#method.sub_dim).
  pub fn sub_dim_mut<const TD: usize>(
    &mut self,
    range: Range<usize>,
  ) -> Option<CslMut<'_, DATA, TD>> {
    sub_dim_mut(self, range)
  }

  /// Intra-swap a single data value.
  ///
  /// # Arguments
  ///
  /// * `a`: First set of indices
  /// * `b`: Second set of indices
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::doc_tests::csl_vec_4;
  /// let mut csl = csl_vec_4();
  /// csl.swap_value([0, 0, 0, 0], [1, 0, 2, 2]);
  /// assert_eq!(csl.data(), &[9, 2, 3, 4, 5, 6, 7, 8, 1]);
  /// ```
  pub fn swap_value(&mut self, a: [usize; D], b: [usize; D]) -> bool {
    if let Some(a_idx) = data_idx(self, a) {
      if let Some(b_idx) = data_idx(self, b) {
        self.data.as_mut().swap(a_idx, b_idx);
        return true;
      }
    }
    false
  }

  /// Truncates all values in the exactly exclusive point defined by `indcs`.
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::{csl::CslVec, doc_tests::csl_vec_4};
  /// let mut csl = csl_vec_4();
  /// csl.truncate([0, 0, 3, 4]);
  /// assert_eq!(
  ///   Ok(csl),
  ///   CslVec::new([0, 0, 4, 5], vec![1, 2, 3, 4], vec![0, 3, 1, 3], vec![0, 2, 3, 3, 4])
  /// );
  /// ```
  pub fn truncate(&mut self, indcs: [usize; D])
  where
    DS: Truncate<Input = usize>,
    IS: Truncate<Input = usize>,
    OS: Push<Input = usize> + Truncate<Input = usize>,
  {
    let [offs_indcs, values] = if let Some(r) = line_offs(&self.dims, &indcs, self.offs.as_ref()) {
      r
    } else {
      return;
    };
    let last_idx = if let Some(r) = indcs.last() {
      *r
    } else {
      return;
    };
    let cut_point = values.start.saturating_add(1);
    self.data.truncate(cut_point);
    self.indcs.truncate(cut_point);
    self.offs.truncate(offs_indcs.start.saturating_add(1));
    self.offs.push(last_idx);
    indcs.iter().zip(self.dims.iter_mut()).filter(|&(a, _)| *a == 0).for_each(|(_, b)| *b = 0);
  }

  /// Mutable version of [`value`](#method.value).
  pub fn value_mut(&mut self, indcs: [usize; D]) -> Option<&mut DATA> {
    let idx = data_idx(self, indcs)?;
    self.data.as_mut().get_mut(idx)
  }
}

#[cfg(feature = "with-rand")]
impl<DATA, DS, IS, OS, const D: usize> Csl<DS, IS, OS, D>
where
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
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// use rand::{Rng, rngs::mock::StepRng};
  /// let mut rng = StepRng::new(0, 1);
  /// let dims = [1, 2, 3];
  /// let mut _random: ndsparse::Result<CslVec<u8, 3>>;
  /// _random = CslVec::new_controlled_random_rand(dims, 9, &mut rng, |r, _| r.gen());
  /// ```
  pub fn new_controlled_random_rand<F, R>(
    dims: [usize; D],
    nnz: usize,
    rng: &mut R,
    cb: F,
  ) -> crate::Result<Self>
  where
    F: FnMut(&mut R, [usize; D]) -> DATA,
    R: rand::Rng,
  {
    let mut csl = Self::default();
    csl.dims = dims;
    csl_rnd::CslRnd::new(&mut csl, nnz, rng)?.fill(cb)?;
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
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// # fn main() -> ndsparse::Result<()> {
  /// use ndsparse::csl::CslVec;
  /// use rand::{rngs::mock::StepRng, seq::SliceRandom};
  /// let mut rng = StepRng::new(0, 1);
  /// let upper_bound = 5;
  /// let random: ndsparse::Result<CslVec<u8, 8>>;
  /// random = CslVec::new_random_rand(&mut rng, upper_bound);
  /// assert!(random?.dims().choose(&mut rng).unwrap() < &upper_bound);
  /// # Ok(()) }
  pub fn new_random_rand<R>(rng: &mut R, upper_bound: usize) -> crate::Result<Self>
  where
    R: rand::Rng,
    rand::distributions::Standard: rand::distributions::Distribution<DATA>,
  {
    let dims = crate::utils::valid_random_dims(rng, upper_bound);
    let max_nnz = max_nnz(&dims);
    let nnz = if max_nnz == 0 { 0 } else { rng.gen_range(0, max_nnz) };
    Self::new_controlled_random_rand(dims, nnz, rng, |rng, _| rng.gen())
  }
}

impl<DS, IS, OS, const D: usize> Default for Csl<DS, IS, OS, D>
where
  DS: Default,
  IS: Default,
  OS: Default + Push<Input = usize>,
{
  fn default() -> Self {
    let mut offs: OS = Default::default();
    offs.push(0);
    Self {
      data: Default::default(),
      dims: crate::utils::default_array(),
      indcs: Default::default(),
      offs,
    }
  }
}
