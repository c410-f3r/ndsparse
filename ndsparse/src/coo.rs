//! COO (Coordinate) format for N-dimensions.

mod coo_error;
mod coo_utils;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use cl_traits::Storage;
pub use coo_error::*;
use coo_utils::*;

/// COO backed by a static array.
pub type CooArray<DATA, const D: usize, const DT: usize> = Coo<[([usize; D], DATA); DT], D>;

/// COO backed by a mutable slice
pub type CooMut<'a, DATA, const D: usize> = Coo<&'a mut [([usize; D], DATA)], D>;

/// COO backed by a slice
pub type CooRef<'a, DATA, const D: usize> = Coo<&'a [([usize; D], DATA)], D>;

#[cfg(feature = "alloc")]
/// COO backed by a dynamic vector.
pub type CooVec<DATA, const D: usize> = Coo<Vec<([usize; D], DATA)>, D>;

/// Base structure for all COO* variants.
///
/// # Types
///
/// * `DA`: Data Array
/// * `DS`: Data Storage
#[cfg_attr(feature = "with-serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Coo<DS, const D: usize> {
  pub(crate) data: DS,
  pub(crate) dims: [usize; D],
}

impl<DS, const D: usize> Coo<DS, D> {
  /// The definitions of all dimensions.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::coo_array_5;
  /// assert_eq!(coo_array_5().dims(), &[2, 3, 4, 3, 3]);
  /// ```
  #[inline]
  pub fn dims(&self) -> &[usize; D] {
    &self.dims
  }
}

impl<DATA, DS, const D: usize> Coo<DS, D>
where
  DS: AsRef<[<DS as Storage>::Item]> + Storage<Item = ([usize; D], DATA)>,
{
  /// Creates a valid COO instance.
  ///
  /// # Arguments
  ///
  /// * `dims`: Array of dimensions
  /// * `data`: Data collection
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::coo::{CooArray, CooVec};
  /// // Sparse array ([8, _, _, _, _, 9, _, _, _, _])
  /// let mut _sparse_array = CooArray::new([10], [([0], 8.0), ([5], 9.0)]);
  /// // A bunch of nothing for your overflow needs
  /// let mut _over_nine: ndsparse::Result<CooVec<(), 9001>>;
  /// _over_nine = CooVec::new([0; 9001], vec![]);
  /// ```
  pub fn new(dims: [usize; D], data: DS) -> crate::Result<Self> {
    if !crate::utils::are_in_ascending_order(data.as_ref(), |a, b| [&a.0, &b.0]) {
      return Err(CooError::InvalidIndcsOrder.into());
    }
    let has_invalid_indcs = !data.as_ref().iter().all(|&(indcs, _)| {
      indcs.iter().zip(dims.iter()).all(
        |(data_idx, dim)| {
          if dim == &0 {
            true
          } else {
            data_idx < dim
          }
        },
      )
    });
    if has_invalid_indcs {
      return Err(CooError::InvalidIndcs.into());
    }
    if !does_not_have_duplicates_sorted(data.as_ref(), |a, b| a.0[..] != b.0[..]) {
      return Err(CooError::DuplicatedIndices.into());
    }
    Ok(Self { data, dims })
  }

  /// The data that is being stored.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::coo_array_5;
  /// assert_eq!(coo_array_5().data().first(), Some(&([0, 0, 1, 1, 2].into(), 1)));
  /// ```
  pub fn data(&self) -> &[([usize; D], DATA)] {
    self.data.as_ref()
  }

  /// If any, retrieves an immutable data reference of a given set of indices.
  ///
  /// # Arguments
  ///
  /// * `indcs`: Indices of the desired data location
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::coo_array_5;
  /// let coo = coo_array_5();
  /// assert_eq!(coo.value([0, 0, 0, 0, 0]), None);
  /// assert_eq!(coo.value([0, 2, 2, 0, 1]), Some(&4));
  /// ```
  pub fn value(&self, indcs: [usize; D]) -> Option<&DATA> {
    value(indcs, &self.data.as_ref())
  }
}

impl<DATA, DS, const D: usize> Coo<DS, D>
where
  DS: AsMut<[<DS as Storage>::Item]> + Storage<Item = ([usize; D], DATA)>,
{
  /// Mutable version of [`value`](#method.value).
  pub fn value_mut(&mut self, indcs: [usize; D]) -> Option<&mut DATA> {
    value_mut(indcs, self.data.as_mut())
  }
}

#[cfg(feature = "with-rand")]
impl<DATA, DS, const D: usize> Coo<DS, D>
where
  DS: AsMut<[<DS as Storage>::Item]>
    + AsRef<[<DS as Storage>::Item]>
    + Default
    + Storage<Item = ([usize; D], DATA)>
    + cl_traits::CapacityUpperBound<Output = usize>
    + cl_traits::Push<Input = <DS as Storage>::Item>,
{
  /// Creates a new random and valid instance delimited by the passed arguments.
  ///
  /// # Arguments
  ///
  /// * `dims`: Array of dimensions
  /// * `nnz`: Number of Non-Zero elements
  /// * `rng`: `rand::Rng` trait
  /// * `cb`: Callback to control data creation
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::coo::CooVec;
  /// use rand::{Rng, rngs::mock::StepRng};
  /// let mut rng = StepRng::new(0, 1);
  /// let dims = [1, 2, 3];
  /// let mut _random: ndsparse::Result<CooVec<u8, 3>>;
  /// _random = CooVec::new_controlled_random_rand(dims, 3, &mut rng, |r, _| r.gen());
  /// ```
  pub fn new_controlled_random_rand<F, R>(
    dims: [usize; D],
    nnz: usize,
    rng: &mut R,
    mut cb: F,
  ) -> crate::Result<Self>
  where
    F: FnMut(&mut R, &[usize; D]) -> DATA,
    R: rand::Rng,
  {
    use rand::distributions::Distribution;
    if nnz > crate::utils::max_nnz(&dims) {
      return Err(CooError::NnzGreaterThanMaximumNnz.into());
    }
    let mut data: DS = Default::default();
    if nnz > data.as_ref().len() {
      return Err(crate::Error::InsufficientCapacity.into());
    }
    for _ in 0..nnz {
      let indcs: [usize; D] = cl_traits::create_array(|idx| {
        let dim = *dims.get(idx).unwrap_or(&0);
        if dim == 0 {
          0
        } else {
          rand::distributions::Uniform::from(0..dim).sample(rng)
        }
      });
      if data.as_ref().iter().all(|value| value.0 != indcs) {
        let _ = data.push({
          let element = cb(rng, &indcs);
          (indcs, element)
        });
      }
    }
    data.as_mut().sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Coo::new(dims, data)
  }

  /// Creates a new random and valid instance.
  ///
  /// # Arguments
  ///
  /// * `rng`: `rand::Rng` trait
  /// * `upper_bound`: The maximum allowed exclusive dimension
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// # fn main() -> ndsparse::Result<()> {
  /// use ndsparse::coo::CooVec;
  /// use rand::{rngs::mock::StepRng, seq::SliceRandom};
  /// let mut rng = StepRng::new(0, 1);
  /// let upper_bound = 5;
  /// let random: ndsparse::Result<CooVec<u8, 8>>;
  /// random = CooVec::new_random_rand(&mut rng, upper_bound);
  /// assert!(random?.dims().choose(&mut rng).unwrap() < &upper_bound);
  /// # Ok(()) }
  pub fn new_random_rand<R>(rng: &mut R, upper_bound: usize) -> crate::Result<Self>
  where
    R: rand::Rng,
    rand::distributions::Standard: rand::distributions::Distribution<DATA>,
  {
    let dims = crate::utils::valid_random_dims(rng, upper_bound);
    let max_nnz = crate::utils::max_nnz(&dims);
    let nnz = if max_nnz == 0 { 0 } else { rng.gen_range(0, max_nnz) };
    Self::new_controlled_random_rand(dims, nnz, rng, |rng, _| rng.gen())
  }
}

impl<DS, const D: usize> Default for Coo<DS, D>
where
  DS: Default,
{
  fn default() -> Self {
    Self { data: DS::default(), dims: cl_traits::default_array() }
  }
}
