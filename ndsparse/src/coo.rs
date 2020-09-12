//! COO (Coordinate) format for N-dimensions.

mod coo_error;
mod coo_utils;

use crate::Dims;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use cl_traits::{ArrayWrapper, Storage};
pub use coo_error::*;
use coo_utils::*;

/// COO backed by a static array.
pub type CooArray<DA, DTA> = Coo<DA, ArrayWrapper<DTA>>;

/// COO backed by a mutable slice
pub type CooMut<'a, DA, DATA> = Coo<DA, &'a mut [(ArrayWrapper<DA>, DATA)]>;

/// COO backed by a slice
pub type CooRef<'a, DA, DATA> = Coo<DA, &'a [(ArrayWrapper<DA>, DATA)]>;

#[cfg(feature = "alloc")]
/// COO backed by a dynamic vector.
pub type CooVec<DA, DATA> = Coo<DA, Vec<(ArrayWrapper<DA>, DATA)>>;

/// Base structure for all COO* variants.
///
/// # Types
///
/// * `DA`: Data Array
/// * `DS`: Data Storage
#[cfg_attr(feature = "with-serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Coo<DA, DS>
where
  DA: Dims,
{
  pub(crate) data: DS,
  pub(crate) dims: ArrayWrapper<DA>,
}

impl<DA, DS> Coo<DA, DS>
where
  DA: Dims,
{
  /// The definitions of all dimensions.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::coo_array_5;
  /// assert_eq!(coo_array_5().dims(), &[2, 3, 4, 3, 3]);
  /// ```
  #[inline]
  pub fn dims(&self) -> &DA {
    &*self.dims
  }
}

impl<DA, DATA, DS> Coo<DA, DS>
where
  DA: Dims,
  DS: AsRef<[<DS as Storage>::Item]> + Storage<Item = (ArrayWrapper<DA>, DATA)>,
{
  /// Creates a valid COO instance.
  ///
  /// # Arguments
  ///
  /// * `into_dims`: Array of dimensions
  /// * `into_data`: Data collection
  ///
  /// # Example
  #[cfg_attr(all(feature = "alloc", feature = "const-generics"), doc = "```rust")]
  #[cfg_attr(not(all(feature = "alloc", feature = "const-generics")), doc = "```ignore")]
  /// use ndsparse::coo::{CooArray, CooVec};
  /// // Sparse array ([8, _, _, _, _, 9, _, _, _, _])
  /// let mut _sparse_array = CooArray::new([10], [([0].into(), 8.0), ([5].into(), 9.0)]);
  /// // A bunch of nothing for your overflow needs
  /// let mut _over_nine: ndsparse::Result<CooVec<[usize; 9001], ()>>;
  /// _over_nine = CooVec::new([0; 9001], vec![]);
  /// ```
  pub fn new<ID, IDS>(into_dims: ID, into_data: IDS) -> crate::Result<Self>
  where
    ID: Into<ArrayWrapper<DA>>,
    IDS: Into<DS>,
  {
    let data = into_data.into();
    let dims = into_dims.into();
    if !crate::utils::are_in_ascending_order(data.as_ref(), |a, b| [&a.0, &b.0]) {
      return Err(CooError::InvalidIndcsOrder.into());
    }
    let has_invalid_indcs = !data.as_ref().iter().all(|(indcs, _)| {
      indcs.slice().iter().zip(dims.slice().iter()).all(|(data_idx, dim)| {
        if dim == &0 {
          true
        } else {
          data_idx < dim
        }
      })
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
  pub fn data(&self) -> &[(ArrayWrapper<DA>, DATA)] {
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
  pub fn value(&self, indcs: DA) -> Option<&DATA> {
    value(indcs.into(), &self.data.as_ref())
  }
}

impl<DA, DATA, DS> Coo<DA, DS>
where
  DA: Dims,
  DS: AsMut<[<DS as Storage>::Item]> + Storage<Item = (ArrayWrapper<DA>, DATA)>,
{
  /// Mutable version of [`value`](#method.value).
  pub fn value_mut(&mut self, indcs: DA) -> Option<&mut DATA> {
    value_mut(indcs.into(), self.data.as_mut())
  }
}

#[cfg(feature = "with-rand")]
impl<DA, DATA, DS> Coo<DA, DS>
where
  DA: Dims,
  DS: AsMut<[<DS as Storage>::Item]>
    + AsRef<[<DS as Storage>::Item]>
    + Default
    + Storage<Item = (ArrayWrapper<DA>, DATA)>
    + cl_traits::Push<Input = <DS as Storage>::Item>,
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
  /// use ndsparse::coo::CooVec;
  /// use rand::{thread_rng, Rng};
  /// let mut rng = thread_rng();
  /// let dims = [1, 2, 3, 4, 5, 6, 7, 8];
  /// let mut _random: ndsparse::Result<CooVec<[usize; 8], u8>>;
  /// _random = CooVec::new_controlled_random_rand(dims, 9, &mut rng, |r, _| r.gen());
  /// ```
  pub fn new_controlled_random_rand<F, ID, R>(
    into_dims: ID,
    nnz: usize,
    rng: &mut R,
    mut cb: F,
  ) -> crate::Result<Self>
  where
    F: FnMut(&mut R, &DA) -> DATA,
    ID: Into<ArrayWrapper<DA>>,
    R: rand::Rng,
  {
    use rand::distributions::Distribution;
    let dims = into_dims.into();
    if nnz > crate::utils::max_nnz(&dims) {
      return Err(CooError::NnzGreaterThanMaximumNnz.into());
    }
    let mut data: DS = Default::default();
    for _ in 0..nnz {
      let indcs_array: DA = cl_traits::create_array(|idx| {
        let dim = *dims.slice().get(idx).unwrap_or(&0);
        if dim == 0 {
          0
        } else {
          rand::distributions::Uniform::from(0..dim).sample(rng)
        }
      });
      let indcs = indcs_array.into();
      if data.as_ref().iter().all(|value| value.0 != indcs) {
        data.push({
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
  /// use rand::{seq::SliceRandom, thread_rng};
  /// let mut rng = thread_rng();
  /// let upper_bound = 5;
  /// let random: ndsparse::Result<CooVec<[usize; 8], u8>>;
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
