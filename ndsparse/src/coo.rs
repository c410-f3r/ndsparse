//! COO (Coordinate) format for N-dimensions.

mod coo_utils;

use coo_utils::*;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use cl_traits::ArrayWrapper;
use core::marker::PhantomData;

/// COO backed by a static array.
pub type CooArray<DATA, const DIMS: usize, const NNZ: usize> =
  Coo<DATA, ArrayWrapper<(ArrayWrapper<usize, DIMS>, DATA), NNZ>, DIMS>;
#[cfg(feature = "with_arrayvec")]
/// COO backed by the `ArrayVec` dependency.
pub type CooArrayVec<DATA, const DIMS: usize, const NNZ: usize> =
  Coo<DATA, cl_traits::ArrayVecArrayWrapper<(ArrayWrapper<usize, DIMS>, DATA), NNZ>, DIMS>;
/// Mutable COO reference.
pub type CooMut<'a, DATA, const DIMS: usize> =
  Coo<DATA, &'a mut [(ArrayWrapper<usize, DIMS>, DATA)], DIMS>;
/// Immutable COO reference.
pub type CooRef<'a, DATA, const DIMS: usize> =
  Coo<DATA, &'a [(ArrayWrapper<usize, DIMS>, DATA)], DIMS>;
#[cfg(feature = "with_smallvec")]
/// COO backed by the `SmallVec` dependency.
pub type CooSmallVec<DATA, const DIMS: usize, const NNZ: usize> =
  Coo<DATA, cl_traits::SmallVecArrayWrapper<(ArrayWrapper<usize, DIMS>, DATA), NNZ>, DIMS>;
#[cfg(feature = "with_staticvec")]
/// COO backed by the `StaticVec` dependency
pub type CooStaticVec<DATA, const DIMS: usize, const NNZ: usize> =
  Coo<DATA, staticvec::StaticVec<(ArrayWrapper<usize, DIMS>, DATA), NNZ>, DIMS>;
#[cfg(feature = "alloc")]
/// COO backed by a dynamic vector.
pub type CooVec<DATA, const DIMS: usize> = Coo<DATA, Vec<(ArrayWrapper<usize, DIMS>, DATA)>, DIMS>;

/// Base structure for all COO* variants.
#[cfg_attr(feature = "with_serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coo<DATA, DS, const DIMS: usize> {
  pub(crate) data: DS,
  pub(crate) dims: ArrayWrapper<usize, DIMS>,
  pub(crate) phantom: PhantomData<DATA>,
}

impl<DATA, DS, const DIMS: usize> Coo<DATA, DS, DIMS> {
  /// The definitions of all dimensions.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::coo_array_5;
  /// assert_eq!(coo_array_5().dims(), &[2, 3, 4, 3, 3]);
  /// ```
  #[inline]
  pub fn dims(&self) -> &[usize; DIMS] {
    &*self.dims
  }
}

impl<DATA, DS, const DIMS: usize> Coo<DATA, DS, DIMS>
where
  DS: AsRef<[(ArrayWrapper<usize, DIMS>, DATA)]>,
{
  /// Creates a valid COO instance.
  ///
  /// # Arguments
  ///
  /// * `into_dims`: Array of dimensions
  /// * `into_data`: Data collection
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::{
  ///   coo::{Coo, CooVec},
  ///   ArrayWrapper,
  /// };
  /// // A sparse array ([8, _, _, _, _, 9, _, _, _, _])
  /// let mut _sparse_array: Coo<f64, [(ArrayWrapper<usize, 1>, f64); 2], 1>;
  /// _sparse_array = Coo::new([10], [([0].into(), 8.0), ([5].into(), 9.0)]);
  /// // A bunch of nothing for your overflow needs
  /// let mut _over_nine: CooVec<(), 9001>;
  /// _over_nine = CooVec::new(ArrayWrapper::default(), vec![]);
  /// ```
  ///
  /// # Assertions
  ///
  /// * Data indices must be in asceding order
  /// ```rust,should_panic
  /// use ndsparse::coo::CooArray;
  /// let _ = CooArray::new([2, 2], [([1, 1].into(), 8), ([0, 0].into(), 9)]);
  /// ```
  ///
  /// * All indices must be lesser than the defined dimensions
  /// ```rust,should_panic
  /// use ndsparse::coo::CooArray;
  /// let _ = CooArray::new([2, 2], [([0, 1].into(), 8), ([9, 9].into(), 9)]);
  /// ```
  ///
  /// * Must not have duplicated indices
  /// ```rust,should_panic
  /// use ndsparse::coo::CooArray;
  /// let _ = CooArray::new([2, 2], [([0, 0].into(), 8), ([0, 0].into(), 9)]);
  /// ```
  pub fn new<ID, IDS>(into_dims: ID, into_data: IDS) -> Self
  where
    ID: Into<ArrayWrapper<usize, DIMS>>,
    IDS: Into<DS>,
  {
    let data = into_data.into();
    let dims = into_dims.into();
    assert!(
      crate::utils::are_in_ascending_order(data.as_ref(), |a, b| [&a.0, &b.0]),
      "Data indices must be in asceding order"
    );
    assert!(
      data
        .as_ref()
        .iter()
        .all(|(indcs, _)| { indcs.iter().zip(dims.iter()).all(|(data_idx, dim)| data_idx < dim) }),
      "All indices must be lesser than the defined dimensions"
    );
    assert!(
      does_not_have_duplicates_sorted(data.as_ref(), |a, b| a.0[..] != b.0[..]),
      "Must not have duplicated indices"
    );
    Self { data, dims, phantom: PhantomData }
  }

  /// The data that is being stored.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::coo_array_5;
  /// assert_eq!(coo_array_5().data().get(0), Some(&([0, 0, 1, 1, 2].into(), 1)));
  /// ```
  pub fn data(&self) -> &[(ArrayWrapper<usize, DIMS>, DATA)] {
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
  pub fn value(&self, indcs: [usize; DIMS]) -> Option<&DATA> {
    value(indcs.into(), &self.data.as_ref())
  }
}

impl<DATA, DS, const DIMS: usize> Coo<DATA, DS, DIMS>
where
  DS: AsMut<[(ArrayWrapper<usize, DIMS>, DATA)]>,
{
  /// Mutable version of [`data`](#method.data).
  ///
  /// # Safety
  ///
  /// Indices can be modified to overflow its dimensions.
  pub unsafe fn data_mut(&mut self) -> &[(ArrayWrapper<usize, DIMS>, DATA)] {
    self.data.as_mut()
  }

  /// Mutable version of [`value`](#method.value).
  pub fn value_mut(&mut self, indcs: [usize; DIMS]) -> Option<&mut DATA> {
    value_mut(indcs.into(), self.data.as_mut())
  }
}

#[cfg(feature = "with_rand")]
impl<DATA, DS, const DIMS: usize> Coo<DATA, DS, DIMS>
where
  DS: AsMut<[(ArrayWrapper<usize, DIMS>, DATA)]>
    + AsRef<[(ArrayWrapper<usize, DIMS>, DATA)]>
    + Default
    + cl_traits::Push<Input = (ArrayWrapper<usize, DIMS>, DATA)>,
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
  /// use ndsparse::coo::CooVec;
  /// use rand::{thread_rng, Rng};
  /// let mut _random: CooVec<u8, 8>;
  /// let mut rng = thread_rng();
  /// _random = CooVec::new_random_with_rand([1, 2, 3, 4, 5, 6, 7, 8], 9, &mut rng, |r, _| r.gen());
  /// ```
  pub fn new_random_with_rand<F, ID, R>(into_dims: ID, nnz: usize, rng: &mut R, mut cb: F) -> Self
  where
    F: FnMut(&mut R, &[usize; DIMS]) -> DATA,
    ID: Into<ArrayWrapper<usize, DIMS>>,
    R: rand::Rng,
  {
    use rand::distributions::Distribution;
    let dims = into_dims.into();
    let mut data: DS = Default::default();
    for _ in 0..nnz {
      data.push({
        let indcs: [usize; DIMS] = cl_traits::create_array(|idx| {
          rand::distributions::Uniform::from(0..dims[idx]).sample(rng)
        });
        let element = cb(rng, &indcs);
        (indcs.into(), element)
      });
    }
    data.as_mut().sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Coo::new(dims, data)
  }
}

#[cfg(all(test, feature = "with_rand"))]
impl<DATA, DS, const DIMS: usize> quickcheck::Arbitrary for Coo<DATA, DS, DIMS>
where
  DATA: Default + quickcheck::Arbitrary,
  DS: AsRef<[(ArrayWrapper<usize, DIMS>, DATA)]>
    + AsMut<[(ArrayWrapper<usize, DIMS>, DATA)]>
    + Clone
    + Default
    + Send
    + cl_traits::Push<Input = (ArrayWrapper<usize, DIMS>, DATA)>
    + 'static,
  rand::distributions::Standard: rand::distributions::Distribution<DATA>,
{
  #[inline]
  fn arbitrary<G>(g: &mut G) -> Self
  where
    G: quickcheck::Gen,
  {
    use rand::Rng;
    let dims = cl_traits::create_array(|_| g.gen_range(0, g.size()));
    let nnz = g.gen_range(0, dims.iter().product::<usize>());
    Self::new_random_with_rand(dims, nnz, g, |g, _| g.gen())
  }
}
