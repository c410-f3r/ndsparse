//! COO (Coordinate) format for N-dimensions.

mod coo_utils;

use crate::Dims;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use cl_traits::{ArrayWrapper, Storage};
use coo_utils::*;

/// COO backed by a static array.
pub type CooArray<DA, DTA> = Coo<DA, ArrayWrapper<DTA>>;
#[cfg(feature = "with_arrayvec")]
/// COO backed by the `ArrayVec` dependency.
pub type CooArrayVec<DA, DTA> = Coo<DA, cl_traits::ArrayVecArrayWrapper<DTA>>;
/// Mutable COO reference.
pub type CooMut<'a, DA, DATA> = Coo<DA, &'a mut [(ArrayWrapper<DA>, DATA)]>;
/// Immutable COO reference.
pub type CooRef<'a, DA, DATA> = Coo<DA, &'a [(ArrayWrapper<DA>, DATA)]>;
#[cfg(feature = "with_smallvec")]
/// COO backed by the `SmallVec` dependency.
pub type CooSmallVec<DA, DTA> = Coo<DA, cl_traits::SmallVecArrayWrapper<DTA>>;
#[cfg(feature = "with_staticvec")]
/// COO backed by the `StaticVec` dependency
pub type CooStaticVec<DATA, const DIMS: usize, const NNZ: usize> =
  Coo<[usize; DIMS], staticvec::StaticVec<(ArrayWrapper<[usize; DIMS]>, DATA), NNZ>>;
#[cfg(feature = "alloc")]
/// COO backed by a dynamic vector.
pub type CooVec<DA, DATA> = Coo<DA, Vec<(ArrayWrapper<DA>, DATA)>>;

/// Base structure for all COO* variants.
///
/// # Types
///
/// * `DS`: Data Storage
/// * `const DIMS: usize`: Dimensions length
#[cfg_attr(feature = "with_serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, Default, PartialEq)]
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
  ///
  /// ```rust
  /// use ndsparse::coo::{CooArray, CooVec};
  /// // Sparse array ([8, _, _, _, _, 9, _, _, _, _])
  /// let mut _sparse_array = CooArray::new([10], [([0].into(), 8.0), ([5].into(), 9.0)]);
  /// // A bunch of nothing for your overflow needs
  /// let mut _over_nine: CooVec<[usize; 9001], ()>;
  /// _over_nine = CooVec::new([0; 9001], vec![]);
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
    ID: Into<ArrayWrapper<DA>>,
    IDS: Into<DS>,
  {
    let data = into_data.into();
    let dims = into_dims.into();
    assert!(
      crate::utils::are_in_ascending_order(data.as_ref(), |a, b| [&a.0, &b.0]),
      "Data indices must be in asceding order"
    );
    assert!(
      data.as_ref().iter().all(|(indcs, _)| {
        indcs.slice().iter().zip(dims.slice().iter()).all(|(data_idx, dim)| data_idx < dim)
      }),
      "All indices must be lesser than the defined dimensions"
    );
    assert!(
      does_not_have_duplicates_sorted(data.as_ref(), |a, b| a.0[..] != b.0[..]),
      "Must not have duplicated indices"
    );
    Self { data, dims }
  }

  /// The data that is being stored.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::doc_tests::coo_array_5;
  /// assert_eq!(coo_array_5().data().get(0), Some(&([0, 0, 1, 1, 2].into(), 1)));
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
  /// Mutable version of [`data`](#method.data).
  ///
  /// # Safety
  ///
  /// Indices can be modified to overflow its dimensions.
  pub unsafe fn data_mut(&mut self) -> &[(ArrayWrapper<DA>, DATA)] {
    self.data.as_mut()
  }

  /// Mutable version of [`value`](#method.value).
  pub fn value_mut(&mut self, indcs: DA) -> Option<&mut DATA> {
    value_mut(indcs.into(), self.data.as_mut())
  }
}

#[cfg(feature = "with_rand")]
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
  ///
  /// ```rust
  /// use ndsparse::coo::CooVec;
  /// use rand::{thread_rng, Rng};
  /// let mut _random: CooVec<[usize; 8], u8>;
  /// let mut rng = thread_rng();
  /// _random = CooVec::new_random_with_rand([1, 2, 3, 4, 5, 6, 7, 8], 9, &mut rng, |r, _| r.gen());
  /// ```
  pub fn new_random_with_rand<F, ID, R>(into_dims: ID, nnz: usize, rng: &mut R, mut cb: F) -> Self
  where
    F: FnMut(&mut R, &DA) -> DATA,
    ID: Into<ArrayWrapper<DA>>,
    R: rand::Rng,
  {
    use rand::distributions::Distribution;
    let dims = into_dims.into();
    let mut data: DS = Default::default();
    for _ in 0..nnz {
      data.push({
        let indcs: DA = cl_traits::create_array(|idx| {
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
impl<DA, DATA, DS> quickcheck::Arbitrary for Coo<DA, DS>
where
  DA: Dims + Clone + Send + 'static,
  DATA: Default + quickcheck::Arbitrary,
  DS: AsRef<[<DS as Storage>::Item]>
    + AsMut<[<DS as Storage>::Item]>
    + Clone
    + Default
    + Send
    + Storage<Item = (ArrayWrapper<DA>, DATA)>
    + cl_traits::Push<Input = <DS as Storage>::Item>
    + 'static,
  rand::distributions::Standard: rand::distributions::Distribution<DATA>,
{
  #[inline]
  fn arbitrary<G>(g: &mut G) -> Self
  where
    G: quickcheck::Gen,
  {
    use rand::Rng;
    let dims: DA = cl_traits::create_array(|_| g.gen_range(0, g.size()));
    let nnz = g.gen_range(0, dims.slice().iter().product::<usize>());
    Self::new_random_with_rand(dims, nnz, g, |g, _| g.gen())
  }
}
