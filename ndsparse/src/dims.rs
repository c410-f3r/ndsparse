use cl_traits::Array;

/// Gathers the typical bounds required by an array of dimensions ([usize; N]).
pub trait Dims: Array<Item = usize> + Copy {}
impl<T> Dims for T where T: Array<Item = usize> + Copy {}
