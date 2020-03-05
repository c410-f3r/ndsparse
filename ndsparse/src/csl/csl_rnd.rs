use crate::csl::{max_nnz, offs_len, outermost_stride, Csl};
use cl_traits::Push;
use rand::{
  distributions::{Distribution, Uniform},
  Rng,
};

#[derive(Debug)]
pub struct CslRnd<'a, DATA, DS, IS, OS, R, const DIMS: usize> {
  csl: &'a mut Csl<DATA, DS, IS, OS, DIMS>,
  nnz: usize,
  rng: &'a mut R,
}

impl<'a, DATA, DS, IS, OS, R, const DIMS: usize> CslRnd<'a, DATA, DS, IS, OS, R, DIMS>
where
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Push<Input = DATA>,
  IS: AsMut<[usize]> + AsRef<[usize]> + Push<Input = usize>,
  R: Rng,
  OS: AsMut<[usize]> + AsRef<[usize]> + Push<Input = usize>,
{
  pub fn new(csl: &'a mut Csl<DATA, DS, IS, OS, DIMS>, nnz: usize, rng: &'a mut R) -> Self {
    Self { csl, nnz, rng }
  }

  pub fn fill<F>(mut self, cb: F)
  where
    F: FnMut(&mut R, [usize; DIMS]) -> DATA,
  {
    self.fill_offs();
    self.fill_indcs();
    self.fill_data(cb);
  }

  fn fill_data<F>(&mut self, mut cb: F)
  where
    F: FnMut(&mut R, [usize; DIMS]) -> DATA,
  {
    let data = &mut self.csl.data;
    let indcs = self.csl.indcs.as_ref();
    let orig_dims = self.csl.dims;
    let outermost_stride = outermost_stride(&orig_dims);
    let rng = &mut self.rng;
    self.csl.offs.as_ref().windows(2).enumerate().for_each(|(line_idx, offset)| {
      let mut dims = *orig_dims;
      dims[0] = if outermost_stride == 0 { 0 } else { line_idx % outermost_stride };
      for (dim, &orig_dim) in dims.iter_mut().zip(orig_dims.iter()).skip(1).rev().skip(1) {
        *dim = if orig_dim == 0 { 0 } else { line_idx % orig_dim };
      }
      for innermost_idx in indcs[offset[0]..offset[1]].iter().copied() {
        *dims.last_mut().unwrap() = innermost_idx;
        data.push(cb(rng, dims));
      }
    });
  }

  fn fill_indcs(&mut self) {
    let dims = &self.csl.dims;
    let rng = &mut self.rng;
    let indcs = &mut self.csl.indcs;
    self.csl.offs.as_ref().windows(2).for_each(|offset| {
      let mut counter = 0;
      let line_nnz = offset[1] - offset[0];
      while counter < line_nnz {
        let rnd = rng.gen_range(0, *dims.last().unwrap());
        if indcs.as_ref()[offset[0]..].contains(&rnd) == false {
          indcs.push(rnd);
          counter += 1;
        }
      }
      indcs.as_mut()[offset[0]..].sort_unstable();
    });
  }

  fn fill_offs(&mut self) {
    for _ in 0..offs_len(&self.csl.dims) {
      self.csl.offs.push(0);
    }
    let nnz = Uniform::from(0..=max_nnz(&self.csl.dims)).sample(self.rng);
    let mut previous_off = 0;
    for idx in 1..self.csl.offs.as_ref().len() {
      if previous_off == nnz {
        *self.csl.offs.as_mut().get_mut(idx).unwrap() = nnz;
      } else if previous_off > nnz {
        *self.csl.offs.as_mut().get_mut(idx - 1).unwrap() = nnz;
        *self.csl.offs.as_mut().get_mut(idx).unwrap() = nnz;
      } else {
        let innermost_dim_len = *self.csl.dims.last().unwrap();
        let line_nnz = Uniform::from(0..=innermost_dim_len).sample(self.rng);
        *self.csl.offs.as_mut().get_mut(idx).unwrap() = previous_off + line_nnz;
      }
      previous_off = *self.csl.offs.as_ref().get(idx).unwrap();
    }
  }
}
