use crate::{
  csl::{offs_len, outermost_stride, Csl},
  Dims,
};
use cl_traits::{Push, Storage};
use core::cmp::Ordering;
use rand::{
  distributions::{Distribution, Uniform},
  Rng,
};

#[derive(Debug)]
pub struct CslRnd<'a, DA, DS, IS, OS, R>
where
  DA: Dims,
{
  csl: &'a mut Csl<DA, DS, IS, OS>,
  nnz: usize,
  rng: &'a mut R,
}

impl<'a, DA, DATA, DS, IS, OS, R> CslRnd<'a, DA, DS, IS, OS, R>
where
  DA: Dims,
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Push<Input = DATA> + Storage<Item = DATA>,
  IS: AsMut<[usize]> + AsRef<[usize]> + Push<Input = usize>,
  R: Rng,
  OS: AsMut<[usize]> + AsRef<[usize]> + Push<Input = usize>,
{
  pub fn new(csl: &'a mut Csl<DA, DS, IS, OS>, nnz: usize, rng: &'a mut R) -> Self {
    Self { csl, nnz, rng }
  }

  pub fn fill<F>(mut self, cb: F)
  where
    F: FnMut(&mut R, DA) -> DATA,
  {
    self.fill_offs();
    self.fill_indcs();
    self.fill_data(cb);
  }

  fn fill_data<F>(&mut self, mut cb: F)
  where
    F: FnMut(&mut R, DA) -> DATA,
  {
    let data = &mut self.csl.data;
    let indcs = self.csl.indcs.as_ref();
    let orig_dims = self.csl.dims;
    let outermost_stride = outermost_stride(&orig_dims);
    let rng = &mut self.rng;
    self.csl.offs.as_ref().windows(2).enumerate().for_each(|(line_idx, offset)| {
      let mut dims = *orig_dims;
      dims.slice_mut()[0] = if outermost_stride == 0 { 0 } else { line_idx % outermost_stride };
      for (dim, &orig_dim) in
        dims.slice_mut().iter_mut().zip(orig_dims.slice().iter()).skip(1).rev().skip(1)
      {
        *dim = if orig_dim == 0 { 0 } else { line_idx % orig_dim };
      }
      for innermost_idx in indcs[offset[0]..offset[1]].iter().copied() {
        *dims.slice_mut().last_mut().unwrap() = innermost_idx;
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
        let rnd = rng.gen_range(0, *dims.slice().last().unwrap());
        if !indcs.as_ref()[offset[0]..].contains(&rnd) {
          indcs.push(rnd);
          counter += 1;
        }
      }
      indcs.as_mut()[offset[0]..].sort_unstable();
    });
  }

  fn fill_offs(&mut self) {
    let nnz = self.nnz;
    for _ in 0..offs_len(&self.csl.dims) {
      self.csl.offs.push(0);
    }
    let mut last_visited_off = self.do_fill_offs(|idl, _, s| Uniform::from(0..=idl).sample(s.rng));
    loop {
      if *self.csl.offs.as_ref().get(last_visited_off).unwrap() >= nnz {
        let slice_opt = self.csl.offs.as_mut().get_mut(last_visited_off..);
        slice_opt.unwrap().iter_mut().for_each(|off| *off = nnz);
        break;
      }
      let mut offs_adjustment = 0;
      last_visited_off = self.do_fill_offs(|idl, idx, s| {
        let offs = s.csl.offs.as_mut();
        let curr = offs[idx] + offs_adjustment;
        let prev = offs[idx - 1];
        let start = curr - prev;
        let line_nnz = Uniform::from(start..=idl).sample(s.rng);
        offs_adjustment += (line_nnz + prev) - curr;
        line_nnz
      });
    }
  }

  fn do_fill_offs<F>(&mut self, mut f: F) -> usize
  where
    F: FnMut(usize, usize, &mut Self) -> usize,
  {
    let nnz = self.nnz;
    let mut idx = 1;
    let mut previous_nnz = *self.csl.offs.as_ref().first().unwrap();
    loop {
      if idx >= self.csl.offs.as_ref().len() {
        break;
      }
      match previous_nnz.cmp(&nnz) {
        Ordering::Equal => {
          break;
        }
        Ordering::Greater => {
          *self.csl.offs.as_mut().get_mut(idx - 1).unwrap() = nnz;
          break;
        }
        Ordering::Less => {
          let innermost_dim_len = *self.csl.dims.slice().last().unwrap();
          let line_nnz = f(innermost_dim_len, idx, self);
          let new_nnz = previous_nnz + line_nnz;
          *self.csl.offs.as_mut().get_mut(idx).unwrap() = new_nnz;
          previous_nnz = new_nnz;
        }
      }
      idx += 1;
    }
    idx - 1
  }
}
