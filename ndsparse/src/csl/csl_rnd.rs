use crate::csl::{correct_offs_len, manage_last_offset, outermost_stride, Csl, CslError};
use cl_traits::{Push, Storage};
use core::cmp::Ordering;
use rand::{
  distributions::{Distribution, Uniform},
  Rng,
};

#[derive(Debug)]
pub(crate) struct CslRnd<'a, DS, IS, OS, R, const D: usize> {
  csl: &'a mut Csl<DS, IS, OS, D>,
  nnz: usize,
  rng: &'a mut R,
}

impl<'a, DATA, DS, IS, OS, R, const D: usize> CslRnd<'a, DS, IS, OS, R, D>
where
  DS: AsMut<[DATA]> + AsRef<[DATA]> + Push<Input = DATA> + Storage<Item = DATA>,
  IS: AsMut<[usize]> + AsRef<[usize]> + Push<Input = usize>,
  R: Rng,
  OS: AsMut<[usize]> + AsRef<[usize]> + Push<Input = usize>,
{
  #[inline]
  pub(crate) fn new(
    csl: &'a mut Csl<DS, IS, OS, D>,
    nnz: usize,
    rng: &'a mut R,
  ) -> crate::Result<Self> {
    if nnz > crate::utils::max_nnz(&csl.dims) {
      return Err(CslError::NnzGreaterThanMaximumNnz.into());
    }
    let _ = manage_last_offset(&mut csl.offs)?;
    Ok(Self { csl, nnz, rng })
  }

  #[inline]
  pub(crate) fn fill<F>(mut self, cb: F) -> crate::Result<()>
  where
    F: FnMut(&mut R, [usize; D]) -> DATA,
  {
    let last_dim_idx = if self.csl.dims.is_empty() {
      return Ok(());
    } else {
      self.csl.dims.len() - 1
    };
    self.fill_offs(last_dim_idx).ok_or(crate::Error::UnknownError)?;
    self.fill_indcs(last_dim_idx).ok_or(crate::Error::UnknownError)?;
    self.fill_data(cb, last_dim_idx).ok_or(crate::Error::UnknownError)?;
    Ok(())
  }

  #[inline]
  fn fill_data<F>(&mut self, mut cb: F, last_dim_idx: usize) -> Option<()>
  where
    F: FnMut(&mut R, [usize; D]) -> DATA,
  {
    let data = &mut self.csl.data;
    let indcs = self.csl.indcs.as_ref();
    let orig_dims = self.csl.dims;
    let outermost_stride = outermost_stride(&orig_dims);
    let rng = &mut self.rng;

    for (line_idx, offset) in self.csl.offs.as_ref().windows(2).enumerate() {
      let mut dims = orig_dims;
      *dims.first_mut()? = if outermost_stride == 0 { 0 } else { line_idx % outermost_stride };
      let iter = dims.iter_mut().zip(orig_dims.iter()).skip(1).rev().skip(1);
      for (dim, &orig_dim) in iter {
        *dim = if orig_dim == 0 { 0 } else { line_idx % orig_dim };
      }
      let range = *offset.first()?..*offset.get(1)?;
      for innermost_idx in indcs.get(range)?.iter().copied() {
        *dims.get_mut(last_dim_idx)? = innermost_idx;
        let _ = data.push(cb(rng, dims)).ok()?;
      }
    }

    Some(())
  }

  #[inline]
  fn fill_indcs(&mut self, last_dim_idx: usize) -> Option<()> {
    let dims = &self.csl.dims;
    let rng = &mut self.rng;
    let indcs = &mut self.csl.indcs;
    for offset in self.csl.offs.as_ref().windows(2) {
      let mut counter = 0;
      let line_nnz = offset.get(1)? - offset.first()?;
      while counter < line_nnz {
        let rnd = rng.gen_range(0, *dims.get(last_dim_idx)?);
        if !indcs.as_ref().get(*offset.first()?..)?.contains(&rnd) {
          let _ = indcs.push(rnd).ok()?;
          counter += 1;
        }
      }
      indcs.as_mut().get_mut(*offset.first()?..)?.sort_unstable();
    }
    Some(())
  }

  #[inline]
  fn fill_offs(&mut self, last_dim_idx: usize) -> Option<()> {
    let nnz = self.nnz;
    for _ in 1..correct_offs_len(&self.csl.dims).ok()? {
      let _ = self.csl.offs.push(0).ok()?;
    }
    let fun = |idl, _, s: &mut Self| Some(Uniform::from(0..=idl).sample(s.rng));
    let mut last_visited_off = self.do_fill_offs(last_dim_idx, fun)?;
    loop {
      if *self.csl.offs.as_ref().get(last_visited_off)? >= nnz {
        if let Some(slice) = self.csl.offs.as_mut().get_mut(last_visited_off..) {
          slice.iter_mut().for_each(|off| *off = nnz);
        }
        break;
      }
      let mut offs_adjustment = 0;
      last_visited_off = self.do_fill_offs(last_dim_idx, |idl, idx, s| {
        let offs = s.csl.offs.as_mut();
        let curr = *offs.get(idx)? + offs_adjustment;
        let prev = *offs.get(idx - 1)?;
        let start = curr - prev;
        let line_nnz = Uniform::from(start..=idl).sample(s.rng);
        offs_adjustment += (line_nnz + prev) - curr;
        Some(line_nnz)
      })?;
    }
    Some(())
  }

  #[inline]
  fn do_fill_offs<F>(&mut self, last_dim_idx: usize, mut f: F) -> Option<usize>
  where
    F: FnMut(usize, usize, &mut Self) -> Option<usize>,
  {
    let nnz = self.nnz;
    let mut idx = 1;
    let mut previous_nnz = *self.csl.offs.as_ref().first()?;
    loop {
      if idx >= self.csl.offs.as_ref().len() {
        break;
      }
      match previous_nnz.cmp(&nnz) {
        Ordering::Equal => {
          break;
        }
        Ordering::Greater => {
          *self.csl.offs.as_mut().get_mut(idx - 1)? = nnz;
          break;
        }
        Ordering::Less => {
          let innermost_dim_len = *self.csl.dims.get(last_dim_idx)?;
          let line_nnz = f(innermost_dim_len, idx, self)?;
          let new_nnz = previous_nnz + line_nnz;
          *self.csl.offs.as_mut().get_mut(idx)? = new_nnz;
          previous_nnz = new_nnz;
        }
      }
      idx += 1;
    }
    Some(idx - 1)
  }
}
