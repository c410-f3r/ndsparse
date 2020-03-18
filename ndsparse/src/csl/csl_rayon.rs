use crate::{
  csl::{CsIterRef, CslIterMut, CslMut, CslRef},
  Dims, ParallelIteratorWrapper, ParallelProducerWrapper,
};
use rayon::iter::{
  plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
  IndexedParallelIterator, ParallelIterator,
};

macro_rules! create_rayon_iter {
  ($csl_rayon_iter:ident, $ref:ident) => {
    impl<'a, DA, T> ParallelIterator for ParallelIteratorWrapper<$csl_rayon_iter<'a, DA, T>>
    where
      DA: Dims + Send + Sync,
      T: Send + Sync + 'a,
    {
      type Item = $ref<'a, DA, T>;
      fn drive_unindexed<C>(self, consumer: C) -> C::Result
      where
        C: UnindexedConsumer<Self::Item>,
      {
        bridge(self, consumer)
      }

      fn opt_len(&self) -> Option<usize> {
        Some(self.0.len())
      }
    }

    impl<'a, DA, T> IndexedParallelIterator for ParallelIteratorWrapper<$csl_rayon_iter<'a, DA, T>>
    where
      DA: Dims + Send + Sync,
      T: Send + Sync + 'a,
    {
      fn with_producer<Cb>(self, callback: Cb) -> Cb::Output
      where
        Cb: ProducerCallback<Self::Item>,
      {
        callback.callback(ParallelProducerWrapper(self.0))
      }

      fn len(&self) -> usize {
        ExactSizeIterator::len(&self.0)
      }

      fn drive<C>(self, consumer: C) -> C::Result
      where
        C: Consumer<Self::Item>,
      {
        bridge(self, consumer)
      }
    }

    impl<'a, DA, T> IntoIterator for ParallelProducerWrapper<$csl_rayon_iter<'a, DA, T>>
    where
      DA: Dims,
      T: 'a,
    {
      type IntoIter = $csl_rayon_iter<'a, DA, T>;
      type Item = <Self::IntoIter as Iterator>::Item;

      fn into_iter(self) -> Self::IntoIter {
        self.0
      }
    }

    impl<'a, DA, T> Producer for ParallelProducerWrapper<$csl_rayon_iter<'a, DA, T>>
    where
      DA: Dims,
      T: 'a,
    {
      type IntoIter = $csl_rayon_iter<'a, DA, T>;
      type Item = <Self::IntoIter as Iterator>::Item;

      fn into_iter(self) -> Self::IntoIter {
        self.0
      }

      fn split_at(self, i: usize) -> (Self, Self) {
        let [a, b] = self.0.split_at(i);
        (ParallelProducerWrapper(a), ParallelProducerWrapper(b))
      }
    }
  };
}

create_rayon_iter!(CsIterRef, CslRef);
create_rayon_iter!(CslIterMut, CslMut);
