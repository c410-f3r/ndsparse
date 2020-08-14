#![no_main]

use libfuzzer_sys::fuzz_target;
use ndsparse::coo::CooVec;

type Array = [usize; 3];

#[derive(Debug, arbitrary::Arbitrary)]
struct Values {
  data: Vec<(Array, i32)>,
  dims: Array,
  nnz: usize,
  value: Array,
}

fuzz_target!(|values: Values| {
  let real_data = values.data.into_iter().map(|x| (x.0.into(), x.1)).collect::<Vec<_>>();

  let coo: CooVec<_, i32> = if let Ok(r) = CooVec::new(values.dims, real_data) {
    r
  } else {
    return;
  };

  let _ = coo.value(values.value);
});
