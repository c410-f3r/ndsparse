//! Random COO

#![no_main]

use libfuzzer_sys::fuzz_target;
use ndsparse::csl::CslVec;
use rand::rngs::mock::StepRng;

fuzz_target!(|values: ([usize; 2], usize)| {
  let (dims, nnz) = values;
  let _ = CslVec::new_controlled_random_rand(dims, nnz, &mut StepRng::new(0, 0), |_, _| 0);
});
