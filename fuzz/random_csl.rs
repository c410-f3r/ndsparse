//! Random CSL

#![allow(missing_docs)]
#![no_main]

use libfuzzer_sys::fuzz_target;
use ndsparse::coo::CooVec;
use rand::rngs::mock::StepRng;

fuzz_target!(|values: ([usize; 2], usize)| {
  let (dims, nnz) = values;
  let _ = CooVec::new_controlled_random_rand(dims, nnz, &mut StepRng::new(0, 0), |_, _| 0);
});
