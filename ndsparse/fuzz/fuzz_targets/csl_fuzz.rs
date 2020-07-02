#![no_main]

use libfuzzer_sys::fuzz_target;
use ndsparse::csl::CslVec;
use rayon::prelude::*;

type Array = [usize; 3];

#[derive(Debug, arbitrary::Arbitrary)]
struct Values {
    data: Vec<i32>,
    dims: Array,
    indcs: Vec<usize>,
    line: Array,
    offs: Vec<usize>,
    value: Array,
}

fuzz_target!(|values: Values| {
    let csl: CslVec<_, i32>;

    csl = if let Ok(r) = CslVec::new(values.dims, values.data, values.indcs, values.offs) {
        r
    }
    else {
        return;
    };

    let _ = csl.line(values.line);

    let _ = csl.value(values.value);

    if let Ok(r) = csl.outermost_line_iter() {
        r.for_each(|_| {});
    }
    else {
        return;
    };

    if let Ok(r) = csl.outermost_line_rayon_iter() {
        r.enumerate().for_each(|(_, _)| {});
    }
    else {
        return;
    };
});
