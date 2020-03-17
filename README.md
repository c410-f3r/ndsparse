# ndsparse 

[![CI](https://github.com/c410-f3r/ndsparse/workflows/CI/badge.svg)](https://github.com/c410-f3r/ndsparse/actions?query=workflow%3ACI)
[![crates.io](https://img.shields.io/crates/v/ndsparse.svg)](https://crates.io/crates/ndsparse)
[![Documentation](https://docs.rs/ndsparse/badge.svg)](https://docs.rs/ndsparse)
[![License](https://img.shields.io/badge/license-APACHE2-blue.svg)](./LICENSE)
[![Rustc](https://img.shields.io/badge/rustc-nightly-lightgray")](https://rustup.rs/)

Structures to store and retrieve N-dimensional sparse data. Well, not any `N ∈ ℕ` but any natural number that fits into the pointer size of the machine that you are using. E.g., an 8-bit microcontroller can manipulate any sparse structure with up to 255 dimensions.

For those that might be wondering about why this crate should be used, it generally comes down to space-efficiency, ergometrics and retrieving speed. The following snippet shows some use-cases for potential replacement with `_cube_of_vecs` being the most inefficient of all.

```rust
let _vec_of_options: Vec<Option<i32>> = Default::default();
let _matrix_of_options: [Option<Option<[Option<i32>; 8]>>; 16] = Default::default();
let _cube_of_vecs: Vec<Vec<Vec<i32>>> = Default::default();
// The list worsens exponentially for higher dimensions
```

See [this blog post](https://c410-f3r.github.io/posts/sparse-multidimensional-structures-written-in-rust/) for more information.

## Example

```rust
use ndsparse::{coo::CooArray, csl::CslVec};

fn main() {
  // A CSL and COO cube.
  //
  //      ___ ___
  //    /   /   /\
  //   /___/___/ /\
  //  / 1 /   /\/2/
  // /_1_/___/ /\/
  // \_1_\___\/ /
  //  \___\___\/
  let coo = CooArray::new([2, 2, 2], [([0, 0, 0].into(), 1.0), ([1, 1, 1].into(), 2.0)]);
  let mut csl = CslVec::default();
  csl
    .constructor()
    .next_outermost_dim(2)
    .push_line(&[1.0], &[0])
    .next_outermost_dim(2)
    .push_empty_line()
    .next_outermost_dim(2)
    .push_empty_line()
    .push_line(&[2.0], &[1]);
  assert_eq!(coo.value([0, 0, 0]), csl.value([0, 0, 0]));
  assert_eq!(coo.value([1, 1, 1]), csl.value([1, 1, 1]));
}
```

## Supported structures

- Compressed Sparse Line (CSL)
- Coordinate format (COO)

## Optional features

- `alloc`
- Bindings (Py03, wasm-bindgen)
- Deserialization/Serialization (serde)
- Dynamic arrays (ArrayVec, SmallVec and StaticVec)
- Parallel iterators (rayon)
- Random instances (rand)

## Nightly compiler

For truly N-dimension structures, constant generics are being used and this feature can only be accessible with a nightly Rustc compiler. 

## Future

Although CSR and COO are general sparse structures, they aren't good enough for certain situations, threfore, the existence of DIA, JDS, ELL, LIL, DOK and many others.

If there are enough interest, the mentioned sparse storages might be added at some point in the future.

## Algebra library

This project isn't and will never be a sparse algebra library because of its own self-contained responsability and complexity. Futhermore, a good implementation of such library would require a titanic amout of work and research for different algorithms, operations, decompositions, solvers and hardwares.

## Alternatives

One of these libraries might suit you better:

* [`sprs`][sprs]: Sparse linear algebra.
* [`ndarray`][ndarray]: Dense N-dimensional operations.
* [`nalgebra`][nalgebra]: Dense linear albegra.

[nalgebra]: https://github.com/rustsim/nalgebra
[ndarray]: https://github.com/rust-ndarray/ndarray
[sprs]: https://github.com/vbarrielle/sprs