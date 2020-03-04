# Ndsparse 

This crate provides structures to store and retrieve N-dimensional sparse data.
Well, not any `N ∈ ℕ` but any natural number that fits into the pointer size of the machine that you are using. E.g., an 8-bit microcontroller can manipulate any sparse structure with up to 255 dimensions.

## Example

```rust
use ndsparse::{coo::CooArray, csl::CslVec};

fn main() {
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
- Coorinate format (COO)

## Optional features

- Bindings (py03)
- Deserialization/Serialization (serde)
- Dynamic arrays (ArrayVec, SmallVec and StaticVec)
- Parallel iterators (rayon)
- Random instances (rand)
- `std` or `alloc`

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