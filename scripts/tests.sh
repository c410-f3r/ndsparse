#!/usr/bin/env bash

set -euxo pipefail

cargo install --git https://github.com/c410-f3r/rust-tools --force

rt='rust-tools --template you-rust'

export CARGO_TARGET_DIR="$($rt target-dir)"
export RUST_BACKTRACE=1
export RUSTFLAGS="$($rt rust-flags)"

$rt rustfmt
$rt clippy -Aclippy::default_numeric_fallback,-Aclippy::integer_arithmetic,-Aunused_crate_dependencies

$rt test-generic ndsparse
$rt test-with-features ndsparse alloc
$rt test-with-features ndsparse std
$rt test-with-features ndsparse with-rand
$rt test-with-features ndsparse with-rayon
$rt test-with-features ndsparse with-serde

$rt test-with-features ndsparse-bindings with-wasm-bindgen
