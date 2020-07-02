#!/usr/bin/env bash

set -eux

export RUST_BACKTRACE=full
export RUSTFLAGS='
    -D bad_style
    -D future_incompatible
    -D missing_debug_implementations
    -D missing_docs
    -D nonstandard_style
    -D rust_2018_compatibility
    -D rust_2018_idioms
    -D trivial_casts
    -D unsafe_code
    -D unused_lifetimes
    -D unused_qualifications
    -D warnings
'

check_package_generic() {
    local package=$1

    /bin/echo -e "\e[0;33m***** Checking ${package} without features *****\e[0m\n"
    cargo check --manifest-path "$(dirname "$0")/../${package}"/Cargo.toml --no-default-features

    /bin/echo -e "\e[0;33m***** Checking ${package} with all features *****\e[0m\n"
    cargo check --all-features --manifest-path "$(dirname "$0")/../${package}"/Cargo.toml
}

run_package_example() {
    local package=$1
    local example=$2

    /bin/echo -e "\e[0;33m***** Running example ${example} of ${package}  *****\e[0m\n"
    cargo run --all-features --example $example --manifest-path "$(dirname "$0")/../${package}"/Cargo.toml
}

test_package_generic() {
    local package=$1

    /bin/echo -e "\e[0;33m***** Testing ${package} without features *****\e[0m\n"
    cargo test --manifest-path "$(dirname "$0")/../${package}"/Cargo.toml --no-default-features

    /bin/echo -e "\e[0;33m***** Testing ${package} with all features *****\e[0m\n"
    cargo test --all-features --manifest-path "$(dirname "$0")/../${package}"/Cargo.toml
}

test_package_with_feature() {
    local package=$1
    local feature=$2

    /bin/echo -e "\e[0;33m***** Testing ${package} with feature '${feature}' *****\e[0m\n"
    cargo test --manifest-path "$(dirname "$0")/../${package}"/Cargo.toml --features "${feature}" --no-default-features
}