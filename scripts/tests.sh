#!/usr/bin/env bash

. "$(dirname "$0")/commons.sh" --source-only

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

# `check` because of https://github.com/PyO3/pyo3/issues/340
check_package_generic "ndsparse-bindings"

test_package_generic "ndsparse"

test_package_with_feature "ndsparse" "alloc"
test_package_with_feature "ndsparse" "std"
test_package_with_feature "ndsparse" "with-rand"
test_package_with_feature "ndsparse" "with-rayon"
test_package_with_feature "ndsparse" "with-serde"

run_package_example "ndsparse-examples" "dynamic_arrays"
