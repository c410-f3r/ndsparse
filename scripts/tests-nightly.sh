#!/usr/bin/env bash

. ./scripts/tests-common.sh --source-only

# `check` because of https://github.com/PyO3/pyo3/issues/340
check_package_generic "ndsparse-bindings"

test_package_generic "ndsparse"

test_package_with_feature "ndsparse" "const_generics"
test_package_with_feature "ndsparse" "with_staticvec"

run_package_example "examples" "dynamic_arrays"
