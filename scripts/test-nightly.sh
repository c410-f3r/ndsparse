#!/usr/bin/env bash

. "$(dirname "$0")/test-common.sh" --source-only

# `check` because of https://github.com/PyO3/pyo3/issues/340
check_package_generic "ndsparse-bindings"

test_package_generic "ndsparse"

test_package_with_feature "ndsparse" "const-generics"

run_package_example "ndsparse-examples" "dynamic_arrays"