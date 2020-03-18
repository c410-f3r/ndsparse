#!/usr/bin/env bash

. ./scripts/tests-common.sh --source-only

test_package_generic "ndsparse"

test_package_with_feature "ndsparse" "const_generics"
test_package_with_feature "ndsparse" "with_staticvec"

run_package_example "examples" "dynamic_arrays"
