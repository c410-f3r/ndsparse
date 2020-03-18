#!/usr/bin/env bash

. ./scripts/tests-common.sh --source-only

#test_package_generic "ndsparse-bindings"

test_package_with_feature "ndsparse" "alloc"
test_package_with_feature "ndsparse" "with_arrayvec"
test_package_with_feature "ndsparse" "with_rand"
test_package_with_feature "ndsparse" "with_rayon"
test_package_with_feature "ndsparse" "with_serde"
test_package_with_feature "ndsparse" "with_smallvec"