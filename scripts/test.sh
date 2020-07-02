#!/usr/bin/env bash

. "$(dirname "$0")/test-common.sh" --source-only

test_package_with_feature "ndsparse" "alloc"
test_package_with_feature "ndsparse" "std"
test_package_with_feature "ndsparse" "with-rand"
test_package_with_feature "ndsparse" "with-rayon"
test_package_with_feature "ndsparse" "with-serde"