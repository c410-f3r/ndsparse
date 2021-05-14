#!/usr/bin/env bash

set -euxo pipefail

cargo fuzz run --fuzz-dir ndsparse-fuzz coo -- -runs=100000
cargo fuzz run --fuzz-dir ndsparse-fuzz csl -- -runs=100000