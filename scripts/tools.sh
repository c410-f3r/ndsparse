#!/usr/bin/env bash

. "$(dirname "$0")/commons.sh" --source-only

cargo fmt  --all -- --check

cargo clippy --all-features --all-targets -- \
    -D clippy::restriction \
    -D warnings \
    -A clippy::implicit_return \
    -A clippy::integer_arithmetic \
    -A clippy::missing_docs_in_private_items \
    -A clippy::missing_inline_in_public_items