#!/usr/bin/env bash
# scripts/bench/: the parsers, scorers, generators and reference files of the baseline tools, offline (no Spark).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
for t in attn_baseline divergence longctx_suite; do python3 "scripts/bench/$t.py" selftest; done
