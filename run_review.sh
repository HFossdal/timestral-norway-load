#!/usr/bin/env bash
# Reproduce the technical-review corrections. Cached real-data forecasts are
# required; synthetic inference is rerun into an isolated versioned cache.
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
python -m pytest tests -q
python -m src.review
python -m src.report --origin noon_before --context 2688
python -m src.report --origin noon_before --context 672
python -m src.report --origin midnight --context 672
python -m src.synthetic --models timestral-209m chronos2 timesfm3 --all --seeds 0 1 2 3 4
python -m src.synthetic --report
python -m src.review_docs
python -m src.headline
