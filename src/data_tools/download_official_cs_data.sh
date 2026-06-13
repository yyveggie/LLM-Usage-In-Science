#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://raw.githubusercontent.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers/main"
# Resolve the project root (two levels up from src/data_tools) and write into data/official_data.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/data/official_data"

mkdir -p "$DATA_DIR/distribution"
mkdir -p "$DATA_DIR/validation_data/CS"

curl -fL "$BASE_URL/distribution/CS.parquet" \
  -o "$DATA_DIR/distribution/CS.parquet"

for alpha in 0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25; do
  curl -fL "$BASE_URL/data/validation_data/CS/ground_truth_alpha_${alpha}.parquet" \
    -o "$DATA_DIR/validation_data/CS/ground_truth_alpha_${alpha}.parquet"
done
