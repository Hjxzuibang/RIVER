#!/bin/bash
# Prepare data for RIVER experiments.
# Usage: bash scripts/prepare_data.sh
#
# This script preprocesses raw h5ad files and builds prior graphs.
# Raw data should be placed in rawdata/{Dataset}/{CellType}.h5ad before running.
#
# Data sources:
#   - Norman K562:    https://doi.org/10.1038/s41588-019-0538-0
#   - Replogle K562/RPE1: https://doi.org/10.1016/j.cell.2022.05.013
#   - Nadig HepG2/Jurkat: https://doi.org/10.1038/s41588-024-02048-1

set -e

echo "============================================================"
echo "RIVER Data Preparation Pipeline"
echo "============================================================"

# Step 1: Preprocess all datasets
echo ""
echo "[Step 1] Preprocessing raw h5ad files..."
echo "  Expected layout: rawdata/{Dataset}/{CellType}.h5ad"
echo ""

if [ ! -d "rawdata" ]; then
    echo "ERROR: rawdata/ directory not found."
    echo "Please download raw h5ad files and place them under rawdata/{Dataset}/{CellType}.h5ad"
    echo ""
    echo "Expected structure:"
    echo "  rawdata/Norman/K562.h5ad"
    echo "  rawdata/Replogle/K562.h5ad"
    echo "  rawdata/Replogle/RPE1.h5ad"
    echo "  rawdata/Nadig/HepG2.h5ad"
    echo "  rawdata/Nadig/Jurkat.h5ad"
    exit 1
fi

python -m data.preprocess \
    --input-dir rawdata \
    --output-dir processed_data \
    --n-hvg 2000 \
    --pert-efficiency-threshold 1.0 \
    --min-cells-per-pert 5

echo ""
echo "[Step 2] Building prior graphs from STRING PPI..."
echo "  Prior graph construction requires data/prior_graph/build_prior_graph.py"
echo "  (See README.md for instructions)"

echo ""
echo "============================================================"
echo "Data preparation complete."
echo "Preprocessed data saved to processed_data/"
echo "============================================================"
