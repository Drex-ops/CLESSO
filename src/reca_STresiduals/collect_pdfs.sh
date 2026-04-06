#!/usr/bin/env bash
##############################################################################
##
## collect_pdfs.sh
##
## Find all unique model-run folders under output/ (matching the pattern
## GROUP[_SUFFIX]_YYYYMMDDTHHMMSS) and copy every PDF from each into a
## single flat directory: output/model_output_viz/
##
## To avoid name collisions, PDFs are prefixed with the run-folder name:
##   AVES_20260311T195250/some_plot.pdf  ->  AVES_20260311T195250__some_plot.pdf
##
## Usage:
##   cd src/reca_STresiduals
##   bash collect_pdfs.sh
##
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
DEST_DIR="${OUTPUT_DIR}/model_output_viz"

# Create (or clear) destination folder
mkdir -p "$DEST_DIR"

echo "=== Collecting PDFs from model-run folders ==="
echo "  Source : ${OUTPUT_DIR}"
echo "  Dest   : ${DEST_DIR}"
echo

count=0
folders=0

# Find run folders matching GROUP[_SUFFIX]_YYYYMMDDTHHMMSS pattern
for run_dir in "$OUTPUT_DIR"/*/; do
    # Strip trailing slash and get basename
    folder_name="$(basename "$run_dir")"

    # Match pattern: WORD(s) followed by _YYYYMMDDTHHmmSS
    if [[ "$folder_name" =~ ^[A-Z]+(_[A-Z]+)?_[0-9]{8}T[0-9]{6}$ ]]; then
        folders=$((folders + 1))
        echo "  Run folder: ${folder_name}"

        for pdf in "$run_dir"*.pdf; do
            [ -f "$pdf" ] || continue   # skip if glob didn't match
            pdf_name="$(basename "$pdf")"
            dest_file="${DEST_DIR}/${folder_name}__${pdf_name}"
            cp "$pdf" "$dest_file"
            count=$((count + 1))
        done
    fi
done

echo
echo "=== Done ==="
echo "  Run folders found : ${folders}"
echo "  PDFs copied       : ${count}"
echo "  Output            : ${DEST_DIR}"
