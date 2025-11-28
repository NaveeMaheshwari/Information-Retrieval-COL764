#!/bin/bash
# phrase_search.sh
# Usage: ./phrase_search.sh <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR>"
    exit 1
fi

INDEX_DIR=$1
QUERY_FILE=$2
OUTPUT_DIR=$3

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Running Phrase Search..."
python3 phrase_search.py "$INDEX_DIR" "$QUERY_FILE" "$OUTPUT_DIR"
