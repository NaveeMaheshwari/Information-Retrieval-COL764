#!/bin/bash
# bm25_retrieval.sh
# Usage: ./bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>"
    exit 1
fi

INDEX_DIR=$1
QUERY_FILE=$2
OUTPUT_DIR=$3
TOPK=$4

# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"

echo "Running BM25 retrieval..."
python3 bm25_retrieval.py "$INDEX_DIR" "$QUERY_FILE" "$OUTPUT_DIR" "$TOPK"
