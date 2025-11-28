#!/bin/bash
# feedback.sh
# Usage: ./feedback.sh <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>"
    exit 1
fi

INDEX_DIR=$1
QUERY_FILE=$2
OUTPUT_DIR=$3
TOPK=$4

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Running Pseudo-Relevance Feedback Retrieval..."
python3 feedback_retrieval.py "$INDEX_DIR" "$QUERY_FILE" "$OUTPUT_DIR" "$TOPK"
