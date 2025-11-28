#!/bin/bash

# Usage: ./build_index.sh <CORPUS_FILE> <VOCAB_FILE> <INDEX_JSON_FILE> <COMPRESSED_INDEX_DIR>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <CORPUS_FILE> <VOCAB_FILE> <INDEX_JSON_FILE> <COMPRESSED_INDEX_DIR>"
    exit 1
fi

CORPUS_FILE=$1
VOCAB_FILE=$2
INDEX_JSON_FILE=$3
COMPRESSED_DIR=$4

echo "Building inverted index..."
python3 build_index.py "$CORPUS_FILE" "$VOCAB_FILE" "$INDEX_JSON_FILE" "$COMPRESSED_DIR"

if [ $? -eq 0 ]; then
    echo "Indexing and compression completed successfully!"
else
    echo "Error during indexing."
    exit 1
fi  
