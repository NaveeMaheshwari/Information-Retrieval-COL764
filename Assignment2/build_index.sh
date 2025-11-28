#!/bin/bash

# Usage: ./build_index.sh <CORPUS_FILE> <VOCAB_FILE> <INDEX_JSON_FILE>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <CORPUS_DIR> <VOCAB_FILE> <INDEX_JSON_DIT>"
    exit 1
fi

CORPUS_DIR=$1
VOCAB_FILE=$2
INDEX_DIR=$3

echo "Building inverted index..."
python3 build_index.py "$CORPUS_DIR" "$VOCAB_FILE" "$INDEX_DIR"
if [ $? -eq 0 ]; then
    echo "Indexing completed successfully!"
else
    echo "Error during indexing."
    exit 1
fi  
