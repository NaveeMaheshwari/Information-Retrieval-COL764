#!/bin/bash

# Usage: ./retrieval.sh <COMPRESSED_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <STOPWORDS_FILE>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <COMPRESSED_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <STOPWORDS_FILE>"
    exit 1
fi

COMPRESSED_DIR=$1
QUERY_FILE=$2
OUTPUT_DIR=$3
STOPWORDS_FILE=$4

echo "Running Boolean Retrieval..."
python3 retrieval.py "$COMPRESSED_DIR" "$QUERY_FILE" "$OUTPUT_DIR" "$STOPWORDS_FILE"

if [ $? -eq 0 ]; then
    echo "Retrieval completed successfully!"
else
    echo "Error during retrieval."
    exit 1
fi
