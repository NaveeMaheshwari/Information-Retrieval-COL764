#!/bin/bash

# Check if correct number of arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <CORPUS_FILE> <VOCAB_DIR>"
    exit 1
fi

CORPUS=$1
VOCAB=$2

# Run the Python script with arguments
python3 tokenize_corpus.py "$CORPUS" "$VOCAB" 
