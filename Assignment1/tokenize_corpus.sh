#!/bin/bash

# Check if correct number of arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <CORPUS_FILE> <STOPWORDS_FILE> <VOCAB_DIR>"
    exit 1
fi

CORPUS=$1
STOPWORDS=$2
VOCAB=$3

# Run the Python script with arguments
python3 tokenize_corpus.py "$CORPUS" "$STOPWORDS" "$VOCAB" 
