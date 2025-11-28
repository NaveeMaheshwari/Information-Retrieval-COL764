Information Retrieval Assignment 2 - Team 21

Overview
This project implements:
- Vocabulary generation from a JSONL corpus
- Inverted index construction with term positions
- Vector Space Model (VSM) retrieval
- BM25 retrieval
- Phrase search (exact phrase using positions)
- Pseudo-relevance feedback retrieval (Rocchio-style over VSM)

Requirements
- Ubuntu/Linux with Python 3.9+ (Python 3.12 supported with spaCy >= 3.7)
- Packages: spacy
m

Input Data Formats
- Corpus directory: contains one or more .jsonl files. Each line is a JSON object with at least:
  { "doc_id": "<string>", ... other text fields ... }
  All non-empty fields except doc_id are tokenized and indexed. Up to 1000 documents are processed.
- Queries file: a JSONL file with one JSON object per line:
  { "query_id": <int or string>, "title": "<query text>" }
  The code tries UTF-8 first and falls back to UTF-16 if needed.

Outputs
- Index directory files (created by build_index.py):
  index.json   -> inverted index with term -> { df, postings{ doc_id -> { tf, pos[] } } }
  vsm.json     -> { N, doc_vector_lengths }
  bm25.json    -> { N, avgdl, doc_lengths }
- Retrieval outputs (saved in specified output directory):
  vsm_docids.txt            -> tab-separated: query_id, doc_id, rank, score
  bm25_retrieval_docids.txt -> tab-separated: query_id, doc_id, rank, score
  phrase_search_docids.txt  -> tab-separated: query_id, doc_id, rank, 1
  feedback_docids.txt       -> tab-separated: query_id, doc_id, rank, score

Ubuntu/Linux Usage
- Verify Python and install dependencies:
  python3 -V
  python3 -m pip install --upgrade pip
  python3 -m pip install "spacy>=3.7" tqdm

Run via Python directly
1) Build vocabulary
   python3 tokenize_corpus.py <CORPUS_DIR> <VOCAB_DIR>
   Example:
   python3 tokenize_corpus.py data/corpus vocab

2) Build index
   python3 build_index.py <CORPUS_DIR> <VOCAB_DIR>/vocab.txt <INDEX_DIR>
   Example:
   python3 build_index.py data/corpus vocab/vocab.txt index

3) Run VSM retrieval (top-k)
   python3 vsm.py <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>
   Example:
   python3 vsm.py index data/queries.json outputs 20
   Output: outputs/vsm_docids.txt

4) Run BM25 retrieval (top-k)
   python3 bm25_retrieval.py <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>
   Example:
   python3 bm25_retrieval.py index data/queries.json outputs 20
   Output: outputs/bm25_retrieval_docids.txt

5) Run phrase search (exact phrase)
   python3 phrase_search.py <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR>
   Example:
   python3 phrase_search.py index data/queries.json outputs
   Output: outputs/phrase_search_docids.txt

6) Run feedback retrieval (pseudo-relevance feedback over VSM, top-k)
   python3 feedback_retrieval.py <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>
   Example:
   python3 feedback_retrieval.py index data/queries.json outputs 20
   Output: outputs/feedback_docids.txt

Run via provided .sh scripts
- Make executable (one-time):
  chmod +x *.sh
- Execute:
  ./tokenize_corpus.sh data/corpus vocab
  ./build_index.sh data/corpus vocab/vocab.txt index
  ./vsm.sh index data/queries.json outputs 20
  ./bm25_retrieval.sh index data/queries.json outputs 20
  ./phrase_search.sh index data/queries.json outputs
  ./feedback.sh index data/queries.json outputs 20

Notes
- Tokenization uses spaCy's blank English model (no model download needed).
- Query file encoding: code attempts UTF-8 first, then UTF-16.
- Ranking limits: top-k is controlled by the last argument for VSM/BM25/feedback.
- Phrase search emits a fixed score of 1 for exact matches and ranks by order found.
- build.sh is a no-op placeholder on Linux.
