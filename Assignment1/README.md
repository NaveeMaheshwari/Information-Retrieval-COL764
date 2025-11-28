# Information Retrieval Assignment1

This project implements an end-to-end Information Retrieval (IR) pipeline, including vocabulary construction, inverted index building, index compression/decompression, and Boolean query retrieval.

---

## 📂 Files Overview
- **tokenize_corpus.py** → Generates vocabulary from corpus + stopwords.  
- **build_index.py** → Builds inverted index (JSON) and compressed index files.  
- **retrieval.py** → Decompresses index and performs Boolean retrieval on queries.  
- **Bash scripts:**  
  - `tokenize_corpus.sh` → Runs `tokenize_corpus.py`.  
  - `build_index.sh` → Runs `build_index.py`.  
  - `retrieval.sh` → Runs `retrieval.py`.  

---

## ⚙️ Requirements
- Allowed libraries: `os, sys, json, re, time, collections, zlib`  
- Corpus: `cord19-trec_covid-docs`  
- Stopwords: `stopwords.txt`  
- Queries: `queries.json`  

---

## 🚀 How to Run

### 1. **Generate Vocabulary**
Creates `vocab.txt` inside the given vocab directory.  

```bash
bash tokenize_corpus.sh < CORPUS_DIR > < PATH_OF_STOPWORDS_FILE > < VOCAB_DIR >
```

---

### 2. **Build Index (JSON + Compressed Files)**
Creates `index.json` and compressed files (`postings.bin`, `lexicon.tsv`, `docmap.tsv`, `meta.json`).  

```bash
bash build_index.sh < CORPUS_DIR > < VOCAB_PATH > < INDEX_DIR > < COMPRESSED_DIR >
```

---

### 3. **Boolean Retrieval**
Runs decompression + query processing. Produces results in `docid.txt`.  

```bash
bash retrieval.sh < COMPRESSED_DIR > < QUERY_FILE_PATH > < OUTPUT_DIR > < STOPWORD_FILE_PATH >
```


---

## 📑 Output Files
- `vocab_dir/vocab.txt` → Vocabulary list.  
- `index_dir/index.json` → Full inverted index (uncompressed).  
- `compressed_index_dir/postings.bin` → Compressed postings.  
- `compressed_index_dir/lexicon.tsv` → Term → offset mapping.  
- `compressed_index_dir/docmap.tsv` → Integer docID → original docID.  
- `compressed_index_dir/meta.json` → Format description.  
- `output_dir/docid.txt` → Final ranked retrieval output.  

---
