# Task 1 — Bi-Encoder Dense Retriever (DistilBERT)

This task implements **Dense Retrieval** using a **Bi-Encoder architecture** based on the **DistilBERT** model.
It retrieves relevant passages from the **MS MARCO** collection using **Pyserini’s FAISS-based dense index**.

---

## 📘 Overview

The system performs the following:
1. Loads the prebuilt **FAISS dense index** for MS MARCO passages.
2. Encodes input queries using the **DistilBERT (TCT-ColBERT)** query encoder.
3. Searches for the top-*k* most relevant passages using inner-product similarity.
4. Writes all ranked results to a TREC-formatted output file.

This approach enables **semantic retrieval** by comparing dense embeddings rather than relying purely on term matching.

---

## ⚙️ Requirements

Install all dependencies before running:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```
pyserini==0.20.0
torch>=2.0.0
transformers>=4.40.0
faiss-cpu>=1.7.4
numpy>=1.26.0
tqdm>=4.66.0
pytrec_eval>=0.5
```

---

## 📂 Directory Structure

```
.
├── team21/
│   ├── task1.py
│   ├── results_final/
│   │   └── task1.txt
│   └── queries.json
├── qrels.txt
├── evaluate.py
└── README.md
```

---

## 🚀 Running the Code

### 🔹 Step 1: Execute the main evaluation script

Run the following command from the root directory to evaluate **Task 1 (DistilBERT Dense Retriever)** along with other tasks:

```bash
python3 evaluate.py --team_dir team21 --ms_marco_qrels_file_path qrels.txt --cord_qrels_file_path qrels_task3.txt --msmarco_distilbert_index_file_path /shared/indexes/msmarco/faiss-flat.msmarco-v1-passage.distilbert-dot-margin_mse-t2/faiss-flat.msmarco-v1-passage.distilbert-dot-margin_mse-t2.20210316.d44c3a --msmarco_splade_index_file_path /shared/indexes/msmarco/lucene-inverted.msmarco-v1-passage.splade-pp-ed --t12_query_file_path queries.json --t3_query_file_path queries_task3.json --task1_distilbert_output_file results_final/task1.txt --task2_splade_output_file results_final/task2.txt --task3_bm25_output_file results_final/task3_bm25.txt --task3_rm3_output_file results_final/task3_rm3.txt --task3_qrf_output_file results_final/task3_qrf.txt --k 50
```

### 🔹 Step 2: Run only Task 1 (optional standalone mode)

If you wish to run **only Task 1**, you can execute:

```bash
python3 team21/task1.py <index_path> <query_path> <output_file> <k>
```

Example:
```bash
python3 team21/task1.py /shared/indexes/msmarco/faiss-flat.msmarco-v1-passage.distilbert-dot-margin_mse-t2/faiss-flat.msmarco-v1-passage.distilbert-dot-margin_mse-t2.20210316.d44c3a queries.json results_final/task1.txt 50
```

---

## 📈 Output

The results are written to:
```
results_final/task1.txt
```

Each line in the output follows **TREC format**:
```
<query_id> <doc_id> <rank> <score>
```

Example:
```
101 Q0 D134 1 12.984
101 Q0 D297 2 11.736
```

---

## 🧠 Notes

- The query encoder used is:
  ```
  castorini/tct_colbert-msmarco
  ```
- The index used is the **FAISS-flat DistilBERT-dot-margin_mse-t2** variant.
- The top-*k* parameter controls how many top results to retrieve for each query (default = 50).
- Ensure that the prebuilt index path provided in the command exists and is accessible on your system.

---

## 🧩 Authors

**Team 21**

- Naveeta Maheshwari — 2024AIZ8309  
- Sagar Singh — 2025SIY7574  

---

## 📚 References

- Pyserini Toolkit: [https://github.com/castorini/pyserini](https://github.com/castorini/pyserini)  
- TCT-ColBERT DistilBERT Model: [https://huggingface.co/castorini/tct_colbert-msmarco](https://huggingface.co/castorini/tct_colbert-msmarco)  
- FAISS Library: [https://faiss.ai](https://faiss.ai)
