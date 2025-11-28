import time
import json
import re
from collections import Counter
from typing import Dict, List, Tuple
import torch
from pyserini.search.lucene import LuceneSearcher # type: ignore
from sentence_transformers import SentenceTransformer, util # type: ignore


DEFAULT_INITIAL_K = 20
DEFAULT_EXPANSION_TOPK = 3
DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------- Helper: Loading ----------
def load_queries(query_path: str) -> List[Dict[str, str]]:
    queries = []
    with open(query_path, "r", encoding="utf-16") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id"))
            text_parts = []
            for field in ["title", "description", "narrative"]:
                value = obj.get(field)
                if value and isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
            query_text = " ".join(text_parts)
            if qid:
                queries.append({"query_id": qid, "query_text": query_text})
    return queries

# ---------- Helper: term extraction ----------
def extract_candidate_terms(docs: List[str], top_n: int = 20) -> List[str]:
    """Extract top frequent non-stopword tokens (>=4 chars) from text."""
    text = " ".join(docs).lower()
    words = re.findall(r'\b[a-z]{4,}\b', text)
    stopwords = {
        'with','this','that','from','which','their','have','been','will','were',
        'such','also','these','those','some','into','about','because','during',
        'after','when','then','them','they','there','here','where','your','using',
        'used','for','and','the','was','has','had','very'
    }
    words = [w for w in words if w not in stopwords]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_n)]

# ---------- Helper: dense reformulation ----------
def dense_reformulate_query(query: str, candidates: List[str], model: SentenceTransformer, topk: int = DEFAULT_EXPANSION_TOPK) -> str:
    """Pick top-k semantically closest terms to query and append them."""
    if not candidates:
        return query
    q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cand_embs = model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, cand_embs)[0]
    top_indices = torch.topk(sims, k=min(topk, len(candidates))).indices.tolist()
    expansions = [candidates[i] for i in top_indices]
    return query + " " + " ".join(expansions)


# ---------- Main callable ----------
def task3_improve( query_path: str, bm25_output_file: str, rm3_output_file: str, qrf_output_file: str, k: int ):
    queries = load_queries(query_path)
    print(f"[INFO] Loaded {len(queries)} queries entries.")

    bm25_searcher = LuceneSearcher.from_prebuilt_index("trec-covid-r1-full-text")
    rm3_searcher = LuceneSearcher.from_prebuilt_index("trec-covid-r1-full-text")
    rm3_searcher.set_rm3()

    print("[INFO] Loading SentenceTransformer model for reformulation...")
    start_model_load = time.time()
    dense_model = SentenceTransformer(DEFAULT_MODEL)
    end_model_load = time.time()
    print(f"[INFO] Model loaded in {end_model_load - start_model_load:.2f} seconds.\n")

    print("\n[INFO] Running BM25...")
    start_bm25 = time.time()
    with open(bm25_output_file, "w", encoding="utf-8") as out:
        for q in queries:
            qid, text = q["query_id"], q["query_text"]
            hits = bm25_searcher.search(text, k=k)
            for rank, hit in enumerate(hits, start=1):
                out.write(f"{qid}\t{hit.docid}\t{rank}\t{hit.score:.4f}\n")
    end_bm25 = time.time()
    print(f"[INFO] BM25 done in {end_bm25 - start_bm25:.2f} seconds.\n")


    print("[INFO] Running RM3...")
    start_rm3 = time.time()
    with open(rm3_output_file, "w", encoding="utf-8") as out:
        for q in queries:
            qid, text = q["query_id"], q["query_text"]
            hits = rm3_searcher.search(text, k=k)
            for rank, hit in enumerate(hits, start=1):
                out.write(f"{qid}\t{hit.docid}\t{rank}\t{hit.score:.4f}\n")
    end_rm3 = time.time()
    print(f"[INFO] RM3 done in {end_rm3 - start_rm3:.2f} seconds.\n")

    print("[INFO] Running Dense PRF (dynamic reformulation)...")
    start_dense = time.time()
    with open(qrf_output_file, "w", encoding="utf-8") as out:
        for q in queries:
            qid, text = q["query_id"], q["query_text"]

            # Step 1: initial BM25
            initial_hits = bm25_searcher.search(text, k = DEFAULT_INITIAL_K)
            docs = [getattr(hit, "raw", "") or getattr(hit, "contents", "") for hit in initial_hits]

            # Step 2: extract candidate terms
            candidates = extract_candidate_terms(docs, top_n=20)

            # Step 3: semantic selection
            expansion_topk = DEFAULT_EXPANSION_TOPK
            reform = dense_reformulate_query(text, candidates, dense_model, topk = expansion_topk)

            # Step 4: rerun BM25
            hits = bm25_searcher.search(reform, k=k)
            for rank, hit in enumerate(hits, start=1):
                out.write(f"{qid}\t{hit.docid}\t{rank}\t{hit.score:.4f}\n")
    end_dense = time.time()
    print(f"[INFO] Dense PRF done in {end_dense - start_dense:.2f} seconds.\n")

    print("\n===== Runtime Summary =====")
    print(f"Model load time:   {end_model_load - start_model_load:.2f} seconds")
    print(f"BM25 runtime:      {end_bm25 - start_bm25:.2f} seconds")
    print(f"RM3 runtime:       {end_rm3 - start_rm3:.2f} seconds")
    print(f"Dense PRF runtime: {end_dense - start_dense:.2f} seconds")
    print("============================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("t3_query_file_path")
    parser.add_argument("task3_bm25_output_file")
    parser.add_argument("task3_rm3_output_file")
    parser.add_argument("task3_qrf_output_file")
    parser.add_argument("k", type=int)
    args = parser.parse_args()
    task3_improve(
        query_path=args.t3_query_file_path,
        bm25_output_file=args.task3_bm25_output_file,
        rm3_output_file=args.task3_rm3_output_file,
        qrf_output_file=args.task3_qrf_output_file,
        k=args.k,
    )