import os
import sys
import json
import math
import spacy
from collections import defaultdict
import time


def load_index(index_dir):
    index_path = os.path.join(index_dir, "index.json")
    bm25_path = os.path.join(index_dir, "bm25.json")

    with open(index_path, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)
    with open(bm25_path, "r", encoding="utf-8") as f:
        bm25_stats = json.load(f)

    return {"inverted_index": inverted_index, "bm25_stats": bm25_stats}


def bm25_query(query: str, index: object, k: int) -> list:
    inverted_index = index["inverted_index"]
    bm25_stats = index["bm25_stats"]

    N = bm25_stats["N"]
    avgdl = bm25_stats["avgdl"]
    doc_lengths = bm25_stats["doc_lengths"]

    # Hardcoded hyperparameters (tuned)
    k1 = 1.2
    b = 0.75

    nlp = spacy.blank("en")
    terms = [tok.text.strip() for tok in nlp(query) if tok.text.strip()]
    
    scores = defaultdict(float)

    for term in terms:
        if term not in inverted_index:
            continue

        df = inverted_index[term]["df"]
        if df == 0:
            continue

        # BM25 IDF formula
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))

        postings = inverted_index[term]["postings"]

        for doc_id, pdata in postings.items():
            tf = pdata["tf"]
            dl = doc_lengths.get(doc_id, 0)
            denom = tf + k1 * (1 - b + b * (dl / avgdl))
            term_score = idf * ((tf * (k1 + 1)) / denom)
            scores[doc_id] += term_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


# --------------------------
# BM25 batch processing
# --------------------------
def bm25(queryFile: str, index_dir: str, k: int, outFile: str) -> None:
    index = load_index(index_dir)
    results = {}

    # with open(queryFile, "r", encoding="utf-16") as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         try:
    #             obj = json.loads(line)
    #             qid = str(obj.get("query_id"))
    #             qtext = obj.get("title", "")

    #             docs = bm25_query(qtext, index, k)
    #             results[qid] = docs

    #         except json.JSONDecodeError as e:
    #             print(f"[WARN] Skipping bad query line: {e}")
    
    lines = []
    try:
        try:
            with open(queryFile, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeError:
            with open(queryFile, "r", encoding="utf-16") as f:
                lines = f.readlines()
    except FileNotFoundError as e:
        print(f"ERROR: Failed to load queries. {e}")
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            qid = str(obj.get("query_id"))
            qtext = obj.get("title", "")

            docs = bm25_query(qtext, index, k)
            results[qid] = docs

        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping bad query line: {e}")

    # Write results file
    with open(outFile, "w", encoding="utf-8") as out:
        for qid, docs in results.items():
            rank = 1
            for doc_id, score in docs:
                out.write(f"{qid}\t{doc_id}\t{rank}\t{score:.4f}\n")
                rank += 1

    print(f"BM25 results saved to {outFile}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python bm25.py <INDEX_DIR> <QUERY_FILE> <OUTPUT_FILE> <k>")
        sys.exit(1)

    index_dir = sys.argv[1]
    query_file = sys.argv[2]
    output_dir = sys.argv[3]
    k = int(sys.argv[4])
    
    out_File = os.path.join(output_dir, "bm25_retrieval_docids.txt")
    start = time.time()
    bm25(query_file, index_dir, k, out_File)
    end = time.time()
    print("Execution time: {:.2f} seconds".format(end - start))
#USAGE: python bm25_retrieval.py INDEX_DIR data\queries.json OUTPUT_DIR 5