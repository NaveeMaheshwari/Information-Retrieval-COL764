import os
import sys
import json
import math
import spacy
from collections import defaultdict
import time


# --------------------------
# Load index (index.json + vsm.json)
# --------------------------
def load_index(index_dir):
    index_path = os.path.join(index_dir, "index.json")
    vsm_path = os.path.join(index_dir, "vsm.json")

    with open(index_path, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)
    with open(vsm_path, "r", encoding="utf-8") as f:
        vsm_stats = json.load(f)

    return {"inverted_index": inverted_index, "vsm_stats": vsm_stats}

# for single query
def vsm_query(query: str, index: object, k: int) -> list:
    inverted_index = index["inverted_index"]
    vsm_stats = index["vsm_stats"]

    nlp = spacy.blank("en")
    tokens = [t.text.strip() for t in nlp(query)]
    terms = [t for t in tokens if t]

    N = vsm_stats["N"]
    query_tf = {}
    for t in terms:
        query_tf[t] = query_tf.get(t, 0) + 1  # term frequency in query

    # Step 2: query vector
    query_weights = {}
    sum_sq = 0.0
    for term, tf in query_tf.items():
        if term in inverted_index:
            df = inverted_index[term]["df"]
            if df > 0:
                tf_weight = (1 + math.log10(tf))
                # idf = math.log10(N / df)
                idf = math.log10((N + 1) / (df + 1)) + 1
                w = tf_weight * idf
                query_weights[term] = w
                sum_sq += w ** 2
    query_length = math.sqrt(sum_sq)

    if query_length == 0:
        return []

    # Step 3: accumulator
    scores = {}
    for term, w_tq in query_weights.items():
        postings = inverted_index[term]["postings"]
        df = inverted_index[term]["df"]
        # idf = math.log10(N / df)
        idf = math.log10((N + 1) / (df + 1)) + 1
        for doc_id, pdata in postings.items():
            tf_d = pdata["tf"]
            w_td = (1 + math.log10(tf_d)) * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + (w_tq * w_td)

    # Step 4: cosine normalization
    doc_lengths = vsm_stats["doc_vector_lengths"]
    for doc_id in list(scores.keys()):
        doc_len = doc_lengths.get(doc_id, 0.0)
        if doc_len > 0:
            scores[doc_id] = scores[doc_id] / (query_length * doc_len)
        else:
            scores[doc_id] = 0.0

    # Step 5: rank
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True) # list of (doc_id, score)
    return ranked[:k]

def vsm(queryFile: str, index_dir: str, k: int, outFile: str) -> None:
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
    #             qtext = obj.get("title", "")  #  using title as query

    #             docs = vsm_query(qtext, index, k)
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

    # Process each query
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            qid = str(obj.get("query_id"))
            qtext = obj.get("title", "")  # using title as query

            docs = vsm_query(qtext, index, k)
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
    print(f"VSM results saved to {outFile}")



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python vsm.py <INDEX_DIR> <QUERY_FILE> <OUTPUT_DIR> <k>")
        sys.exit(1)
    index_dir = sys.argv[1]
    query_file = sys.argv[2]
    output_dir = sys.argv[3]
    k = int(sys.argv[4])
    
    out_File = os.path.join(output_dir, "vsm_docids.txt")
    start = time.time()
    vsm(query_file, index_dir, k, out_File)
    end = time.time()
    print("Execution time: {:.2f} seconds".format(end - start))
    
#USAGE: python vsm.py INDEX_DIR data\queries.json OUTPUT_DIR 5
