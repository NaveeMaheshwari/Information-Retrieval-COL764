import os
import sys
import json
import math
import time
from collections import defaultdict

import spacy
import vsm as vsm_module


# Module-level tokenizer for efficiency
_NLP = spacy.blank("en")


def _tokenize(text: str) -> list:
    return [t.text.strip() for t in _NLP(text)]


def _compute_query_weights(terms: list, inverted_index: dict, N: int) -> dict:
    query_tf = {}
    for t in terms:
        if not t:
            continue
        query_tf[t] = query_tf.get(t, 0) + 1

    weights = {}
    sum_sq = 0.0
    for term, tf in query_tf.items():
        if term in inverted_index:
            df = inverted_index[term]["df"]
            if df > 0:
                tf_weight = (1 + math.log10(tf))
                idf = math.log10((N + 1) / (df + 1)) + 1
                w = tf_weight * idf
                weights[term] = w
                sum_sq += w * w
    length = math.sqrt(sum_sq)
    return weights, length


def _build_doc_centroid(pseudo_docs: list, inverted_index: dict, N: int) -> dict:
    """
    Compute average tf-idf vector over pseudo relevant docs.
    pseudo_docs: list of doc_ids
    """
    centroid = defaultdict(float) 
    if not pseudo_docs:
        return {}

    for doc_id in pseudo_docs:
        # accumulate tf-idf weights for each term in doc
        for term, entry in inverted_index.items():
            postings = entry["postings"]
            if doc_id not in postings:
                continue
            tf = postings[doc_id]["tf"]
            if tf <= 0:
                continue
            df = entry["df"]
            idf = math.log10((N + 1) / (df + 1)) + 1
            w_td = (1 + math.log10(tf)) * idf
            centroid[term] += w_td

    inv_count = 1.0 / float(len(pseudo_docs))
    for term in list(centroid.keys()):
        centroid[term] *= inv_count
    return dict(centroid)


def _score_with_query_weights(query_weights: dict, inverted_index: dict, doc_vector_lengths: dict, N) -> dict:
    scores = {}
    # accumulate dot-products
    for term, w_tq in query_weights.items():
        if term not in inverted_index:
            continue
        entry = inverted_index[term]
        df = entry["df"]
        idf = math.log10((N + 1) / (df + 1)) + 1
        postings = entry["postings"]
        for doc_id, pdata in postings.items():
            tf_d = pdata["tf"]
            w_td = (1 + math.log10(tf_d)) * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + (w_tq * w_td)

    # cosine normalization
    sum_sq = sum(w * w for w in query_weights.values())
    qlen = math.sqrt(sum_sq) if sum_sq > 0 else 0.0
    if qlen == 0:
        return {}

    for doc_id in list(scores.keys()):
        dlen = doc_vector_lengths.get(doc_id, 0.0)
        if dlen > 0:
            scores[doc_id] = scores[doc_id] / (qlen * dlen)
        else:
            scores[doc_id] = 0.0
    return scores


def prf_query(query: str, index: object, k: int) -> list:
    """
    One round of pseudo-relevance feedback (Rocchio-style) for VSM.
    Returns top-k list of (doc_id, score).
    """
    inverted_index = index["inverted_index"]
    vsm_stats = index["vsm_stats"]
    N = vsm_stats["N"]
    doc_vector_lengths = vsm_stats["doc_vector_lengths"]

    # Initial retrieval
    initial_k = max(k, 20)  # retrieve more for feedback
    initial_ranked = vsm_module.vsm_query(query, index, initial_k)

    # Select pseudo-relevant docs (top R)
    R = min(10, len(initial_ranked))
    pseudo_docs = [doc_id for doc_id, _ in initial_ranked[:R]]

    # Original query weights
    terms = _tokenize(query)
    q_weights, _ = _compute_query_weights(terms, inverted_index, N)

    # Pseudo feedback centroid
    centroid = _build_doc_centroid(pseudo_docs, inverted_index, N)

    # Rocchio update (no negative component)
    alpha = 1.0
    beta = 0.75
    prf_weights = defaultdict(float)
    for term, w in q_weights.items():
        prf_weights[term] += alpha * w
    for term, w in centroid.items():
        prf_weights[term] += beta * w

    # Score with updated query weights
    scores = _score_with_query_weights(prf_weights, inverted_index, doc_vector_lengths,N)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def prf(queryFile: str, index_dir: str, k: int, outFile: str) -> None:
    index = vsm_module.load_index(index_dir)
    results = {}

    # with open(queryFile, "r", encoding="utf-16") as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         try:
    #             obj = json.loads(line)
    #         except json.JSONDecodeError:
    #             continue
    #         qid = str(obj.get("query_id"))
    #         qtext = obj.get("title", "")
    #         if not qid:
    #             continue

    #         docs = prf_query(qtext, index, k)
    #         results[qid] = docs
            
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
        except json.JSONDecodeError:
            continue

        qid = str(obj.get("query_id"))
        qtext = obj.get("title", "")
        if not qid:
            continue

        docs = prf_query(qtext, index, k)
        results[qid] = docs


    with open(outFile, "w", encoding="utf-8") as out:
        for qid, docs in results.items():
            rank = 1
            for doc_id, score in docs:
                out.write(f"{qid}\t{doc_id}\t{rank}\t{score:.4f}\n")
                rank += 1


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python feedback_retrieval.py <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k>")
        sys.exit(1)

    index_dir = sys.argv[1]
    query_file = sys.argv[2]
    output_dir = sys.argv[3]
    k = int(sys.argv[4])

    os.makedirs(output_dir, exist_ok=True)
    outFile = os.path.join(output_dir, "feedback_docids.txt")

    start = time.time()
    prf(query_file, index_dir, k, outFile)
    end = time.time()
    print("Execution time: {:.2f} seconds".format(end - start))


# USAGES: python feedback_retrieval.py INDEX_DIR data\queries.json OUTPUT_DIR 20
