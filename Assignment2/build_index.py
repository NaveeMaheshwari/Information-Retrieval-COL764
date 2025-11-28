import os
import json
import time
import sys
import spacy
import math
from collections import defaultdict
from tqdm import tqdm


# load vocab
def load_vocab(vocab_path):
    vocab = set()
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                vocab.add(tok)
    return vocab

# load data
def load_data(corpus_dir: str):           
    docs = []
    for filename in os.listdir(corpus_dir):
        filepath = os.path.join(corpus_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # skip empty lines
                        try:
                            docs.append(json.loads(line))
                            if len(docs) >= 1000:
                                return docs
                        except json.JSONDecodeError as e:
                            print(f"Skipping bad line in {filename}: {e}")
    return docs

# build inverted index (Task 1 + extra for VSM + BM25)

def build_index(corpus_dir, vocab_path):
    vocab = load_vocab(vocab_path)
    print(f"Loaded vocab {len(vocab)} terms from {vocab_path}")

    nlp = spacy.blank("en")   # tokenizer only (same as Task-0)

    inverted_index = defaultdict(
        lambda: {"df": 0, "postings": defaultdict(lambda: {"tf": 0, "pos": []})}
    )

    doc_lengths = {}   # doc_id -> total token count (for BM25 etc.)
    doc_count = 0

    # --- PASS 1: Build inverted index ---
    for doc in load_data(corpus_dir):
        doc_id = doc.get("doc_id")
        if not doc_id:
            print("Warning: skipping document without 'doc_id' field")
            continue

        doc_count += 1
        seen_terms = set()
        pos_counter = 0

        for key, value in doc.items():
            if key == "doc_id" or not value:
                continue
            tokens = nlp(value)
            for tok in tokens:
                term = tok.text.strip()
                if term and term in vocab:
                    posting = inverted_index[term]["postings"][doc_id]
                    posting["tf"] += 1
                    posting["pos"].append(pos_counter)

                    if term not in seen_terms:
                        inverted_index[term]["df"] += 1
                        seen_terms.add(term)
                pos_counter += 1

        doc_lengths[doc_id] = pos_counter

    print(f"Processed {doc_count} documents for inverted index.")

    # --- PASS 2: Compute vector lengths (efficiently) ---
    doc_vector_lengths = defaultdict(float)
    
    for term, entry in inverted_index.items():
        
        df = entry["df"]
        # smoothed idf (never 0 or negative)
        idf = math.log10((doc_count + 1) / (df + 1)) + 1
        for doc_id, posting in entry["postings"].items():
            tf = posting["tf"]
            tf_weight = (1 + math.log10(tf)) if tf > 0 else 0
            weight = tf_weight * idf
            doc_vector_lengths[doc_id] += weight ** 2

    # finalize sqrt
    for doc_id in doc_vector_lengths:
        doc_vector_lengths[doc_id] = math.sqrt(doc_vector_lengths[doc_id])

    # --- SORTED INDEX for saving ---
    final_index = {}
    for term in sorted(inverted_index.keys()):
        postings = inverted_index[term]["postings"]
        sorted_postings = {}
        # if doc_ids are numeric strings, convert to int for sorting
        for docid in sorted(postings.keys()):
            sorted_postings[docid] = {
                "tf": postings[docid]["tf"],
                "pos": sorted(postings[docid]["pos"])
            }
        final_index[term] = {"df": inverted_index[term]["df"], "postings": sorted_postings}

    return final_index, doc_lengths, doc_count, dict(doc_vector_lengths)

def save_index(final_index, index_dir, doc_lengths, doc_count, doc_vector_lengths):
    os.makedirs(index_dir, exist_ok=True)

    # Save inverted index
    index_path = os.path.join(index_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=2)
    print(f"Inverted index saved to {index_path}")

    # ---- SAVE FOR VSM ----
    # Precompute VSM IDF per term (smoothed)
    # vsm_idf = {}
    # for term, entry in final_index.items():
    #     df = entry["df"]
    #     vsm_idf[term] = math.log10((doc_count + 1) / (df + 1)) + 1

    # vsm_stats = {
    #     "N": doc_count,
    #     "doc_vector_lengths": doc_vector_lengths,
    #     "idf": vsm_idf
    # }
    vsm_stats = {
        "N": doc_count,
        "doc_vector_lengths": doc_vector_lengths
    }
    
    vsm_path = os.path.join(index_dir, "vsm.json")
    with open(vsm_path, "w", encoding="utf-8") as f:
        json.dump(vsm_stats, f, indent=2)
    print(f"VSM stats saved to {vsm_path}")

    # ---- SAVE FOR BM25 ----
    avgdl = (sum(doc_lengths.values()) / doc_count) if doc_count else 0.0
    # Precompute BM25 IDF per term
    # bm25_idf = {}
    # for term, entry in final_index.items():
    #     df = entry["df"]
    #     bm25_idf[term] = math.log(1 + (doc_count - df + 0.5) / (df + 0.5)) if df >= 0 else 0.0

    bm25_stats = {
        "N": doc_count,
        "avgdl": avgdl,
        "doc_lengths": doc_lengths
    }
    bm25_path = os.path.join(index_dir, "bm25.json")
    with open(bm25_path, "w", encoding="utf-8") as f:
        json.dump(bm25_stats, f, indent=2)
    print(f"BM25 stats saved to {bm25_path}")



def load_index(index_dir):
    """
    Load inverted index, VSM stats, and BM25 stats from index_dir.
    Returns a dictionary with all components.
    """
    index_path = os.path.join(index_dir, "index.json")
    vsm_path = os.path.join(index_dir, "vsm.json")
    bm25_path = os.path.join(index_dir, "bm25.json")

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
    except FileNotFoundError:
        inverted_index = {}
        print(f"index.json not found in {index_dir}")

    try:
        with open(vsm_path, "r", encoding="utf-8") as f:
            vsm_stats = json.load(f)
    except FileNotFoundError:
        vsm_stats = {}
        print(f"vsm.json not found in {index_dir}")

    try:
        with open(bm25_path, "r", encoding="utf-8") as f:
            bm25_stats = json.load(f)
    except FileNotFoundError:
        bm25_stats = {}
        print(f"bm25.json not found in {index_dir}")

    return {
        "inverted_index": inverted_index,
        "vsm": vsm_stats,
        "bm25": bm25_stats
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python build_index.py <CORPUS_DIR> <VOCAB_PATH> <INDEX_DIR>")
        sys.exit(1)

    corpus_dir = sys.argv[1]
    vocab_path = sys.argv[2]
    index_dir = sys.argv[3]
    
    start_time = time.time()
    inverted_index, doc_lengths, doc_count, doc_vector_lengths = build_index(corpus_dir, vocab_path)
    save_index(inverted_index, index_dir, doc_lengths, doc_count, doc_vector_lengths)
    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))
    
# Example usage:
# python build_index.py corpus_dir VOCAB_DIR/vocab.txt INDEX_DIR