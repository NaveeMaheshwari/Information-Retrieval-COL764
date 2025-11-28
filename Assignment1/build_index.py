import os
import json
import re
import time
import sys
from collections import defaultdict

# ------------------- Step 1: Load corpus and vocab -------------------
def load_data(corpus_dir: str, vocab_path: str):
    # sourcery skip: collection-builtin-to-comprehension, for-append-to-extend
    docs = []   # list of dictionaries
    with open(corpus_dir, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))   # total ~192,502 documents
            
    # docs = []
    # for fname in os.listdir(corpus_dir):
    #     if fname.endswith(".json"):
    #         with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
    #             data = json.load(f)
    #             if isinstance(data, dict):
    #                 docs.append(data)      # single doc
    #             elif isinstance(data, list):
    #                 docs.extend(data)      # list of docs

    # Load vocab into set for fast lookup
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = set(line.strip() for line in f)
    return docs, vocab
# ------------------- Step 2: Preprocess text -------------------
def preprocess(text, vocab):
    text = text.lower()
    text = re.sub(r"[0-9]", '', text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove non-ASCII chars
    tokens = text.split()
    # Keep only tokens that are in vocab
    tokens = [tok for tok in tokens if tok in vocab]
    return tokens

# ------------------- Step 3: Build inverted index -------------------
def build_index(collection_dir: str, vocab_path: str):
    inverted_index = defaultdict(lambda: defaultdict(list))
    corpus, vocab = load_data(collection_dir, vocab_path)

    # seen_docs = set()   # to handle duplicate doc_ids

    for item in corpus:
        doc_id = item["doc_id"]

        # if doc_id in seen_docs:
        #     continue  # skip duplicates
        # seen_docs.add(doc_id)

        doc_text = " ".join([str(v) for k, v in item.items() if k != "doc_id"])
        tokens = preprocess(doc_text, vocab)
        
        # for token in set(tokens):  # iterate unique tokens
        #     positions = [i for i, t in enumerate(tokens) if t == token]
        #     inverted_index[token][doc_id] = positions
        
        for pos, token in enumerate(tokens):
            inverted_index[token][doc_id].append(pos)

    return inverted_index

# ------------------- Step 4: Save index as JSON -------------------
def save_index(inverted_index: dict, index_dir: str):
    os.makedirs(index_dir,exist_ok=True) 
    sorted_index = {}
    for term in sorted(inverted_index.keys()):  # sort terms lexicographically
        sorted_index[term] = {}
        for doc_id in sorted(inverted_index[term].keys()):  # sort doc ids lexicographically
            positions = sorted(inverted_index[term][doc_id])  # ensure positions sorted
            sorted_index[term][doc_id] = positions
    index_path = os.path.join(index_dir,"index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(sorted_index, f, indent=2)

    # print(f"Inverted index saved to {index_dir}")
################################################################################################################################
# ------------------- Step 5: Compression helpers -------------------
def vb_encode_number(n: int) -> bytes:
    if n < 0:
        raise ValueError("VB requires non-negative integers")
    digits = []
    while True:
        digits.append(n % 128)
        n //= 128
        if n == 0:
            break
    digits = list(reversed(digits))       # big-endian
    digits[-1] += 128                     # mark last byte
    return bytes(digits)

def vb_encode_list(nums) -> bytes:
    out = bytearray()
    for x in nums:
        out.extend(vb_encode_number(x))
    return bytes(out)

def _collect_all_docs(index_dict) -> list:
    docs = set()
    for term, postings in index_dict.items():
        for doc_str in postings.keys():
            docs.add(doc_str)
    return sorted(docs)  # deterministic

def _build_docid_mapping(doc_strings: list) -> dict:
    return {doc_str: i + 1 for i, doc_str in enumerate(doc_strings)}

def _write_docmap(docmap_path: str, doc_strings: list, str2int: dict) -> None:
    with open(docmap_path, "w", encoding="utf-8") as f:
        for doc_str in doc_strings:
            f.write(f"{str2int[doc_str]}\t{doc_str}\n")

def _write_postings_and_lexicon(index_dict: dict, str2int: dict,
                                postings_path: str, lexicon_path: str) -> None:
    terms_sorted = sorted(index_dict.keys())
    with open(postings_path, "wb") as fpost, open(lexicon_path, "w", encoding="utf-8") as flex:
        for term in terms_sorted:
            postings = index_dict[term]  # {doc_str: [pos, ...]}
            entries = []
            for doc_str, pos_list in postings.items():
                doc_int = str2int[doc_str]
                entries.append((doc_int, sorted(pos_list)))
            entries.sort(key=lambda x: x[0])

            start_offset = fpost.tell()
            df = len(entries)

            fpost.write(vb_encode_number(df))

            prev_doc = 0
            for doc_int, pos_sorted in entries:
                doc_gap = doc_int - prev_doc
                prev_doc = doc_int

                tf = len(pos_sorted)
                pos_gaps = []
                prev_pos = 0
                for p in pos_sorted:
                    pos_gaps.append(p - prev_pos)
                    prev_pos = p

                fpost.write(vb_encode_number(doc_gap))
                fpost.write(vb_encode_number(tf))
                if tf:
                    fpost.write(vb_encode_list(pos_gaps))

            end_offset = fpost.tell()
            length = end_offset - start_offset

            flex.write(f"{term}\t{df}\t{start_offset}\t{length}\n")

def _write_meta(meta_path: str) -> None:
    meta = {
        "format": "vb-gap-v1",
        "postings_file": "postings.bin",
        "lexicon_file": "lexicon.tsv",
        "docmap_file": "docmap.tsv",
        "docids_start_from": 1,
        "layout": "VB(df); then for each doc: VB(doc_gap), VB(tf), VB(pos_gap)*tf",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def compress_index(index_dict, path_to_compressed_files_directory: str) -> None:
    # index_dict = _load_index_json(path_to_index_file)
    out_dir = path_to_compressed_files_directory
    # _ensure_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    postings_path = os.path.join(out_dir, "postings.bin")
    lexicon_path  = os.path.join(out_dir, "lexicon.tsv")
    docmap_path   = os.path.join(out_dir, "docmap.tsv")
    meta_path     = os.path.join(out_dir, "meta.json")

    all_docs_sorted = _collect_all_docs(index_dict)
    str2int = _build_docid_mapping(all_docs_sorted)
    _write_docmap(docmap_path, all_docs_sorted, str2int)
    _write_postings_and_lexicon(index_dict, str2int, postings_path, lexicon_path)
    _write_meta(meta_path)

# ------------------- Step 6: Main -------------------
def main():  # sourcery skip: use-fstring-for-formatting
    if len(sys.argv) != 5:
        print("Usage: python build_index.py <CORPUS_FILE> <VOCAB_FILE> <INDEX_FILE_JSON> <COMPRESSED_INDEX_DIR>")
        sys.exit(1)

    corpus_file = sys.argv[1]
    vocab_file = sys.argv[2]
    index_file_json = sys.argv[3]
    compressed_index_dir = sys.argv[4]

    # Validate required inputs
    for path, name in [
        (corpus_file, "<CORPUS_FILE>"),
        (vocab_file, "<VOCAB_FILE>"),
        (index_file_json, "<INDEX_FILE_JSON>"),
        (compressed_index_dir, "<COMPRESSED_INDEX_DIR>"),
    ]:
        if not path:
            # raise ValueError(f"{name} is required (must be provided via command line)") 
            raise ValueError("{} is required (must be provided via command line)".format(name))


    start_time = time.time()
    inverted_index = build_index(corpus_file, vocab_file)
    save_index(inverted_index, index_file_json)

    compress_index(inverted_index, compressed_index_dir)
    end_time = time.time()
    # print(f"Execution time for compress index: {end_time - start_time:.2f} seconds")
    print("Execution time: %.2f seconds" % (end_time - start_time))

if __name__ == "__main__":
    main()

## USAGE python build_index.py cord19-trec_covid-docs VOCAB_DIR/vocab.txt INDEX_DIR COMPRESSED_DIR


