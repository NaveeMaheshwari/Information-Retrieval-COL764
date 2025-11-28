import re
import json
import os 
import time
import sys

#### decompression code 
#-------------functions for reading from files -----------------------------
def read_docmap(docmap_path: str):
    int2str = {}
    with open(docmap_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            int_id_str, doc_str = line.rstrip("\n").split("\t", 1)
            int2str[int(int_id_str)] = doc_str
    return int2str

def read_lexicon(lexicon_path: str):
    entries = []
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            term, df_str, off_str, len_str = line.rstrip("\n").split("\t")
            entries.append((term, int(df_str), int(off_str), int(len_str)))
    return entries

#-------------functions for decoding from bytes ------------------------
def vb_decode_stream(b: bytes, start: int = 0):
    n = 0
    i = start
    while i < len(b):
        byte = b[i]
        if byte >= 128:
            n = 128 * n + (byte - 128)
            return n, i + 1
        else:
            n = 128 * n + byte
            i += 1
    raise ValueError("Truncated VB stream")

def vb_decode_list(b: bytes):
    numbers = []
    i = 0
    while i < len(b):
        n, i = vb_decode_stream(b, i)
        numbers.append(n)
    return numbers

def decode_term_postings(blob: bytes, int2str_doc: dict) -> dict:
    i = 0
    term_postings = {}
    df, i = vb_decode_stream(blob, i)
    prev_doc = 0
    for _ in range(df):
        doc_gap, i = vb_decode_stream(blob, i)
        doc_id = prev_doc + doc_gap
        prev_doc = doc_id

        tf, i = vb_decode_stream(blob, i)
        positions = []
        prev_pos = 0
        for _ in range(tf):
            pos_gap, i = vb_decode_stream(blob, i)
            pos = prev_pos + pos_gap
            prev_pos = pos
            positions.append(pos)

        term_postings[int2str_doc[doc_id]] = positions
    return term_postings

def decompress_index(compressed_index_dir: str):
    postings_path = os.path.join(compressed_index_dir, "postings.bin")
    lexicon_path  = os.path.join(compressed_index_dir, "lexicon.tsv")
    docmap_path   = os.path.join(compressed_index_dir, "docmap.tsv")

    int2str_doc = read_docmap(docmap_path)
    lexicon = read_lexicon(lexicon_path)

    with open(postings_path, "rb") as f:
        postings_bytes = f.read()

    index = {}
    for term, df, offset, length in lexicon:
        blob = postings_bytes[offset: offset + length]
        index[term] = decode_term_postings(blob, int2str_doc)
    return index

####################################################
##### query parsing 
def load_stopwords(stopword_file):
    with open(stopword_file, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)
    return stopwords

def load_queries(file_path: str):
    queries = []
    with open(file_path, "r", encoding="utf-16") as f:
        for line in f:
            if line.strip():
                query = json.loads(line)
                queries.append(query)
    return queries

def query_tokenizer(query_title, stopwords):
    q = query_title.lower()
    q = re.sub(r"\d+", " ", q)
    q = re.sub(r"[^\x00-\x7F]+", "", q)
    raw_tokens = q.split()
    tokens = []
    for tok in raw_tokens:
        while tok.startswith("("):
            tokens.append("(")
            tok = tok[1:]
        trailing = []
        while tok.endswith(")"):
            trailing.append(")")
            tok = tok[:-1]

        if tok:
            if tok in ("and", "or", "not"):
                tokens.append(tok.upper())
            elif tok not in stopwords:
                tokens.append(tok)

        tokens.extend(trailing)

    final_tokens = []
    for i, tok in enumerate(tokens):
        final_tokens.append(tok)
        if i == len(tokens) - 1:
            continue
        curr, nxt = tok, tokens[i + 1]
        left_ok = (curr not in ["AND", "OR", "NOT", "("])
        right_ok = (nxt not in ["AND", "OR", ")",])
        if left_ok and right_ok:
            final_tokens.append("AND")
    return final_tokens

def query_parser(query_vector):
    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    assoc = {"NOT": "right", "AND": "left", "OR": "left"}
    output = []
    stack = []
    for tok in query_vector:
        if tok not in ("AND", "OR", "NOT", "(", ")"):
            output.append(tok)
        elif tok in ("AND", "OR", "NOT"):
            while stack and stack[-1] != "(":
                top = stack[-1]
                if (assoc[tok] == "left" and precedence[top] >= precedence[tok]) or \
                   (assoc[tok] == "right" and precedence[top] > precedence[tok]):
                    output.append(stack.pop())
                else:
                    break
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.pop()
    while stack:
        output.append(stack.pop())
    return output

##########################################################
##### boolean retrieval
def eval_postfix(postfix, index, all_docs):
    stack = []
    for token in postfix:
        if token in {"AND", "OR", "NOT"}:
            if token == "NOT":
                s = stack.pop()
                stack.append(all_docs - s)
            else:
                s2 = stack.pop()
                s1 = stack.pop()
                if token == "AND":
                    stack.append(s1 & s2)
                elif token == "OR":
                    stack.append(s1 | s2)
        else:
            stack.append(set(index.get(token, {}).keys()))
    return stack.pop() if stack else set()

def boolean_retrieval(inverted_index, path_to_query_file, output_dir, stopwords):
    queries = load_queries(path_to_query_file)
    # queries = queries[:25]  # for testing only
    # print(queries)
    all_docs = set()
    for postings in inverted_index.values():
        all_docs |= set(postings.keys())

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "docid.txt")
    with open(out_path, "w") as f:
        for q in queries:
            title = q.get("title")
            qid = q.get("query_id")
            query_vector = query_tokenizer(title, stopwords)
            postfix = query_parser(query_vector)
            result_docs = eval_postfix(postfix, inverted_index, all_docs)
            for rank, docid in enumerate(sorted(result_docs), start=1):
                f.write(f"{qid} {docid} {rank} 1\n")

# ------------------- main function using sys.argv -------------------
def main():  # sourcery skip: use-fstring-for-formatting
    if len(sys.argv) != 5:
        print("Usage: python retrieval.py <COMPRESSED_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <STOPWORDS_FILE>")
        sys.exit(1)

    compressed_dir = sys.argv[1]
    query_files = sys.argv[2]
    output_dir = sys.argv[3]
    stopwords_file = sys.argv[4]

    # Validate inputs
    for path, name in [
        (compressed_dir, "<COMPRESSED_DIR>"),
        (query_files, "<QUERY_FILE_PATH>"),
        (output_dir, "<OUTPUT_DIR>"),
        (stopwords_file, "<STOPWORDS_FILE>"),
    ]:
        if not path:
            # raise ValueError(f"{name} is required (must be provided via command line)")
            raise ValueError("{} is required (must be provided via command line)".format(name))

    start_time = time.time()
    decompressed_index = decompress_index(compressed_dir)
    # print(decompressed_index)
    stopwords = load_stopwords(stopwords_file)
    boolean_retrieval(decompressed_index, query_files, output_dir, stopwords)
    end_time = time.time()
    print("time taken for retrieval.py file: {:.2f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    main()

# USAGE python retrieval.py COMPRESSED_DIR queries.json OUTPUT_DIR stopwords.txt