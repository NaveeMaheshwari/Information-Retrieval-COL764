import os
import json
import re
import time
import sys

def load_data(corpus_dir: str, stopwords_file: str):
    docs = []   # will be a list of dictionaries
    with open(corpus_dir, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))   # total of ~192,502 documents
            
    # docs = []
    # for fname in os.listdir(corpus_dir):
    #     if fname.endswith(".json"):
    #         with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
    #             data = json.load(f)
    #             if isinstance(data, dict):
    #                 docs.append(data)      # single doc
    #             elif isinstance(data, list): 
    #                 docs.extend(data)      # list of docs

    # Load stopwords into a set
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)

    return docs, stopwords

def preprocess(text): 
    text = text.lower()
    text = re.sub(r"[0-9]", '', text)              # remove digits
    text = re.sub(r"[^\x00-\x7F]+", "", text)      # remove non-ASCII chars
    tokens = text.split()
    # Remove stopwords
    # tokens = [tok for tok in tokens if tok not in stopwords]
    return tokens


def build_vocab(corpus_dir: str, stopwords_file: str, vocab_dir: str) -> None:
    os.makedirs(vocab_dir, exist_ok=True)

    corpus, stopwords = load_data(corpus_dir, stopwords_file)
    vocab = set()
    for item in corpus:
        for key, value in item.items():
            if key == "doc_id":
                continue
            processed_tokens = preprocess(value)
            for tok in processed_tokens:
                if tok not in stopwords:
                    vocab.add(tok)
                    # print(tok)
            # vocab.update(preprocess(value, stopwords))
    # print(vocab)
    # Write vocab to file
    with open(os.path.join(vocab_dir, "vocab.txt"), 'w', encoding='utf-8') as f:
        for token in sorted(vocab):
            f.write(token + "\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python tokenize_corpus.py <CORPUS_FILE> <STOPWORDS_FILE> <VOCAB_DIR>")
        sys.exit(1)

    corpus_file = sys.argv[1]
    stopwords_file = sys.argv[2]
    vocab_dir = sys.argv[3]

    # Validate required inputs
    for path, name in [
        (corpus_file, "<CORPUS_FILE>"),
        (stopwords_file, "<STOPWORDS_FILE>"),
        (vocab_dir, "<VOCAB_DIR>"),
    ]:
        if not path:
            # raise ValueError(f"{name} is required (must be provided via command line)")
            raise ValueError("{} is required (must be provided via command line)".format(name))

    start_time = time.time()
    build_vocab(corpus_file, stopwords_file, vocab_dir)
    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))



if __name__ == "__main__":
    main()
    
## USAGE:  python .\tokenize_corpus.py .\cord19-trec_covid-docs .\stopwords.txt .\VOCAB_DIR\