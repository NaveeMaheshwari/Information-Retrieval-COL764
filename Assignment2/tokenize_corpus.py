import os
import json
import re
import time
import sys
import spacy # type: ignore

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
                            if len(docs) >=1000:
                                return docs
                        except json.JSONDecodeError as e:
                            print(f"Skipping bad line in {filename}: {e}")
    return docs


def build_vocab(corpus_dir: str, vocab_dir: str) -> None:
    os.makedirs(vocab_dir, exist_ok=True)
    nlp = spacy.blank("en") ## Initialize spaCy tokenizer
    corpus = load_data(corpus_dir)
    vocab = set()
    for item in corpus:
        for key, value in item.items():
            if key == "doc_id":
                continue
            # processed_tokens = nlp(value) # Tokenize using spaCy
            processed_tokens = nlp.make_doc(value)
            for tok in processed_tokens:
                tok_text = tok.text.strip()
                vocab.add(tok_text)
    print(f"Vocab size: {len(vocab)}")

    vocab_file = os.path.join(vocab_dir, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")
    print(f"Vocabulary of size {len(vocab)} saved at {vocab_file}")
    
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tokenize_corpus.py <CORPUS_DIR> <VOCAB_DIR>")
        sys.exit(1)

    corpus_dir = sys.argv[1]
    vocab_dir = sys.argv[2]

    start_time = time.time()
    build_vocab(corpus_dir,vocab_dir)
    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))

# Example usage:
# python tokenize_corpus.py corpus_dir VOCAB_DIR