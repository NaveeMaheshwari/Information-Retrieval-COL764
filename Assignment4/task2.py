import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import os
import sys
from pyserini.search.lucene import LuceneImpactSearcher # type: ignore
import json
from typing import Dict, List, Tuple
import time

def load_queries(query_path: str) -> List[Dict[str, str]]:
    queries = []
    with open(query_path, 'r', encoding='utf-16') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                queries.append(json.loads(line))
    return queries

def write_results(results: List[Tuple[str, float]], output_file: str, qid: str, k: int):
    """Write baseline results to file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for rank, (docid, score) in enumerate(results[:k], 1):
            f.write(f"{qid} {docid} {rank} {score:.4f}\n")

def task2_splade( index_path: str, query_path: str, splade_output_file: str, k: int ) -> None:
    
    queries = load_queries(query_path)
    
    # prebuilt_index_name = 'msmarco-v1-passage.splade-pp-ed'
    prebuilt_index_name = index_path
    encoder_model_name = 'naver/splade-cocondenser-ensembledistil'
    searcher = LuceneImpactSearcher.from_prebuilt_index(
        prebuilt_index_name,
        query_encoder= encoder_model_name
    )
    print("Searcher loaded and configured successfully.")
    
    for item in queries:
        qid = item.get("query_id")
        query_text = item.get("text")
        # print(f"\n🔹 Query ID: {qid}")
        # print(f"🔹 Query Text: {query_text}")

        hits = searcher.search(query_text, k=k)
        formatted_results = [(hit.docid, hit.score) for hit in hits]
        write_results(formatted_results, splade_output_file,qid, k=k)
    print(f"SPLADE retrieval completed and results written in {splade_output_file}.")

def main():
    index_path = sys.argv[1]  # Path to the prebuilt SPLADE index
    query_path = sys.argv[2]  # Path to the queries file
    splade_output_file = sys.argv[3]  # Output file for SPLADE results
    k = int(sys.argv[4])  # Number of top documents to retrieve
    
    with open(splade_output_file, 'w') as f:
        pass 
    task2_splade(index_path, query_path, splade_output_file, k)
    

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")
# USAGE: python task2.py msmarco-v1-passage.splade-pp-ed queries.json splade_out.txt 100
