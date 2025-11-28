import json
import torch
from pyserini.search.faiss import FaissSearcher # type: ignore
from pyserini.encode import QueryEncoder # type: ignore
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
import sys



###################### File Readers and writers #####################

def load_queries(query_path: str) -> Dict[str, str]:
    """Load queries from JSON file (MS MARCO format)."""
    queries = {}
    with open(query_path, 'r', encoding='utf-16') as f:
        for line in f:
            if line.strip():
                query_data = json.loads(line.strip())
                query_id = query_data['query_id']
                # Use title as the main query text
                query_text = query_data['text']
                queries[query_id] = query_text
    return queries

def write_results(results: List[Tuple[str, float]], output_file: str, qid: str, k: int):
    """Write baseline results to file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for rank, (docid, score) in enumerate(results[:k], 1):
            f.write(f"{qid} {docid} {rank} {score:.4f}\n")

###################### Core Retrieval Task #####################

def task1_distilbert(index_path: str, query_path: str, distilbert_output_file: str,k: int) :
    """
    Performs dense retrieval using a pre-built Pyserini index and a matching query encoder.
    """
    print("Starting dense retrieval task...")

    # Load queries from the file
    queries = load_queries(query_path)
    query_encoder_name = "castorini/tct_colbert-msmarco"

    # Initialize the dense searcher with the index path and the loaded encoder
    searcher = FaissSearcher(
        index_path,
        query_encoder=query_encoder_name
    )
    print("FaissSearcher initialized successfully.")

    # Iterate through queries, search, and store results
    for qid, qtext in tqdm(queries.items(), desc = "processing queries"):
        
        hits = searcher.search(qtext, k=k)
        formatted_results = [(hit.docid, hit.score) for hit in hits]
        write_results(formatted_results, distilbert_output_file,qid, k=k)
    print(f"Retrieval complete for {len(queries)} queries.")
    print(f"results are write in {distilbert_output_file} file")
    

###################### Main Execution #####################

if __name__ == "__main__":
    
    # Path to the directory containing the pre-downloaded FAISS index and other files
    index_dir = "/shared/indexes/msmarco/faiss-flat.msmarco-v1-passage.distilbert-dot-margin_mse-t2/faiss-flat.msmarco-v1-passage.distilbert-dot-margin_mse-t2.20210316.d44c3a"
    
    index_path = sys.argv[1]
    query_path = sys.argv[2]
    distilbert_output_file = sys.argv[3]
    k = int(sys.argv[4])
    with open(distilbert_output_file, 'w', encoding='utf-8') as f:
        pass
    task1_distilbert(index_path, query_path, distilbert_output_file, k)
