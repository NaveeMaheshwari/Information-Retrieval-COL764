
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher # type: ignore
import time

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


def initialize_msmarco_searcher():
    """Initialize MS MARCO passage searcher."""
    try:
        # Initialize BM25 searcher with MS MARCO prebuilt index
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')
        print("Successfully initialized MS MARCO passage searcher")
        return searcher
    except Exception as e:
        print(f"Error initializing MS MARCO searcher: {e}")
        return None

def search_msmarco_bm25(searcher, query: str, k: int) -> List[Tuple[str, float]]:
    hits = searcher.search(query, k=k)  # Get candidates for reranking
    results = []
    for hit in hits:
        docid = hit.docid
        score = hit.score
        results.append((docid, score))
    return results

def write_results(results: List[Tuple[str, float]], output_file: str, qid: str, k: int):
    """Write reranked results to file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for rank, (docid, score) in enumerate(results[:k], 1):
            f.write(f"{qid} {docid} {rank} {score:.4f}\n")
start = time.time()
# Initialize MS MARCO searcher
searcher = initialize_msmarco_searcher()

# Load queries
queries = load_queries("queries.json")

# Clear output files
bm25_output_file = "bm25_results.txt"
k =100
with open(bm25_output_file, 'w', encoding='utf-8') as f:
    pass  # Clear BM25 output file


# Process each query
for qid, query in tqdm(queries.items(), desc="Processing queries"):
    
    # Search with BM25 using MS MARCO
    bm25_results = search_msmarco_bm25(searcher, query, k)
    
    if not bm25_results:
        print(f"No BM25 results for query {qid}")
        continue
    
    # Write BM25 results
    write_results(bm25_results, bm25_output_file, qid, k)
    
print(f"BM25 results written to {bm25_output_file}")
end = time.time()
print(f"Total time taken: {end - start:.2f} seconds")

