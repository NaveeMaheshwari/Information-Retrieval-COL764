#!/usr/bin/env python3
"""
Task 1: Reranking BM25 Results using MS MARCO and BERT
This module implements reranking of BM25 results using MS MARCO prebuilt index and BERT reranking.
"""
import warnings
warnings.filterwarnings("ignore")


import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import re
from collections import Counter
from tqdm import tqdm

# Import required libraries for MS MARCO and BERT
try:
    from pyserini.search.lucene import LuceneSearcher
    import torch
    import numpy as np
    from sentence_transformers import CrossEncoder
except ImportError as e:
    print(f"Warning: Required libraries not found: {e}")
    # print("Please install: pip install pyserini transformers sentence-transformers torch")
    # print("For MS MARCO support, you may also need: pip install faiss-cpu")


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


def initialize_bert_reranker():
    """Initialize BERT model for reranking."""
    try:
        # Use a pre-trained BERT model for reranking
        # You can use different models like msmarco-distilbert-base-tas-b, msmarco-MiniLM-L-6-v3, etc.
        model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        model = CrossEncoder(model_name)
        # model_name = 'msmarco-distilbert-base-tas-b'
        # model = SentenceTransformer(model_name)
        print(f"Successfully initialized BERT reranker: {model_name}")
        return model
    except Exception as e:
        print(f"Error initializing BERT reranker: {e}")
        return None


def search_msmarco_bm25(searcher, query: str, k: int) -> List[Tuple[str, float]]:
    """Search using MS MARCO BM25 index."""
    try:
        # Search with BM25
        hits = searcher.search(query, k=k)  # Get candidates for reranking
        
        results = []
        for hit in hits:
            docid = hit.docid
            score = hit.score
            results.append((docid, score))
        return results
    except Exception as e:
        print(f"Error in BM25 search: {e}")
        return []


def get_document_content(searcher, docid: str) -> str:
    """Get document content by docid."""
    try:
        doc = searcher.doc(docid)
        if doc:
            return doc.raw()
        return ""
    except Exception as e:
        print(f"Error retrieving document {docid}: {e}")
        return ""


def bert_rerank(query: str, bm25_results: List[Tuple[str, float]], bert_model, searcher, k: int) -> List[Tuple[str, float]]:
    """Rerank BM25 results using BERT."""
    if bert_model is None:
        print("BERT model not available")
        return bm25_results[:k]
    try:
        # Prepare query-document pairs for BERT
        query_doc_pairs = []
        docids = []
        
        for docid, bm25_score in bm25_results: 
            doc_content = get_document_content(searcher, docid)
            if doc_content:
                # Create query-document pair
                query_doc_pairs.append([query, doc_content])
                docids.append(docid)
        
        if not query_doc_pairs:
            print("No documents found for reranking")
            return bm25_results[:k]
        
        # Get BERT scores
        bert_scores = bert_model.predict(query_doc_pairs)
        
        # Combine BM25 and BERT scores
        reranked_results = []
        for i, (docid, bm25_score) in enumerate(bm25_results):
            if i < len(bert_scores):
                bert_score = bert_scores[i]
                # Combine scores (you can adjust weights)
                combined_score = bm25_score * 0.0 + bert_score * 1.0
                reranked_results.append((docid, combined_score))
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:k]
        
    except Exception as e:
        print(f"Error in BERT reranking: {e}")
        return bm25_results[:k]

def write_results(results: List[Tuple[str, float]], output_file: str, qid: str, k: int):
    """Write reranked results to file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for rank, (docid, score) in enumerate(results[:k], 1):
            f.write(f"{qid} {docid} {rank} {score:.6f}\n")


def task1_rerank(query_path: str, bm25_output_file: str, reranked_output_file: str, k: int):
    """
    Task 1: Rerank BM25 results using MS MARCO prebuilt index and BERT reranking.
    
    Args:
        query_path: Path to queries JSON file
        bm25_output_file: Path to write BM25 results (will be generated from MS MARCO)
        reranked_output_file: Path to write reranked results
        k: Number of documents to retrieve and rerank
    """
    print(f"Task 1: Reranking BM25 results with MS MARCO and BERT (k={k})")
    
    # Initialize MS MARCO searcher
    searcher = initialize_msmarco_searcher()
    
    # Initialize BERT reranker
    bert_model = initialize_bert_reranker()
    
    # Load queries
    queries = load_queries(query_path)
    
    # Clear output files
    Path(bm25_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(reranked_output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(bm25_output_file, 'w', encoding='utf-8') as f:
        pass  # Clear BM25 output file
    with open(reranked_output_file, 'w', encoding='utf-8') as f:
        pass  # Clear reranked output file
    
    # Process each query
    for qid, query in tqdm(queries.items(), desc="Processing queries"):
        # print(f"Processing query {qid}: {query[:50]}...")
        
        # Search with BM25 using MS MARCO
        bm25_results = search_msmarco_bm25(searcher, query, k)
        
        if not bm25_results:
            print(f"No BM25 results for query {qid}")
            continue
        
        # Write BM25 results
        write_results(bm25_results, bm25_output_file, qid, k)
        
        # Rerank with BERT
        reranked_results = bert_rerank(query, bm25_results, bert_model, searcher, k)
        
        # Write reranked results
        write_results(reranked_results, reranked_output_file, qid, k)
    
    print(f"Task 1 completed.")
    print(f"BM25 results written to {bm25_output_file}")
    print(f"Reranked results written to {reranked_output_file}")


if __name__ == "__main__":
    # Test the function
    import sys
    if len(sys.argv) >= 5:
        query_path = sys.argv[1]
        bm25_output_file = sys.argv[2]
        reranked_output_file = sys.argv[3]
        k = int(sys.argv[4])
        task1_rerank(query_path, bm25_output_file, reranked_output_file, k)
    else:
        print("Usage: python task1.py <query_path> <bm25_output_file> <reranked_output_file> <k>")
# USAGES: python .\team21\task1.py .\ir_data\queries.json .\results\bm25_task1.txt .\results\reranked_task1.txt 50