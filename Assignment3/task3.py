#!/usr/bin/env python3
"""
Task 3: Improve Retrieval Performance
This module implements advanced techniques to improve retrieval performance.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import math
import random
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


def initialize_baseline_reranker():
    """Initialize  model for reranking."""
    try:
        # Use a pre-trained model for reranking
        model_name = 'cross-encoder/ms-marco-electra-base'
        model = CrossEncoder(model_name)  
        print(f"Successfully initialized reranker: {model_name}")
        return model
    except Exception as e:
        print(f"Error initializing reranker: {e}")
        return None



def apply_pseudo_relevance_feedback(query: str, top_docs: List[Tuple[str, float]],searcher, top_k_docs: int = 5, top_k_terms: int = 5) -> List[str]:
    """
    Apply pseudo-relevance feedback using top documents.
    """
    expanded_terms = re.findall(r'\b\w+\b', query.lower())
    
    # Simulate extracting terms from top documents
    # In real scenario, you'd extract actual terms from document content
    feedback_terms = []
    for docid, score in top_docs[:top_k_docs]:
        try:
            # Get actual document content
            doc = searcher.doc(docid)
            if doc:
                doc_text = doc.raw()  # full text of the document
                # Tokenize document text into words
                doc_words = re.findall(r'\b\w+\b', doc_text.lower())
                feedback_terms.extend(doc_words)
        except Exception as e:
            print(f"Error retrieving document {docid}: {e}")
            continue
    
    # Select most frequent terms from feedback
    term_counts = Counter(feedback_terms)
    top_feedback_terms = [term for term, count in term_counts.most_common(top_k_terms)]
    
    # Combine with original query terms
    expanded_terms.extend(top_feedback_terms)
    
    return expanded_terms

def baseline_rerank(query: str, bm25_results: List[Tuple[str, float]], reranker, searcher, k: int) -> List[Tuple[str, float]]:
    """
    Baseline 1:
    """
    try:
        # Prepare query-document pairs for BERT
        query_doc_pairs = []
        docids = []
        
        # ---Apply pseudo-relevance feedback to expand query ---
        expanded_terms = apply_pseudo_relevance_feedback(query, bm25_results[:k], searcher, top_k_docs=5, top_k_terms=5) 
        expanded_query = " ".join(expanded_terms)
        
        for docid, bm25_score in bm25_results:
            
            doc_content = get_document_content(searcher, docid)
            if doc_content:
                # Create query-document pair
                query_doc_pairs.append([expanded_query, doc_content])
                docids.append(docid)
        
        if not query_doc_pairs:
            print("No documents found for reranking")
            return bm25_results
        
        
        
        # Get reranker scores
        reranker_scores = reranker.predict(query_doc_pairs)
        
        reranked_results = []
        for i, (docid, bm25_score) in enumerate(bm25_results):
            if i < len(reranker_scores):
                reranker_score = reranker_scores[i]
                # Combine scores (you can adjust weights)
                combined_score = bm25_score * 0.0 + reranker_score * 1.0
                reranked_results.append((docid, combined_score))
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results
        
    except Exception as e:
        print(f"Error in BERT reranking: {e}")
        return bm25_results


def write_results(results: List[Tuple[str, float]], output_file: str, qid: str, k: int):
    """Write improved results to file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for rank, (docid, score) in enumerate(results[:k], 1):
            f.write(f"{qid} {docid} {rank} {score:.6f}\n")


def task3_improve(query_path: str, best_output_file: str, k: int):
    """
    Task 3: Improve retrieval performance using advanced techniques.
    
    Args:
        query_path: Path to queries JSON file
        best_output_file: Path to write improved results
        k: Number of documents to retrieve
    """
    print(f"Task 3: Improving retrieval performance with k={k}")
    
    # Initialize MS MARCO searcher
    searcher = initialize_msmarco_searcher()
    
    # Initialize rerankers for baseline
    baseline_reranker = initialize_baseline_reranker()
    
    # Load queries
    queries = load_queries(query_path)
    
    # Clear output file
    Path(best_output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(best_output_file, 'w', encoding='utf-8') as f:
        pass  # Clear file
    
    # Process each query
    for qid, query in tqdm(queries.items(), desc="Processing queries"):
        
        # Search with BM25 using MS MARCO
        bm25_results = search_msmarco_bm25(searcher, query, k)
        
        if not bm25_results:
            print(f"No BM25 results for query {qid}")
            continue
        
        # Apply ensemble improvement method
        improved_results =  baseline_rerank(query, bm25_results, baseline_reranker, searcher, k)
        
        # Write results
        write_results(improved_results, best_output_file, qid, k)
    
    print(f"Task 3 completed. Improved results written to {best_output_file}")


if __name__ == "__main__":
    # Test the function
    import sys
    if len(sys.argv) >= 4:
        query_path = sys.argv[1]
        best_output_file = sys.argv[2]
        k = int(sys.argv[3])
        task3_improve(query_path, best_output_file, k)
    else:
        print("Usage: python task3.py <query_path> <best_output_file> <k>")
