#!/usr/bin/env python3
"""
Task 2: Implement Two Baselines
This module implements two different baseline retrieval methods:
"""

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
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
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

def initialize_baseline1_reranker():
    """Initialize model for reranking."""
    try:
        # Use a pre-trained model for reranking
        model_name = 'cross-encoder/ms-marco-electra-base'
        model = CrossEncoder(model_name)  
        print(f"Successfully initialized reranker: {model_name}")
        return model
    except Exception as e:
        print(f"Error initializing reranker: {e}")
        return None


def baseline1_rerank(query: str, bm25_results: List[Tuple[str, float]], reranker, searcher, k: int) -> List[Tuple[str, float]]:
    """
    Baseline 1: Random reranking with BM25 influence
    This baseline introduces controlled randomness to BM25 scores while maintaining some 
    correlation with the original ranking. This simulates a simple reranking approach.
    """
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
            return bm25_results
        
        # Get reranker scores
        reranker_scores = reranker.predict(query_doc_pairs)
        
        # reranker_scores = reranker.predict( 
        #     query_doc_pairs, 
        #     batch_size=32
        # )
        
        # Combine BM25 and BERT scores
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
    

#################### For baseline 2 ####################

def initialize_monot5_reranker():
    # model_name='castorini/monot5-base-msmarco'
    model_name='castorini/monot5-small-msmarco-10k'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    # # Get token ID for "true" (relevance)
    # true_token_id = tokenizer.convert_tokens_to_ids("▁true")  # T5 uses SentencePiece
    return tokenizer, model



def baseline2_rerank(query: str, bm25_results: List[Tuple[str, float]], tokenizer, model, searcher, k: int) -> List[Tuple[str, float]]:
    """
    Baseline 2: Rerank BM25 results using MonoT5
    """
    query_terms = re.findall(r'\b\w+\b', query.lower())
    query_length = len(query_terms)
    
    reranked_results = []
    for docid, bm25_score in bm25_results:
        doc_content = get_document_content(searcher, docid)
        if not doc_content:
            continue
        
        input_text = f"Query: {query} Document: {doc_content}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        decoder_input_ids = tokenizer.encode("true", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, -1]
            true_id = tokenizer.encode("true", add_special_tokens=False)[0]
            false_id = tokenizer.encode("false", add_special_tokens=False)[0]
            score = (logits[true_id] - logits[false_id]).item()
            reranked_results.append((docid, score))
    
    # Sort by combined score
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return reranked_results


def write_results(results: List[Tuple[str, float]], output_file: str, qid: str, k: int):
    """Write baseline results to file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        for rank, (docid, score) in enumerate(results[:k], 1):
            f.write(f"{qid} {docid} {rank} {score:.6f}\n")


def task2_baselines(query_path: str, bm25_output_file: str, baseline1_output_file: str, baseline2_output_file: str, k: int):
    """
    Task 2: Implement two baseline retrieval methods.
    
    Args:
        query_path: Path to queries JSON file
        bm25_output_file: Path to BM25 results file
        baseline1_output_file: Path to write baseline1 results
        baseline2_output_file: Path to write baseline2 results
        k: Number of documents to retrieve
    """
    print(f"Task 2: Implementing baselines with k={k}")
    
    # Initialize MS MARCO searcher
    searcher = initialize_msmarco_searcher()
    
    # Initialize rerankers for baseline1 and baseline2
    baseline1_reranker = initialize_baseline1_reranker()
    
    T5tokenizer, T5model = initialize_monot5_reranker()
    
    # Load queries and BM25 results
    queries = load_queries(query_path)
    # bm25_results = load_bm25_results(bm25_output_file)
    
    # Clear output files
    for output_file in [baseline1_output_file, baseline2_output_file, bm25_output_file]:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # Clear file
    
    # Process each query
    for qid, query in tqdm(queries.items()):
        
        # Search with BM25 using MS MARCO
        bm25_results = search_msmarco_bm25(searcher, query, k)
        
        if not bm25_results:
            print(f"No BM25 results for query {qid}")
            continue
        
        # Write BM25 results
        write_results(bm25_results, bm25_output_file, qid, k)
        
        # Baseline 1:
        baseline1_results = baseline1_rerank(query, bm25_results, baseline1_reranker, searcher, k)
        write_results(baseline1_results, baseline1_output_file, qid, k)
        
        # Baseline 2: 
        # baseline2_rerank(query: str, bm25_results: List[Tuple[str, float]], tokenizer, model, true_token_id, k: int) -> List[Tuple[str, float]]:
        baseline2_results = baseline2_rerank(query, bm25_results,T5tokenizer, T5model, searcher, k)
        write_results(baseline2_results, baseline2_output_file, qid, k)
    
    print(f"Task 2 completed.")
    print(f"BM25 results written to {bm25_output_file}")
    print(f"Baseline 1 results written to {baseline1_output_file}")
    print(f"Baseline 2 results written to {baseline2_output_file}")
    

if __name__ == "__main__":
    # Test the function
    import sys
    if len(sys.argv) >= 6:
        query_path = sys.argv[1]
        bm25_output_file = sys.argv[2]
        baseline1_output_file = sys.argv[3]
        baseline2_output_file = sys.argv[4]
        k = int(sys.argv[5])
        task2_baselines(query_path, bm25_output_file, baseline1_output_file, baseline2_output_file, k)
    else:
        print("Usage: python task2.py <query_path> <bm25_output_file> <baseline1_output_file> <baseline2_output_file> <k>")
# USAGES: python .\team21\task2.py .\ir_data\queries.json .\results\bm25_task2.txt .\results\baseline1_task2.txt .\results\baseline2_task2.txt 50