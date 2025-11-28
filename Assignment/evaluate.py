#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
import math
import csv
import sys
import os

# ----------------- dynamic import -----------------
def import_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

# ----------------- run/qrels parsing -----------------
def read_run_4col(run_path: Path) -> Dict[str, List[Tuple[int, str, float]]]:
    """
    Required run format (4 columns per line):
        qid docid rank score
    Returns: {qid: [(rank, docid, score), ...]} sorted by rank asc, docids deduped.
    """
    runs: Dict[str, List[Tuple[int, str, float]]] = {}
    with open(run_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 4:
                raise ValueError(f"{run_path.name}:{ln} must have 4 columns: qid docid rank score")
            qid, docid, rank_s, score_s = parts
            try:
                rank = int(rank_s)
                score = float(score_s)
            except ValueError:
                raise ValueError(f"{run_path.name}:{ln} rank must be int and score must be float")
            runs.setdefault(qid, []).append((rank, docid, score))
    # sort + dedup per qid
    for qid, items in runs.items():
        items.sort(key=lambda x: x[0])
        seen = set()
        dedup = []
        for r, d, s in items:
            if d in seen:
                continue
            seen.add(d)
            dedup.append((r, d, s))
        runs[qid] = dedup
    return runs

def read_qrels_3col(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """
    qrels format (3 columns per line):
        qid docid rel
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 3:
                raise ValueError(f"{qrels_path.name}:{ln} must have 3 columns: qid docid rel")
            qid, docid, rel_s = parts  # <-- per your spec
            try:
                rel = int(rel_s)
            except ValueError:
                raise ValueError(f"{qrels_path.name}:{ln} rel must be int")
            qrels.setdefault(qid, {})[docid] = rel
    return qrels

# ----------------- metrics (no external deps) -----------------
def dcg_at_k(rels: List[int], k: int) -> float:
    return sum(((2**rels[i] - 1) / math.log2(i + 2)) for i in range(min(k, len(rels))))

def ndcg_at_k(pred_docids: List[str], qrels_for_q: Dict[str, int], k: int) -> float:
    gains = [qrels_for_q.get(docid, 0) for docid in pred_docids]
    dcg = dcg_at_k(gains, k)
    ideal = sorted(qrels_for_q.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0.0 else dcg / idcg

def mrr_at_k(pred_docids: List[str], qrels_for_q: Dict[str, int], k: int) -> float:
    for i, docid in enumerate(pred_docids[:k], start=1):
        if qrels_for_q.get(docid, 0) > 0:
            return 1.0 / i
    return 0.0

def eval_runs(run: Dict[str, List[Tuple[int, str, float]]], qrels: Dict[str, Dict[str, int]]):
    ks = [1, 5, 10]
    agg = {f"ndcg@{k}": 0.0 for k in ks}
    agg["mrr@10"] = 0.0
    qids = sorted(qrels.keys())
    n = len(qids) or 1
    for qid in qids:
        pred = [docid for _, docid, _ in run.get(qid, [])]
        qrels_q = qrels[qid]
        for k in ks:
            agg[f"ndcg@{k}"] += ndcg_at_k(pred, qrels_q, k)
        agg["mrr@10"] += mrr_at_k(pred, qrels_q, 10)
    for k in list(agg.keys()):
        agg[k] /= n
    return agg

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate IR assignment outputs produced by task1/2/3.")
    # Unchanged essentials
    ap.add_argument("--team_dir", required=True, help="Path to extracted teamXX/ folder (contains task1/2/3.py)")
    ap.add_argument("--qrels_file_path", required=True, help="Path to qrels file in 3-column format: 'qid docid rel'")

    # Your modified arg surface:
    ap.add_argument("--query_file_path", required=True, help="Path to query file (students read this)")
    ap.add_argument("--task1_bm25_output_file", required=True, help="Path to write Task1 BM25 run (txt)")
    ap.add_argument("--reranked_output_file", required=True, help="Path to write Task1 reranked run (txt)")
    # ap.add_argument("--task1_plots_dir", required=True, help="Directory to save Task1 plots")

    ap.add_argument("--task2_bm25_output_file", required=True, help="Path to write Task2 BM25 baseline run (txt)")
    ap.add_argument("--baseline1_output_file", required=True, help="Path to write Task2 baseline1 run (txt)")
    ap.add_argument("--baseline2_output_file", required=True, help="Path to write Task2 baseline2 run (txt)")
    # ap.add_argument("--task2_plots_dir", required=True, help="Directory to save Task2 plots")

    ap.add_argument("--best_output_file", required=True, help="Path to write Task3 best/improved run (txt)")
    # ap.add_argument("--task3_plots_dir", required=True, help="Directory to save Task3 plots")

    ap.add_argument("--k", type=int, default=50, help="Retrieval depth K to pass to student code")

    args = ap.parse_args()

    team_dir = Path(args.team_dir).resolve()

    # Ensure plot dirs exist; ensure parent dirs for run outputs exist
    # for d in [args.task1_plots_dir, args.task2_plots_dir, args.task3_plots_dir]:
    #     ensure_dir(Path(d))
    for p in [
        args.task1_bm25_output_file,
        args.reranked_output_file,
        args.task2_bm25_output_file,
        args.baseline1_output_file,
        args.baseline2_output_file,
        args.best_output_file,
    ]:
        ensure_parent(p)

    # Import student modules (strict names)
    t1 = import_from_file("task1", team_dir / "task1.py")
    t2 = import_from_file("task2", team_dir / "task2.py")
    t3 = import_from_file("task3", team_dir / "task3.py")

    # Verify required functions exist (strict names)
    if not hasattr(t1, "task1_rerank"):
        raise AttributeError("task1.py must define task1_rerank(...)")
    if not hasattr(t2, "task2_baselines"):
        raise AttributeError("task2.py must define task2_baselines(...)")
    if not hasattr(t3, "task3_improve"):
        raise AttributeError("task3.py must define task3_improve(...)")

    # ----------------- Execute student code -----------------
    # Task 1
    t1.task1_rerank(
        query_path=args.query_file_path,
        bm25_output_file=args.task1_bm25_output_file,
        reranked_output_file=args.reranked_output_file,
        k=args.k,
    )

    # Task 2
    t2.task2_baselines(
        query_path=args.query_file_path,
        bm25_output_file=args.task2_bm25_output_file,
        baseline1_output_file=args.baseline1_output_file,
        baseline2_output_file=args.baseline2_output_file,
        k=args.k,
    )

    # Task 3
    t3.task3_improve(
        query_path=args.query_file_path,
        best_output_file=args.best_output_file,
        k=args.k,
    )

    # ----------------- Validate + Evaluate -----------------
    # Read qrels (3-column format)
    qrels = read_qrels_3col(Path(args.qrels_file_path))

    def check_and_eval(label: str, run_path_str: str):
        run_path = Path(run_path_str)
        if not run_path.exists():
            raise FileNotFoundError(f"Expected run file not found: {run_path}")
        run = read_run_4col(run_path)
        metrics = eval_runs(run, qrels)
        return label, metrics

    results = [
        check_and_eval("task1_bm25",   args.task1_bm25_output_file),
        check_and_eval("task1_rerank", args.reranked_output_file),
        check_and_eval("task2_bm25",   args.task2_bm25_output_file),
        check_and_eval("task2_base1",  args.baseline1_output_file),
        check_and_eval("task2_base2",  args.baseline2_output_file),
        check_and_eval("task3_best",   args.best_output_file),
    ]

    # Write metrics.csv beside best_output_file
    metrics_csv = Path(args.best_output_file).parent / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "ndcg@1", "ndcg@5", "ndcg@10", "mrr@10"])
        for name, m in results:
            w.writerow([name, f"{m['ndcg@1']:.5f}", f"{m['ndcg@5']:.5f}", f"{m['ndcg@10']:.5f}", f"{m['mrr@10']:.5f}"])

    # Pretty print
    print("\nrun            | ndcg@1 | ndcg@5 | ndcg@10 | mrr@10")
    print("---------------------------------------------------")
    for name, m in results:
        print(f"{name:14s} | {m['ndcg@1']:.4f} | {m['ndcg@5']:.4f} | {m['ndcg@10']:.4f} | {m['mrr@10']:.4f}")
    print(f"\nWrote {metrics_csv}")

if __name__ == "__main__":
    main()
