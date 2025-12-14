#!/usr/bin/env python3
"""
Pure scorer for generated QA CSVs (no retrieval, no embedding, no LLM calls).
- Input CSV must have columns: query, gt, ans.
- Optional column: contexts (JSON list of strings) to enable ragas metrics.
- Outputs: one CSV with query, gt, ans, rouge1, rouge2, rougeL, bleu,
  faithfulness, answer_correctness, context_recall, context_precision,
  answer_relevancy, accuracy.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure MedDiscover package is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
env_meddiscover = os.environ.get("MEDDISCOVER_PATH")
MEDDISCOVER_DIR = Path(env_meddiscover).expanduser().resolve() if env_meddiscover else (REPO_ROOT / "MedDiscover")
if str(MEDDISCOVER_DIR) not in sys.path:
    sys.path.insert(0, str(MEDDISCOVER_DIR))

from med_discover_ai.evaluation import evaluate_response

# Optional ragas metrics
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness as ragas_faithfulness,
        answer_correctness as ragas_answer_correctness,
        context_recall as ragas_context_recall,
        context_precision as ragas_context_precision,
        answer_relevancy as ragas_answer_relevancy,
    )
    from datasets import Dataset

    HAS_RAGAS = True
except Exception:
    HAS_RAGAS = False


def compute_ragas(query: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, Any]:
    if not HAS_RAGAS or not contexts:
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
            "accuracy": None,
        }
    try:
        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truths": [[ground_truth]],
                "reference": [ground_truth],  # some ragas metrics expect this field
            }
        )
        scores = ragas_evaluate(
            ds,
            metrics=[
                ragas_faithfulness,
                ragas_answer_correctness,
                ragas_context_recall,
                ragas_context_precision,
                ragas_answer_relevancy,
            ],
        )
        s = {k: float(v) for k, v in scores.items()}
        s["accuracy"] = float(s["answer_correctness"] > 0.5) if s.get("answer_correctness") is not None else None
        return s
    except Exception as e:
        print(f"[warn] ragas evaluation failed ({e}); setting ragas metrics to None")
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
            "accuracy": None,
        }


def parse_contexts(val: Any) -> List[str]:
    """Accept list/JSON/string; return list of non-empty strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x]
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x]
    except Exception:
        pass
    return [str(val)] if str(val).strip() else []


def run(args):
    df = pd.read_csv(args.qa_csv)
    required = {"query", "gt", "ans"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required}; found {df.columns.tolist()}")

    has_contexts_col = "contexts" in df.columns
    has_context_text = "context_text" in df.columns

    rows = []
    for _, r in df.iterrows():
        query = str(r["query"])
        gt = str(r["gt"])
        ans = str(r["ans"])

        rouge = evaluate_response(gt, ans)
        contexts = []
        if has_contexts_col:
            contexts = parse_contexts(r["contexts"])
        elif has_context_text:
            contexts = parse_contexts(r["context_text"])
        ragas_scores = compute_ragas(query, ans, contexts, gt)

        rows.append(
            {
                "query": query,
                "gt": gt,
                "ans": ans,
                **rouge,
                **ragas_scores,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[done] wrote {len(rows)} rows to {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Pure scorer for QA CSVs (no retrieval/LLM).")
    ap.add_argument("--qa_csv", required=True, help="Input CSV with columns query,gt,ans; optional contexts.")
    ap.add_argument("--out_csv", default="./eval_outputs/metrics_offline.csv", help="Output CSV path.")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
