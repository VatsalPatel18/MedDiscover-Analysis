#!/usr/bin/env python3
"""
Slim evaluation CLI for generated QA files.
- Inputs: PDFs + QA CSV (columns: query, gt).
- Outputs: single CSV with question, answer, and the 10 metrics used in prior studies
  (rouge1/2/L, bleu, faithfulness, answer_correctness, context_recall, context_precision,
   answer_relevancy, accuracy) plus model.
- Omits retrieved contexts to keep the artifact compact.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure the MedDiscover package is importable (same pattern as run_evaluation.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
env_meddiscover = os.environ.get("MEDDISCOVER_PATH")
MEDDISCOVER_DIR = Path(env_meddiscover).expanduser().resolve() if env_meddiscover else (REPO_ROOT / "MedDiscover")
if str(MEDDISCOVER_DIR) not in sys.path:
    sys.path.insert(0, str(MEDDISCOVER_DIR))

from med_discover_ai.chunking import chunk_text
from med_discover_ai.pdf_utils import extract_text_from_pdf
from med_discover_ai.embeddings import embed_documents
from med_discover_ai.index import build_faiss_index, save_index
from med_discover_ai.retrieval import search_and_rerank
from med_discover_ai.llm_inference import get_llm_answer
from med_discover_ai.evaluation import evaluate_response
from med_discover_ai.config import (
    CHUNK_SIZE,
    OVERLAP,
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LLM_MODEL,
    DEFAULT_K,
    DEFAULT_RERANK_ENABLED,
    DEFAULT_MAX_TOKENS,
    get_embedding_model_id,
)

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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_qa(qa_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(qa_csv)
    if not {"query", "gt"}.issubset(df.columns):
        raise ValueError(f"QA CSV must contain columns ['query','gt']; found {df.columns.tolist()}")
    return df[["query", "gt"]]


def build_index_from_pdfs(pdf_paths: List[Path], embedding_model: str, artifacts_dir: Path):
    """Copy of main logic but without returning contexts; we only need index + metadata."""
    all_chunks, metadata_list = [], []
    doc_id_counter = 0
    for pdf in pdf_paths:
        text = extract_text_from_pdf(str(pdf))
        if not text or text.startswith("Error reading"):
            print(f"[skip] {pdf}: no text extracted.")
            continue
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        if not chunks:
            print(f"[skip] {pdf}: no chunks produced.")
            continue
        for chunk_id, chunk_text_content in enumerate(chunks):
            metadata_list.append(
                {
                    "doc_id": doc_id_counter,
                    "filename": pdf.name,
                    "chunk_id": chunk_id,
                    "text": chunk_text_content,
                }
            )
            all_chunks.append(chunk_text_content)
        doc_id_counter += 1

    if not all_chunks:
        raise RuntimeError("No text extracted/chunked from provided PDFs.")

    embeddings = embed_documents(all_chunks, embedding_model)
    if embeddings is None or embeddings.shape[0] == 0:
        raise RuntimeError("Embedding generation failed.")

    model_id = get_embedding_model_id(embedding_model)
    use_ip = model_id == AVAILABLE_EMBEDDING_MODELS.get("MedCPT (GPU Recommended)")
    index = build_faiss_index(embeddings, use_ip_metric=use_ip)
    if index is None:
        raise RuntimeError("FAISS index build failed.")

    ensure_parent(artifacts_dir)
    save_index(index, str(artifacts_dir / "index.faiss"))
    with open(artifacts_dir / "doc_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    return index, metadata_list


def compute_ragas_metrics(query: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    """Single-sample ragas evaluation to mirror prior runs."""
    if not HAS_RAGAS:
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
        }
    if not contexts or str(answer).startswith("Error"):
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
        }
    try:
        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truths": [[ground_truth]],
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
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        print(f"[warn] ragas evaluation failed ({e}); returning None metrics.")
        return {
            "faithfulness": None,
            "answer_correctness": None,
            "context_recall": None,
            "context_precision": None,
            "answer_relevancy": None,
        }


def evaluate_one(query: str, gt: str, contexts: List[Dict[str, Any]], ans: str) -> Dict[str, Any]:
    context_texts = [c.get("text", "") for c in contexts]
    ragas_scores = compute_ragas_metrics(query, ans, context_texts, gt)
    rouge_bleu = evaluate_response(gt, ans)
    accuracy = None
    if ragas_scores.get("answer_correctness") is not None:
        accuracy = float(ragas_scores["answer_correctness"] > 0.5)
    return {
        "rouge1": rouge_bleu.get("rouge1"),
        "rouge2": rouge_bleu.get("rouge2"),
        "rougeL": rouge_bleu.get("rougeL"),
        "bleu": rouge_bleu.get("bleu"),
        **ragas_scores,
        "accuracy": accuracy,
    }


def run(args):
    qa_df = load_qa(Path(args.qa_csv))
    pdf_paths = [Path(p) for p in args.pdfs]

    artifacts_dir = Path(args.out_artifacts)
    index, metadata = build_index_from_pdfs(pdf_paths, args.embedding_model, artifacts_dir)

    all_rows = []
    for model in args.llm_models:
        print(f"\n=== Evaluating model: {model} ===")
        for _, row in qa_df.iterrows():
            query = str(row["query"])
            gt = str(row["gt"])

            candidates = search_and_rerank(
                query=query,
                index=index,
                doc_metadata=metadata,
                embedding_model_display_name=args.embedding_model,
                k=args.k,
                enable_rerank=args.rerank,
            )

            answer, _ctx_block, _usage = get_llm_answer(
                query=query,
                retrieved_candidates=candidates,
                llm_model=model,
                max_tokens=args.max_tokens,
            )

            metrics = evaluate_one(query, gt, candidates, answer)
            row_out = {
                "query": query,
                "gt": gt,
                "ans": answer,
                **metrics,
                "model": model,
            }
            all_rows.append(row_out)

    out_path = Path(args.out_csv)
    ensure_parent(out_path)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nCombined results written to {out_path} ({len(all_rows)} rows).")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated QA CSV and emit metric-only table.")
    parser.add_argument("--pdfs", nargs="+", required=True, help="PDF file paths.")
    parser.add_argument("--qa_csv", required=True, help="CSV with columns query,gt.")
    parser.add_argument(
        "--embedding_model",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help=f"Embedding model display name (default: {DEFAULT_EMBEDDING_MODEL_NAME}).",
    )
    parser.add_argument(
        "--llm_models",
        default=DEFAULT_LLM_MODEL,
        help="Comma-separated list of LLM model identifiers (OpenAI or ollama:...).",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k for retrieval.")
    parser.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=DEFAULT_RERANK_ENABLED,
        help="Enable cross-encoder rerank (GPU only).",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable re-ranking even if GPU is available.",
    )
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for generation.")
    parser.add_argument("--out_csv", default="./eval_outputs/metrics_only.csv", help="Output CSV (metrics only).")
    parser.add_argument(
        "--out_artifacts",
        default="./eval_outputs/artifacts",
        help="Where to store index/doc_metadata (for reuse/debug).",
    )
    args = parser.parse_args()
    args.llm_models = [m.strip() for m in args.llm_models.split(",") if m.strip()]
    return args


if __name__ == "__main__":
    run(parse_args())
