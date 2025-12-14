# Evaluation Runner

Standalone CLI that reuses the existing MedDiscover code to:
1) ingest and chunk PDFs,
2) build a FAISS index,
3) run retrieval + LLM answer generation for a QA list,
4) compute metrics, and
5) save per-model CSVs plus a combined table.

## Quick start
```bash
python -m evaluation_runner.run_evaluation \
  --pdfs /path/to/a.pdf /path/to/b.pdf \
  --qa_csv /path/to/questions.csv \
  --embedding_model "MedCPT (GPU Recommended)" \
  --llm_models gpt-4.1-mini,gpt-4.1-nano \
  --k 5 \
  --out_dir ./eval_outputs
```

Expected QA CSV columns: `query,gt`. Additional columns are ignored.

Outputs (under `--out_dir`):
- `per_model/<model>.csv` — one row per question with: `query, gt, ans, retrieved_contexts, rouge1, rouge2, rougeL, bleu, faithfulness, answer_correctness, context_recall, context_precision, answer_relevancy, accuracy, model`
- `all_models_combined.csv` — concatenation of all per-model rows.
- `artifacts/index.faiss` and `artifacts/doc_metadata.json` — built index + metadata for reuse.

## Quick start (metrics-only CSV, no retrieved contexts)
```bash
python -m evaluation_runner.evaluate_qas_simple \
  --pdfs /path/to/a.pdf /path/to/b.pdf \
  --qa_csv /path/to/questions.csv \
  --embedding_model "MedCPT (GPU Recommended)" \
  --llm_models gpt-4.1-mini \
  --k 5 \
  --out_csv ./eval_outputs/metrics_only.csv \
  --out_artifacts ./eval_outputs/artifacts
```
Outputs a single CSV with: `query, gt, ans, rouge1, rouge2, rougeL, bleu, faithfulness, answer_correctness, context_recall, context_precision, answer_relevancy, accuracy, model`. No contexts are stored. Artifacts (index + metadata) are kept under `--out_artifacts` for reuse/debug.

## Quick start (pure scorer, no retrieval/LLM; optional contexts for ragas)
```bash
python -m evaluation_runner.score_qas_offline \
  --qa_csv /path/to/answers.csv \  # columns: query, gt, ans; optional contexts
  --out_csv ./eval_outputs/metrics_offline.csv
```
Computes ROUGE-1/2/L and BLEU always; ragas metrics (faithfulness, answer_correctness, context_recall, context_precision, answer_relevancy, accuracy) are filled only if a `contexts` column is present (JSON list). Otherwise those are set to None. No embedding, retrieval, or LLM calls are performed.
