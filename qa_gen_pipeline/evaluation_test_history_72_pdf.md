## QA Generation / Evaluation History (72 PDFs)

- **Latest corpus run (incremental, resume enabled)**  
  Command template used (CPU, OpenAI Ada embeddings, gpt-4.1-mini, ~20s throttle):  
  ```bash
  python qa_gen_pipeline/generate_qa.py \
    --pdfs "/home/vatsal1/Documents/github/MedDiscover-Project/RAG Biomedical Publication - Additional Articles for Reviewers"/*/*.pdf \
    --out_csv qa_gen_pipeline/outputs/qa_full_v2.csv \
    --llm_model gpt-4.1-mini \
    --embedding_model "OpenAI Ada-002 (CPU/Cloud)" \
    --questions_per_pdf 3 \
    --max_questions 100000 \
    --max_tokens 1024 \
    --disease_category biomedical-review \
    --min_interval_sec 20 \
    --context_truncate_words 300 \
    --jsonl \
    --resume \
    --seed 42 \
    --max_retries 6 \
    --retry_delay 5
  # GPU Available: False, Using device: cpu
  ```
  - Result file: `qa_gen_pipeline/outputs/qa_full_v2.csv`  
    - 216 rows (72 unique source_file PDFs), 3 QAs per PDF.  
    - JSONL sidecar: `qa_gen_pipeline/outputs/qa_full_v2.csv.jsonl`.
  - This run resumed from the prior 60-row file and skipped already-seen PDFs by `source_file`.

- **Earlier preview**  
  - `qa_gen_pipeline/outputs/qa_full.csv` — 60 rows, 21 PDFs (3 QAs each).  
  - `qa_gen_pipeline/outputs/qa_20_papers.csv` — same 60-row preview; alt exports:  
    - Tab-separated: `qa_gen_pipeline/outputs/qa_20_papers.tsv`  
    - Semicolon-separated: `qa_gen_pipeline/outputs/qa_20_papers_semicolon.csv`

### MedCPT execution plan (not yet run for this corpus)
- To generate QAs with MedCPT embeddings on CPU:  
  ```bash
  export ALLOW_MEDCPT_CPU=1
  python qa_gen_pipeline/generate_qa.py ... \
    --embedding_model "MedCPT (GPU Recommended)" \
    --llm_model gpt-4.1-mini \
    --min_interval_sec 20 \
    --resume --seed 42 --max_retries 6 --retry_delay 5
  ```
  (LLM remains gpt-4.1-mini; MedCPT embed will be slower on CPU.)
- For evaluation with MedCPT: same env var + embedding flag in `meddiscover-eval` (see EVALUATION-README.md).

### Git inventory (top-level, max depth 2)
- Repos with `.git`: `deepseek-vl2-small`, `granite-docling-258m-demo`, `VisRAG`, `MedDiscover`, `gpt-oss-20b-demo`, `gemma-3-12b-it`, `MiniCPM-V`, `MedDiscover-HF`, `rag-llm-metabolomics`, `MedDiscover-Analysis`, `granite-vision-demo`, `rag_paper`.
- Non-git dirs (selected): `qa_gen_pipeline`, `RAG Biomedical Publication - Additional Articles for Reviewers`, `rag-gen-code-samples`, `rag-gen-papers`, `new-QA-set`, `evaluation_runner`, `eval_outputs_*`, `RAG-QA-GENERATION-PlAN.md` docs, tokens, ZIPs, etc.
