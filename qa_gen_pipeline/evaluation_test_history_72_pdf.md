## QA Generation / Evaluation History (72 PDFs)

- **Latest status (Sun Dec 14 19:53 IST 2025)**  
  - Ada QAs (CPU, OpenAI Ada-002, gpt-4.1-mini): `qa_gen_pipeline/outputs/qa_full_v2.csv` — 216 rows, 72 PDFs (JSONL sidecar: `.csv.jsonl`).  
  - MedCPT QAs (CPU override, MedCPT encoders, gpt-4.1-mini): `qa_gen_pipeline/outputs/qa_full_v2_medcpt.csv` — 218 rows, 72 PDFs (JSONL sidecar).  
  - Corpus: `/home/vatsal1/Documents/github/MedDiscover-Project/RAG Biomedical Publication - Additional Articles for Reviewers/*/*.pdf`.  
  - MedCPT run completed on CPU with `ALLOW_MEDCPT_CPU=1`; `45_b22-00185.pdf` was skipped in logs earlier for no text, but resume filled remaining files to 72/72.

- **Ada corpus run (incremental, resume enabled)**  
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

- **MedCPT corpus run (CPU override, resume enabled)**  
  Command template used:  
  ```bash
  export ALLOW_MEDCPT_CPU=1
  python qa_gen_pipeline/generate_qa.py \
    --pdfs "/home/vatsal1/Documents/github/MedDiscover-Project/RAG Biomedical Publication - Additional Articles for Reviewers"/*/*.pdf \
    --out_csv qa_gen_pipeline/outputs/qa_full_v2_medcpt.csv \
    --llm_model gpt-4.1-mini \
    --embedding_model "MedCPT (GPU Recommended)" \
    --questions_per_pdf 3 --max_questions 100000 \
    --max_tokens 1024 --disease_category biomedical-review \
    --min_interval_sec 20 --context_truncate_words 300 \
    --jsonl --resume --seed 42 --max_retries 6 --retry_delay 5
  # GPU Available: False, Using device: cpu (ALLOW_MEDCPT_CPU=1)
  ```
  - Result file: `qa_gen_pipeline/outputs/qa_full_v2_medcpt.csv`  
    - 218 rows (72 unique source_file PDFs), 3 QAs per PDF.  
    - JSONL sidecar: `qa_gen_pipeline/outputs/qa_full_v2_medcpt.csv.jsonl`.  
  - Run completed after a resume; logged one PDF with no text (`45_b22-00185.pdf`) but final counts cover all 72 PDFs.

- **Earlier preview**  
  - `qa_gen_pipeline/outputs/qa_full.csv` — 60 rows, 21 PDFs (3 QAs each).  
  - `qa_gen_pipeline/outputs/qa_20_papers.csv` — same 60-row preview; alt exports:  
    - Tab-separated: `qa_gen_pipeline/outputs/qa_20_papers.tsv`  
    - Semicolon-separated: `qa_gen_pipeline/outputs/qa_20_papers_semicolon.csv`

### Git inventory (top-level, max depth 2)
- Repos with `.git`: `deepseek-vl2-small`, `granite-docling-258m-demo`, `VisRAG`, `MedDiscover`, `gpt-oss-20b-demo`, `gemma-3-12b-it`, `MiniCPM-V`, `MedDiscover-HF`, `rag-llm-metabolomics`, `MedDiscover-Analysis`, `granite-vision-demo`, `rag_paper`.
- Non-git dirs (selected): `qa_gen_pipeline`, `RAG Biomedical Publication - Additional Articles for Reviewers`, `rag-gen-code-samples`, `rag-gen-papers`, `new-QA-set`, `evaluation_runner`, `eval_outputs_*`, `RAG-QA-GENERATION-PlAN.md` docs, tokens, ZIPs, etc.
