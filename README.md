# Coffee Order Parsing Preprocessing Study (Local Ollama + Qwen2.5-Coder)

This project is a reproducible NLP coursework pipeline to study:

**How text preprocessing affects structured coffee-order parsing accuracy**, with extra focus on:
- correction
- cancellation
- negation

It uses `train.csv` / `test.csv` / `menu.md` and runs controlled preprocessing experiments, validation metrics, subset metrics, and error analysis.

## What this project includes

- Modular Python package under `src/`
- Multiple named preprocessing strategies (raw, lowercase, normalization variants)
- Rule-based subset tagging (`correction`, `cancellation`, `negation`, `multi_item`, `long_context`)
- Prompt builder with strict JSON schema and optional few-shot examples
- Isolated API adapter (`src/api_client.py`)
- Robust JSON extraction + minor repair + schema validation for model outputs
- Validation experiment runner with per-strategy metrics and subset comparisons
- Error categorization and representative failure case export
- Test prediction generation and submission export

## Project structure

- `src/data_loader.py`: CSV loading, schema inference, train-label parsing/validation
- `src/preprocess.py`: preprocessing strategies and batch application
- `src/prompt_builder.py`: menu-aware strict prompting + few-shot sampling
- `src/api_client.py`: API wrapper with retries + raw response logging
- `src/parser.py`: model output JSON extraction/repair/validation
- `src/evaluator.py`: exact/item/field/price/subset metrics
- `src/splitters.py`: subset tagging + stratified train/val split
- `src/error_analysis.py`: error type categorization and summaries
- `src/utils.py`: logging, schema helpers, serialization utilities
- `src/main.py`: CLI commands (`validate`, `predict-test`, `make-submission`)

Output artifacts are written to:
- `outputs/predictions/`
- `outputs/metrics/`
- `outputs/error_analysis/`
- `outputs/logs/`

## Setup

1) Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env` and set:
- `API_PROVIDER=ollama`
- `API_BASE_URL=http://localhost:11434`
- `GEMMA_MODEL_NAME=qwen2.5-coder:7b-instruct`

Then start local Ollama and pull model once:

```bash
brew services start ollama
ollama pull qwen2.5-coder:7b-instruct
```

## Run experiments

### 1) Validate preprocessing strategies on train split

```bash
python -m src.main validate --strategies raw lowercase full_normalized
```

Useful options:
- `--val-ratio 0.2`
- `--seed 42`
- `--few-shot-examples 2`
- `--disable-few-shot`
- `--max-examples 50` (quick debugging)
- `--mock-mode empty` (pipeline smoke test without API calls)
- `--api-provider ollama`
- `--api-base-url http://localhost:11434`

### 2) Predict on test set

```bash
python -m src.main predict-test --strategy full_normalized
```

### 3) Build submission CSV

```bash
python -m src.main make-submission --input outputs/predictions/test_predictions_full_normalized_YYYYMMDD_HHMMSS.jsonl
```

## Output files you will use in the paper

- Strategy comparison table CSV (`outputs/metrics/strategy_comparison_*.csv`)
- Subset comparison table CSV (`outputs/metrics/subset_comparison_*.csv`)
- Per-strategy metrics JSON (`outputs/metrics/val_metrics_<strategy>_*.json`)
- Per-example scored rows (`outputs/error_analysis/scored_rows_<strategy>_*.jsonl`)
- Error counts + representative failures (`outputs/error_analysis/error_counts_*.json`, `error_examples_*.json`)
- Raw model responses for debugging (`outputs/logs/raw_responses/*.json`)

## Notes on schema assumptions

The loader uses robust column detection with fallbacks:
- Train text column candidates: `order`, `text`, `input`, `utterance`, `query`
- Train label candidates: `expected_json`, `label`, `target`, `output_json`, `gold_json`
- ID candidates: `id`, `row_id`, `sample_id`

For your current files, detected columns are:
- `train.csv`: `id`, `order`, `expected_json`
- `test.csv`: `id`, `order`
- `sample_submission.csv`: `id`, `predicted_json`

## API adapter note

`src/api_client.py` is ready-to-use as a provider adapter. By default it uses local Ollama mode, and endpoint payload/response parsing is isolated in:
- `_build_request(...)`
- `_extract_text(...)`

If your provider endpoint differs from the current request format, adjust only these methods.

## Limitations

- Preprocessing and subset tags are heuristic, not linguistic parsing.
- JSON repair is conservative by design (does not invent missing fields).
- Field-level item comparison uses greedy pairing (practical, not globally optimal).
