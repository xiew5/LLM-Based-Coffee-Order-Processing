from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

from .api_client import GemmaAPIClient, GemmaAPIConfig
from .data_loader import inspect_dataframe, load_test_data, load_train_data
from .error_analysis import analyze_errors
from .evaluator import evaluate_records
from .parser import parse_and_validate_model_output
from .preprocess import STRATEGY_REGISTRY, apply_strategy
from .prompt_builder import build_prompt, load_menu_text, sample_few_shots
from .splitters import add_subset_tags, kfold_splits, train_val_split
from .utils import ensure_output_dirs, get_logger, read_jsonl, set_seed, write_json, write_jsonl


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_empty_prediction() -> Dict[str, Any]:
    return {"items": [], "total_price": 0.0}


def _resolve_project_root(cli_project_root: Optional[str]) -> Path:
    if cli_project_root:
        return Path(cli_project_root).resolve()
    return Path(__file__).resolve().parents[1]


def _maybe_load_dotenv(project_root: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(project_root / ".env")


def _build_client(args: argparse.Namespace, paths: Dict[str, Path]) -> GemmaAPIClient:
    config = GemmaAPIConfig(
        api_key_env=args.api_key_env,
        model_name=args.model_name,
        api_provider=args.api_provider,
        api_base_url=args.api_base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff,
        requests_per_minute=args.requests_per_minute,
    )
    return GemmaAPIClient(config=config, raw_response_dir=paths["raw_responses"])


def _run_inference_single(
    text: str,
    strategy: str,
    menu_text: str,
    few_shots: List[Dict[str, Any]],
    include_few_shot: bool,
    client: Optional[GemmaAPIClient],
    request_id: str,
    mock_mode: str,
) -> Dict[str, Any]:
    processed_text = apply_strategy(text, strategy)

    if mock_mode == "empty":
        model_output = json.dumps(_default_empty_prediction(), ensure_ascii=False)
    elif mock_mode == "echo":
        model_output = processed_text
    else:
        if client is None:
            raise RuntimeError("Client is required when mock_mode='none'.")
        prompt_bundle = build_prompt(
            order_text=processed_text,
            menu_text=menu_text,
            few_shots=few_shots,
            include_few_shot=include_few_shot,
        )
        model_output = client.generate(prompt_bundle=prompt_bundle, request_id=request_id)

    parse_result = parse_and_validate_model_output(model_output)
    pred_obj = parse_result.parsed_obj if parse_result.is_valid else None

    return {
        "processed_text": processed_text,
        "model_output": model_output,
        "pred_obj": pred_obj,
        "parse_valid": parse_result.is_valid,
        "parse_repair_used": parse_result.repair_used,
        "parse_errors": parse_result.errors,
    }


def command_validate(args: argparse.Namespace) -> None:
    project_root = _resolve_project_root(args.project_root)
    paths = ensure_output_dirs(project_root)
    log_path = paths["logs"] / f"validate_{_timestamp()}.log"
    logger = get_logger(log_file=log_path)
    _maybe_load_dotenv(project_root)

    set_seed(args.seed)

    train_data = load_train_data(project_root / "train.csv")
    valid_df = train_data.df[train_data.df["label_is_valid"]].copy().reset_index(drop=True)
    tagged_df = add_subset_tags(valid_df, train_data.text_col)

    if args.sample_size and args.sample_size > 0:
        tagged_df = tagged_df.head(args.sample_size).copy().reset_index(drop=True)

    logger.info("Data summary: %s", inspect_dataframe(valid_df, train_data.text_col, train_data.label_col))
    logger.info("Effective validation dataset size: %d", len(tagged_df))

    menu_text = load_menu_text(project_root / "menu.md")
    client = None if args.mock_mode != "none" else _build_client(args, paths)

    comparison_rows: List[Dict[str, Any]] = []
    subset_comparison_rows: List[Dict[str, Any]] = []

    if args.k_folds > 1:
        fold_splits = kfold_splits(
            tagged_df,
            k=args.k_folds,
            seed=args.seed,
            stratify=not args.no_stratify,
        )
        logger.info("Using %d-fold cross-validation.", args.k_folds)
    else:
        train_split, val_split = train_val_split(
            tagged_df,
            val_ratio=args.val_ratio,
            seed=args.seed,
            stratify=not args.no_stratify,
        )
        if args.max_examples and args.max_examples > 0:
            val_split = val_split.head(args.max_examples).copy()
        fold_splits = [(train_split, val_split)]
        logger.info("Validation split size: %d | Train split size: %d", len(val_split), len(train_split))

    for strategy in args.strategies:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_records: List[Dict[str, Any]] = []
        fold_overalls: List[Dict[str, Any]] = []

        for fold_idx, (train_split, val_split) in enumerate(fold_splits, start=1):
            few_shots = sample_few_shots(
                train_df=train_split,
                text_col=train_data.text_col,
                gold_obj_col="parsed_label",
                n_examples=args.few_shot_examples,
                seed=args.seed + fold_idx,
            )

            records: List[Dict[str, Any]] = []
            for _, row in val_split.iterrows():
                row_id = row[train_data.id_col]
                text = str(row[train_data.text_col])
                try:
                    infer = _run_inference_single(
                        text=text,
                        strategy=strategy,
                        menu_text=menu_text,
                        few_shots=few_shots,
                        include_few_shot=not args.disable_few_shot,
                        client=client,
                        request_id=f"validate_{strategy}_fold{fold_idx}_{row_id}",
                        mock_mode=args.mock_mode,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Inference failed for strategy=%s fold=%d id=%s: %s",
                        strategy,
                        fold_idx,
                        row_id,
                        exc,
                    )
                    infer = {
                        "processed_text": apply_strategy(text, strategy),
                        "model_output": "",
                        "pred_obj": None,
                        "parse_valid": False,
                        "parse_repair_used": False,
                        "parse_errors": [f"inference_error: {exc}"],
                    }

                record = {
                    "id": row_id,
                    "fold": fold_idx,
                    "text": text,
                    "strategy": strategy,
                    "gold_obj": row["parsed_label"],
                    **{k: bool(row.get(k, False)) for k in ["correction", "cancellation", "negation", "multi_item", "long_context"]},
                    **infer,
                }
                records.append(record)

            eval_result_fold = evaluate_records(records)
            fold_overalls.append(eval_result_fold["overall"])
            all_records.extend(records)
            logger.info(
                "Completed strategy '%s' fold %d/%d. Fold metrics: %s",
                strategy,
                fold_idx,
                len(fold_splits),
                eval_result_fold["overall"],
            )

        eval_result = evaluate_records(all_records)
        error_result = analyze_errors(eval_result["scored_rows"], max_examples_per_type=args.max_error_examples)

        predictions_path = paths["predictions"] / f"val_predictions_{strategy}_{_timestamp()}.jsonl"
        write_jsonl(predictions_path, all_records)

        metrics_path = paths["metrics"] / f"val_metrics_{strategy}_{_timestamp()}.json"
        write_json(
            metrics_path,
            {
                "overall": eval_result["overall"],
                "subsets": eval_result["subsets"],
                "k_folds": args.k_folds,
                "fold_overall_metrics": fold_overalls,
            },
        )

        error_counts_path = paths["error_analysis"] / f"error_counts_{strategy}_{_timestamp()}.json"
        error_examples_path = paths["error_analysis"] / f"error_examples_{strategy}_{_timestamp()}.json"
        write_json(error_counts_path, error_result["counts"])
        write_json(error_examples_path, error_result["examples"])

        scored_rows_path = paths["error_analysis"] / f"scored_rows_{strategy}_{_timestamp()}.jsonl"
        write_jsonl(scored_rows_path, eval_result["scored_rows"])

        comparison_row = {"strategy": strategy, **eval_result["overall"], "k_folds": args.k_folds}
        comparison_rows.append(comparison_row)

        for subset_name, metrics in eval_result["subsets"].items():
            subset_row = {"strategy": strategy, "subset": subset_name, "k_folds": args.k_folds, **metrics}
            subset_comparison_rows.append(subset_row)

        logger.info("Completed strategy '%s'. Overall CV metrics: %s", strategy, eval_result["overall"])

    comparison_df = pd.DataFrame(comparison_rows)
    subset_df = pd.DataFrame(subset_comparison_rows)

    comparison_csv = paths["metrics"] / f"strategy_comparison_{_timestamp()}.csv"
    subset_csv = paths["metrics"] / f"subset_comparison_{_timestamp()}.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    subset_df.to_csv(subset_csv, index=False)

    logger.info("Saved strategy comparison: %s", comparison_csv)
    logger.info("Saved subset comparison: %s", subset_csv)


def command_predict_test(args: argparse.Namespace) -> None:
    project_root = _resolve_project_root(args.project_root)
    paths = ensure_output_dirs(project_root)
    log_path = paths["logs"] / f"predict_test_{_timestamp()}.log"
    logger = get_logger(log_file=log_path)
    _maybe_load_dotenv(project_root)

    set_seed(args.seed)

    train_data = load_train_data(project_root / "train.csv")
    train_valid_df = train_data.df[train_data.df["label_is_valid"]].copy().reset_index(drop=True)

    test_data = load_test_data(project_root / "test.csv")
    test_df = test_data.df.copy()
    if args.max_examples and args.max_examples > 0:
        test_df = test_df.head(args.max_examples).copy()

    menu_text = load_menu_text(project_root / "menu.md")
    few_shots = sample_few_shots(
        train_df=train_valid_df,
        text_col=train_data.text_col,
        gold_obj_col="parsed_label",
        n_examples=args.few_shot_examples,
        seed=args.seed,
    )

    client = None if args.mock_mode != "none" else _build_client(args, paths)

    rows: List[Dict[str, Any]] = []
    for _, row in test_df.iterrows():
        row_id = row[test_data.id_col]
        text = str(row[test_data.text_col])
        infer = _run_inference_single(
            text=text,
            strategy=args.strategy,
            menu_text=menu_text,
            few_shots=few_shots,
            include_few_shot=not args.disable_few_shot,
            client=client,
            request_id=f"test_{args.strategy}_{row_id}",
            mock_mode=args.mock_mode,
        )

        pred_obj = infer["pred_obj"] if infer["pred_obj"] is not None else _default_empty_prediction()

        rows.append(
            {
                "id": row_id,
                "text": text,
                "strategy": args.strategy,
                "predicted_json_obj": pred_obj,
                "predicted_json": json.dumps(pred_obj, ensure_ascii=False),
                "parse_valid": infer["parse_valid"],
                "parse_repair_used": infer["parse_repair_used"],
                "parse_errors": infer["parse_errors"],
            }
        )

    out_path = Path(args.output_path) if args.output_path else paths["predictions"] / f"test_predictions_{args.strategy}_{_timestamp()}.jsonl"
    write_jsonl(out_path, rows)
    logger.info("Saved test predictions to: %s", out_path)


def _extract_submission_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(row.get("predicted_json_obj"), dict):
        return row["predicted_json_obj"]

    candidate = row.get("predicted_json") or row.get("pred_obj")
    if isinstance(candidate, dict):
        return candidate

    if isinstance(candidate, str):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return _default_empty_prediction()


def command_make_submission(args: argparse.Namespace) -> None:
    project_root = _resolve_project_root(args.project_root)
    paths = ensure_output_dirs(project_root)
    logger = get_logger(log_file=paths["logs"] / f"make_submission_{_timestamp()}.log")

    prediction_rows = read_jsonl(Path(args.input))
    pred_map: Dict[Any, Dict[str, Any]] = {}
    for row in prediction_rows:
        pred_map[row.get("id")] = _extract_submission_obj(row)

    sample_path = project_root / "sample_submission.csv"
    sample_df = pd.read_csv(sample_path)

    if "id" not in sample_df.columns or "predicted_json" not in sample_df.columns:
        raise ValueError("sample_submission.csv must contain 'id' and 'predicted_json' columns")

    output_rows: List[Dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        sample_id = row["id"]
        pred_obj = pred_map.get(sample_id, _default_empty_prediction())
        output_rows.append(
            {
                "id": sample_id,
                "predicted_json": json.dumps(pred_obj, ensure_ascii=False),
            }
        )

    out_df = pd.DataFrame(output_rows)
    output_path = Path(args.output) if args.output else paths["predictions"] / f"submission_{_timestamp()}.csv"
    out_df.to_csv(output_path, index=False)
    logger.info("Saved submission file to: %s", output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coffee order parsing NLP coursework pipeline")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")

    common_api = argparse.ArgumentParser(add_help=False)
    common_api.add_argument("--api-key-env", type=str, default=os.getenv("API_KEY_ENV", "GEMMA_API_KEY"))
    common_api.add_argument("--model-name", type=str, default=os.getenv("GEMMA_MODEL_NAME", "qwen2.5-coder:7b-instruct"))
    common_api.add_argument(
        "--api-provider",
        type=str,
        default=os.getenv("API_PROVIDER", "ollama"),
        choices=["ollama", "qwen_openai", "openai_compatible", "google_genai"],
    )
    common_api.add_argument(
        "--api-base-url",
        type=str,
        default=os.getenv("API_BASE_URL", "http://localhost:11434"),
        help="Base URL for ollama/openai-compatible providers (ignored for google_genai).",
    )
    common_api.add_argument("--temperature", type=float, default=0.0)
    common_api.add_argument("--max-tokens", type=int, default=512)
    common_api.add_argument("--timeout", type=int, default=60)
    common_api.add_argument("--max-retries", type=int, default=3)
    common_api.add_argument("--retry-backoff", type=float, default=1.5)
    common_api.add_argument("--requests-per-minute", type=int, default=30)
    common_api.add_argument("--mock-mode", choices=["none", "empty", "echo"], default="none")

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", parents=[common_api], help="Run validation experiments")
    validate_parser.add_argument(
        "--strategies",
        nargs="+",
        default=["raw", "normalize_numbers", "full_stateful_normalized"],
    )
    validate_parser.add_argument("--val-ratio", type=float, default=0.2)
    validate_parser.add_argument("--seed", type=int, default=42)
    validate_parser.add_argument("--max-examples", type=int, default=0)
    validate_parser.add_argument("--sample-size", type=int, default=0)
    validate_parser.add_argument("--k-folds", type=int, default=1)
    validate_parser.add_argument("--few-shot-examples", type=int, default=2)
    validate_parser.add_argument("--disable-few-shot", action="store_true")
    validate_parser.add_argument("--max-error-examples", type=int, default=10)
    validate_parser.add_argument("--no-stratify", action="store_true")

    predict_parser = subparsers.add_parser("predict-test", parents=[common_api], help="Predict on test set")
    predict_parser.add_argument("--strategy", type=str, default="full_stateful_normalized")
    predict_parser.add_argument("--seed", type=int, default=42)
    predict_parser.add_argument("--max-examples", type=int, default=0)
    predict_parser.add_argument("--few-shot-examples", type=int, default=2)
    predict_parser.add_argument("--disable-few-shot", action="store_true")
    predict_parser.add_argument("--output-path", type=str, default="")

    submit_parser = subparsers.add_parser("make-submission", help="Create submission csv from predictions jsonl")
    submit_parser.add_argument("--input", type=str, required=True)
    submit_parser.add_argument("--output", type=str, default="")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "validate":
        command_validate(args)
    elif args.command == "predict-test":
        command_predict_test(args)
    elif args.command == "make-submission":
        command_make_submission(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
