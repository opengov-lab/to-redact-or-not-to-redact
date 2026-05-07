"""
Evaluate all result CSVs and print an overview table.

Usage:
  python evaluate_all.py                          # scan current directory
  python evaluate_all.py --dir path/to/           # scan a specific directory
  python evaluate_all.py --dataset K1 K2          # filter by dataset(s)
  python evaluate_all.py --model gemini           # filter by model(s)
  python evaluate_all.py --sort macro_f1          # sort by a column
  python evaluate_all.py --csv out.csv            # also write results to a CSV
  python evaluate_all.py --markdown               # print a markdown table
  python evaluate_all.py --aggregate              # combine splits, report overall metrics per (model, variant)
  python evaluate_all.py --dataset K1 --chart rec_delib          # bar chart of recall on K1
  python evaluate_all.py --dataset K1 --chart rec_delib --chart-out chart.png  # save chart to file
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# Files to skip (not experiment result files)
SKIP_FILES = {"test.csv", "agentic.csv"}

# Filename pattern: {model}-{variant}_{dataset}.csv
# e.g. gemini-fewshot-cot-hard-multi_R4.csv
FILENAME_RE = re.compile(r"^(?P<model>[^-]+)-(?P<variant>.+)_(?P<dataset>[^_]+)\.csv$")

PREV_PAPER_RESULTS = {
    'K2': {
        'LR': {
            'prec_delib': 0.7, 'rec_delib': 0.6, 'f1_delib': 0.646, 'f2_delib': 0.618, 'mcc': 0.44
        },
        'SVM': {
            'prec_delib': 0.81, 'rec_delib': 0.42, 'f1_delib': 0.553, 'f2_delib': 0.465, 'mcc': 0.42
        },
        'BERT': {
            'prec_delib': 0.79, 'rec_delib': 0.57, 'f1_delib': 0.662, 'f2_delib': 0.604, 'mcc': 0.49
        },
    },
}


def parse_filename(fname: str) -> dict:
    """Extract model, variant, dataset from filename. Returns raw name on mismatch."""
    m = FILENAME_RE.match(fname)
    if m:
        return {
            "model":   m.group("model"),
            "variant": m.group("variant"),
            "dataset": m.group("dataset"),
        }
    # Fallback for files that don't match the pattern
    stem = Path(fname).stem
    return {"model": stem, "variant": "-", "dataset": "-"}


def _compute_metrics(y_true: list, y_pred: list, label: str, model: str, variant: str, dataset: str) -> dict:
    """Compute all classification metrics for the given predictions."""
    n = len(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    mcc      = matthews_corrcoef(y_true, y_pred)

    p1 = precision_score(y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0)
    r1 = recall_score(   y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0)
    f1_delib = f1_score( y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0)
    f2_delib = fbeta_score(y_true, y_pred, beta=2,   average="binary", pos_label=1, zero_division=0)

    p0 = precision_score(y_true, y_pred, labels=[0], average="binary", pos_label=0, zero_division=0)
    r0 = recall_score(   y_true, y_pred, labels=[0], average="binary", pos_label=0, zero_division=0)
    f1_not   = f1_score( y_true, y_pred, labels=[0], average="binary", pos_label=0, zero_division=0)
    f2_not   = fbeta_score(y_true, y_pred, beta=2,   average="binary", pos_label=0, zero_division=0)

    macro_f2 = fbeta_score(y_true, y_pred, beta=2, labels=[0, 1], average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "file":        label,
        "model":       model,
        "variant":     variant,
        "dataset":     dataset,
        "n":           n,
        "accuracy":    round(accuracy, 3),
        "macro_f1":    round(macro_f1, 3),
        "macro_f2":    round(macro_f2, 3),
        "mcc":         round(mcc, 3),
        "f1_delib":    round(f1_delib, 3),
        "f2_delib":    round(f2_delib, 3),
        "prec_delib":  round(p1, 3),
        "rec_delib":   round(r1, 3),
        "f1_not":      round(f1_not, 3),
        "f2_not":      round(f2_not, 3),
        "prec_not":    round(p0, 3),
        "rec_not":     round(r0, 3),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def evaluate_csv(path: Path) -> dict | None:
    """Load a CSV and compute classification metrics. Returns None on failure."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [SKIP] {path.name}: could not read ({e})")
        return None

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        print(f"  [SKIP] {path.name}: missing y_true or y_pred columns")
        return None

    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        print(f"  [SKIP] {path.name}: no valid rows after dropping NaNs")
        return None

    y_true = df["y_true"].astype(int).tolist()
    y_pred = df["y_pred"].astype(int).tolist()

    meta = parse_filename(path.name)
    return _compute_metrics(y_true, y_pred, label=path.name,
                            model=meta["model"], variant=meta["variant"], dataset=meta["dataset"])


def evaluate_aggregated(paths: list[Path]) -> dict | None:
    """Concatenate predictions from multiple CSVs and compute combined metrics."""
    all_true, all_pred = [], []
    datasets = []
    meta = None
    for path in paths:
        try:
            df = pd.read_csv(path).dropna(subset=["y_true", "y_pred"])
        except Exception as e:
            print(f"  [SKIP] {path.name}: could not read ({e})")
            continue
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            continue
        all_true.extend(df["y_true"].astype(int).tolist())
        all_pred.extend(df["y_pred"].astype(int).tolist())
        m = parse_filename(path.name)
        datasets.append(m["dataset"])
        if meta is None:
            meta = m

    if not all_true:
        return None

    dataset_label = "+".join(sorted(set(datasets)))
    label = f"{meta['model']}-{meta['variant']} [{dataset_label}]"
    return _compute_metrics(all_true, all_pred, label=label,
                            model=meta["model"], variant=meta["variant"], dataset=dataset_label)


SORT_COLUMNS = {
    "accuracy",
    "macro_f1", "macro_f2", "mcc",
    "f1_delib", "f2_delib", "prec_delib", "rec_delib",
    "f1_not",   "f2_not",   "prec_not",   "rec_not",
    "model", "variant", "dataset", "n",
}


_NAN = float("nan")

_PAPER_RESULT_DATASETS = set(PREV_PAPER_RESULTS.keys())


def _build_paper_rows() -> list[dict]:
    """Convert PREV_PAPER_RESULTS into metric dicts matching the standard row schema."""
    rows = []
    for dataset, models in PREV_PAPER_RESULTS.items():
        for model_name, metrics in models.items():
            rows.append({
                "file":       f"[paper] {model_name}_{dataset}",
                "model":      model_name,
                "variant":    "paper-baseline",
                "dataset":    dataset,
                "n":          _NAN,
                "accuracy":   _NAN,
                "macro_f1":   _NAN,
                "macro_f2":   _NAN,
                "mcc":        metrics.get("mcc",        _NAN),
                "f1_delib":   metrics.get("f1_delib",   _NAN),
                "f2_delib":   metrics.get("f2_delib",   _NAN),
                "prec_delib": metrics.get("prec_delib", _NAN),
                "rec_delib":  metrics.get("rec_delib",  _NAN),
                "f1_not":     _NAN,
                "f2_not":     _NAN,
                "prec_not":   _NAN,
                "rec_not":    _NAN,
                "TP": _NAN, "TN": _NAN, "FP": _NAN, "FN": _NAN,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate all FOIA result CSVs and print an overview table.")
    parser.add_argument("--dir",  default=".", help="Directory to scan for result CSVs (default: current dir)")
    parser.add_argument("--dataset", nargs="+", metavar="KEY",
                        help="Filter by dataset key(s), e.g. --dataset K1 R4")
    parser.add_argument("--model", nargs="+", metavar="NAME",
                        help="Filter by model name(s), e.g. --model gemini qwen")
    parser.add_argument("--sort", default="model", choices=sorted(SORT_COLUMNS),
                        help="Column to sort results by (default: model)")
    parser.add_argument("--asc",  action="store_true", help="Sort ascending (default: descending for metrics)")
    parser.add_argument("--csv",  default=None, help="Optional path to write the results table as CSV")
    parser.add_argument("--markdown", "-m", action="store_true", help="Print the overview table in Markdown format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-class classification report for each file")
    parser.add_argument("--aggregate", "-a", action="store_true",
                        help="Combine all data splits and report one row per (model, variant)")
    parser.add_argument("--no-paper", action="store_true",
                        help="Exclude prior paper baseline results (PREV_PAPER_RESULTS) from the table")
    parser.add_argument("--chart", metavar="METRIC", default=None,
                        help="Plot a bar chart of the given metric (e.g. rec_delib, macro_f1) after filtering")
    parser.add_argument("--chart-out", metavar="PATH", default=None,
                        help="Save the bar chart to this file instead of showing it interactively")
    args = parser.parse_args()

    scan_dir = Path(args.dir)
    csv_files = sorted(
        f for f in scan_dir.glob("*.csv")
        if f.name not in SKIP_FILES
    )

    if not csv_files:
        print(f"No result CSVs found in {scan_dir.resolve()}")
        return

    print(f"Scanning {len(csv_files)} CSV files in {scan_dir.resolve()} ...\n")

    rows = []
    if args.aggregate:
        # Group files by (model, variant) and concatenate predictions across splits
        from collections import defaultdict
        groups: dict[tuple, list[Path]] = defaultdict(list)
        for path in csv_files:
            meta = parse_filename(path.name)
            groups[(meta["model"], meta["variant"])].append(path)
        for _key, paths in sorted(groups.items()):
            result = evaluate_aggregated(paths)
            if result is not None:
                rows.append(result)
    else:
        for path in csv_files:
            result = evaluate_csv(path)
            if result is not None:
                rows.append(result)

    # Inject prior paper baselines when the relevant dataset is represented
    if not args.no_paper and not args.model:
        present_datasets = {r["dataset"].upper() for r in rows}
        for row in _build_paper_rows():
            if row["dataset"].upper() in present_datasets:
                rows.append(row)

    if not rows:
        print("No evaluable files found.")
        return

    df = pd.DataFrame(rows)

    # Apply filters
    if args.dataset:
        keys = [k.upper() for k in args.dataset]
        df = df[df["dataset"].str.upper().isin(keys)]
        if df.empty:
            print(f"No results match dataset filter: {args.dataset}")
            return
    if args.model:
        names = [m.lower() for m in args.model]
        df = df[df["model"].str.lower().isin(names)]
        if df.empty:
            print(f"No results match model filter: {args.model}")
            return

    # Sort: metrics descending by default, text columns ascending
    metric_cols = {
        "accuracy",
        "macro_f1", "macro_f2", "mcc",
        "f1_delib", "f2_delib", "prec_delib", "rec_delib",
        "f1_not",   "f2_not",   "prec_not",   "rec_not",   "n",
    }
    ascending = args.asc if args.sort in metric_cols else True
    df = df.sort_values(args.sort, ascending=ascending).reset_index(drop=True)

    # ── Variant translation and ordering for markdown output ─────────────────
    VARIANT_LABELS = {
        "simple":              "Simple (baseline)",
        "fewshot":             "Few-Shot",
        "fewshot-hard":        "Error-Based Few-Shot",
        "single":              "CoT",
        "fewshot-cot-hard":    "CoT + Error-Based Few-Shot",
        "multi-nocot":         "Multi-Agent",
        "multi":               "CoT + Multi-agent",
        "fewshot-cot-hard-multi": "CoT + Error-Based Few-Shot + Multi-agent",
    }
    # ── Print overview table ──────────────────────────────────────────────────
    display_cols = [
        "model", "variant", "dataset", "n",
        "accuracy", "prec_delib", "rec_delib", "f1_delib", "f2_delib", "mcc",
    ]
    display = df[display_cols].copy()

    # Pretty column headers
    display.columns = [
        "Model", "Variant", "Dataset", "N",
        "Accuracy", "Prec(delib)", "Rec(delib)", "F1(delib)", "F2(delib)", "MCC",
    ]

    if args.markdown:
        md_cols = ["Variant", "Accuracy", "Prec(delib)", "Rec(delib)", "F1(delib)", "F2(delib)", "MCC"]
        md_display = display[md_cols].copy()
        md_display["Variant"] = md_display["Variant"].map(lambda v: VARIANT_LABELS.get(v, v))

        # Sort by VARIANT_ORDER
        variant_rank = {label: i for i, label in enumerate(VARIANT_LABELS.values())}
        md_display = md_display.sort_values(
            "Variant", key=lambda s: s.map(lambda v: variant_rank.get(v, len(variant_rank)))
        ).reset_index(drop=True)

        def md_row(row):
            return "| " + " | ".join(str(row[c]) for c in md_cols) + " |"

        header = "| " + " | ".join(md_cols) + " |"
        separator_md = "| " + " | ".join("---" for _ in md_cols) + " |"

        print(header)
        print(separator_md)
        for _, row in md_display.iterrows():
            print(md_row(row))
    else:
        # Compute column widths
        col_widths = {col: max(len(col), int(display[col].astype(str).str.len().max())) for col in display.columns}

        def fmt_row(row):
            return "  ".join(str(row[c]).ljust(col_widths[c]) for c in display.columns)

        def fmt_header():
            return "  ".join(c.ljust(col_widths[c]) for c in display.columns)

        def separator():
            return "  ".join("-" * col_widths[c] for c in display.columns)

        print("=" * (sum(col_widths.values()) + 2 * (len(col_widths) - 1)))
        print("FOIA Classification — All Variants Overview")
        print("=" * (sum(col_widths.values()) + 2 * (len(col_widths) - 1)))
        print(fmt_header())
        print(separator())
        for _, row in display.iterrows():
            print(fmt_row(row))
        print(separator())

    print(f"\nTotal variants evaluated: {len(df)}")

    # ── Summary stats (exclude paper baselines which have NaN for some metrics) ─
    df_eval = df[df["variant"] != "paper-baseline"]
    if not df_eval.empty:
        print(f"\n{'Best Macro-F1':>20}: {df_eval.loc[df_eval['macro_f1'].idxmax(), 'file']}  ({df_eval['macro_f1'].max():.4f})")
        print(f"{'Best Macro-F2':>20}: {df_eval.loc[df_eval['macro_f2'].idxmax(), 'file']}  ({df_eval['macro_f2'].max():.4f})")
        print(f"{'Best MCC':>20}: {df_eval.loc[df_eval['mcc'].idxmax(), 'file']}  ({df_eval['mcc'].max():.4f})")
        print(f"{'Best F1(delib)':>20}: {df_eval.loc[df_eval['f1_delib'].idxmax(), 'file']}  ({df_eval['f1_delib'].max():.4f})")
        print(f"{'Best F2(delib)':>20}: {df_eval.loc[df_eval['f2_delib'].idxmax(), 'file']}  ({df_eval['f2_delib'].max():.4f})")

    # ── Per-model average ─────────────────────────────────────────────────────
    print("\n--- Average metrics by model ---")
    avg = df.groupby("model")[["accuracy", "macro_f1", "macro_f2", "mcc", "f1_delib", "f2_delib"]].mean().round(4)
    print(avg.to_string())

    # ── Prior paper baselines (if present) ───────────────────────────────────
    paper_rows = df[df["variant"] == "paper-baseline"]
    if not paper_rows.empty:
        print("\n--- Prior paper baselines ---")
        paper_cols = ["model", "dataset", "prec_delib", "rec_delib", "f1_delib", "f2_delib", "mcc"]
        print(paper_rows[paper_cols].to_string(index=False))

    # ── Verbose: full classification report per file ───────────────────────────
    if args.verbose:
        print("\n" + "=" * 60)
        print("Detailed per-file classification reports")
        print("=" * 60)
        for path in sorted(scan_dir.glob("*.csv")):
            if path.name in SKIP_FILES:
                continue
            try:
                d = pd.read_csv(path).dropna(subset=["y_true", "y_pred"])
                if d.empty:
                    continue
                yt = d["y_true"].astype(int).tolist()
                yp = d["y_pred"].astype(int).tolist()
                print(f"\n--- {path.name} ---")
                print(classification_report(yt, yp, labels=[0, 1],
                                            target_names=["not_deliberative", "deliberative"]))
                cm = confusion_matrix(yt, yp, labels=[0, 1])
                print(f"Confusion matrix:\n{cm}")
            except Exception:
                pass

    # ── Optional CSV export ────────────────────────────────────────────────────
    if args.csv:
        out_path = Path(args.csv)
        df.to_csv(out_path, index=False)
        print(f"\nResults written to {out_path.resolve()}")

    # ── Optional bar chart ────────────────────────────────────────────────────
    if args.chart:
        metric = args.chart
        valid_metrics = sorted(SORT_COLUMNS - {"model", "variant", "dataset"})
        if metric not in df.columns:
            print(f"\n[ERROR] Unknown metric '{metric}' for --chart. Choose from: {', '.join(valid_metrics)}")
        else:
            import matplotlib.pyplot as plt
            import numpy as np

            chart_df = df[["model", "variant", "dataset", metric]].copy()
            # Paper baselines get their model name as label; others use VARIANT_LABELS
            chart_df["label"] = chart_df.apply(
                lambda r: r["model"] if r["variant"] == "paper-baseline"
                          else VARIANT_LABELS.get(r["variant"], r["variant"]),
                axis=1,
            )

            # Split experiment rows from paper baseline rows
            exp_df   = chart_df[chart_df["variant"] != "paper-baseline"].copy()
            paper_df = chart_df[chart_df["variant"] == "paper-baseline"].copy()

            # Sort experiment rows by canonical VARIANT_LABELS order
            variant_rank = {label: i for i, label in enumerate(VARIANT_LABELS.values())}
            exp_df = exp_df.sort_values(
                "label", key=lambda s: s.map(lambda v: variant_rank.get(v, len(variant_rank)))
            ).reset_index(drop=True)

            # Sort paper rows alphabetically by model name
            paper_df = paper_df.sort_values("label").reset_index(drop=True)

            exp_labels   = list(dict.fromkeys(exp_df["label"]))
            paper_labels = list(dict.fromkeys(paper_df["label"]))
            all_labels   = paper_labels + exp_labels   # paper on the left
            n_exp   = len(exp_labels)
            n_paper = len(paper_labels)

            models_present   = sorted(exp_df["model"].unique()) if not exp_df.empty else []
            datasets_present = sorted(exp_df["dataset"].unique()) if not exp_df.empty else []
            n_models   = len(models_present)
            n_datasets = len(datasets_present)

            model_labels = {
                'gemini': 'Gemini 2.5-Flash',
                'qwen': 'Qwen-3.5 9B',
            }

            use_model_grouping = not args.model and n_models > 1

            fig, ax = plt.subplots(figsize=(max(8, len(all_labels) * 1.5), 5))
            colors = plt.cm.tab10.colors
            x_exp = np.arange(n_paper, n_paper + n_exp)   # experiment bars start after paper

            # ── Paper baseline bars (left side) ──────────────────────────
            if n_paper > 0:
                x_paper = np.arange(n_paper)
                paper_vals = paper_df.groupby("label")[metric].mean().reindex(paper_labels).fillna(0)
                bars = ax.bar(x_paper, paper_vals, color="silver", edgecolor="white",
                              hatch="//", label="Paper baselines (trained on K1 and K3)")
                for bar, val in zip(bars, paper_vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
                # Dashed separator between paper baselines and variants
                if n_exp > 0:
                    ax.axvline(x=n_paper - 0.5, color="gray", linestyle="--", linewidth=1)

            # ── Experiment bars ───────────────────────────────────────────
            if not exp_df.empty:
                if use_model_grouping:
                    agg = exp_df.groupby(["model", "label"])[metric].mean().reset_index()
                    width = 0.8 / n_models
                    for i, mdl in enumerate(models_present):
                        sub = agg[agg["model"] == mdl].set_index("label").reindex(exp_labels)
                        bars = ax.bar(x_exp + i * width - (n_models - 1) * width / 2,
                                      sub[metric].fillna(0), width=width,
                                      label=model_labels[mdl], color=colors[i % len(colors)], edgecolor="white")
                        for bar, val in zip(bars, sub[metric].fillna(0)):
                            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
                elif n_datasets == 1:
                    sub = exp_df.groupby("label")[metric].mean().reindex(exp_labels)
                    bars = ax.bar(x_exp, sub.fillna(0), color="steelblue", edgecolor="white")
                    for bar, val in zip(bars, sub.fillna(0)):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
                else:
                    width = 0.8 / n_datasets
                    for i, ds in enumerate(datasets_present):
                        sub = exp_df[exp_df["dataset"] == ds].set_index("label").reindex(exp_labels)
                        bars = ax.bar(x_exp + i * width - (n_datasets - 1) * width / 2,
                                      sub[metric].fillna(0), width=width,
                                      label=ds, color=colors[i % len(colors)], edgecolor="white")
                        for bar, val in zip(bars, sub[metric].fillna(0)):
                            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)


            ax.set_xticks(np.arange(len(all_labels)))
            ax.set_xticklabels(all_labels)

            needs_legend = use_model_grouping or (n_datasets > 1 and not use_model_grouping) or n_paper > 0
            if needs_legend:
                legend_title = "Model" if use_model_grouping else ("Dataset" if n_datasets > 1 else "")
                ax.legend(title=legend_title)

            dataset_title = " + ".join(datasets_present) if datasets_present else "all"

            metric_label = {
                "rec_delib": "Recall", "prec_delib": "Precision",
                "f1_delib": "F1 (deliberative)", "f2_delib": "F2",
                "rec_not": "Recall (not deliberative)", "prec_not": "Precision (not deliberative)",
                "f1_not": "F1 (not deliberative)", "f2_not": "F2 (not deliberative)",
                "macro_f1": "Macro F1", "macro_f2": "Macro F2",
                "accuracy": "Accuracy", "mcc": "MCC",
            }.get(metric, metric)

            ax.set_ylim(0, 1.08)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by variant — dataset: {dataset_title}")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()

            if args.chart_out:
                fig.savefig(args.chart_out, dpi=150)
                print(f"\nChart saved to {Path(args.chart_out).resolve()}")
            else:
                plt.show()


if __name__ == "__main__":
    main()
