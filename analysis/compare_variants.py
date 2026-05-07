"""
Qualitative comparison of prediction variants on the same dataset (or a combined corpus).

For each pair of variants the script shows:
  - Which examples one variant gets right and the other gets wrong (exclusive wins)
  - Which examples both get wrong (shared mistakes)
  - Which examples both get right (shared correct)

A cross-variant agreement matrix is also printed.

When a multi-agent CSV is compared against a single-agent CSV from the same model,
--hypothesis runs an additional analysis testing the claim:
  "Hard cases cause agent2 to find conflicting reasons, agent3 follows agent2,
   but agent1 already had the correct prediction."

Usage examples
--------------
# Auto-discover all CSVs for dataset K1 and compare them
  python compare_variants.py --dataset K1

# Combine multiple datasets into one big corpus before comparing
  python compare_variants.py --dataset K1 K2 K3 R4

# Compare a hand-picked set of files
  python compare_variants.py --files gemini-simple_K1.csv gemini-fewshot_K1.csv

# Limit output to false-negatives only and show sentence text
  python compare_variants.py --dataset K1 --fn-only --text

# Only show examples with a specific ground-truth label (1 = deliberative)
  python compare_variants.py --dataset K1 --label 1

# Export all per-example judgements to a CSV
  python compare_variants.py --dataset K1 K2 --out comparison_combined.csv

# Compare only a specific pair of variants
  python compare_variants.py --dataset K1 --pair simple fewshot-cot-hard

# Test multi-agent failure hypothesis across all datasets combined
  python compare_variants.py --dataset K1 K2 K3 R4 --hypothesis --text
"""

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

# Files that are not experiment result files
SKIP_FILES = {"test.csv", "agentic.csv"}

FILENAME_RE = re.compile(r"^(?P<model>[^-]+)-(?P<variant>.+)_(?P<dataset>[^_]+)\.csv$")

LABEL_NAMES = {0: "not-deliberative", 1: "deliberative"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_filename(fname: str) -> dict:
    m = FILENAME_RE.match(fname)
    if m:
        return {"model": m.group("model"), "variant": m.group("variant"), "dataset": m.group("dataset")}
    stem = Path(fname).stem
    return {"model": stem, "variant": stem, "dataset": "-"}


def load_csv(path: Path, index_prefix: str = "") -> pd.DataFrame | None:
    """Load a result CSV, returning a frame with at least row/sentence/y_true/y_pred.

    If *index_prefix* is given (e.g. "K1"), each row index becomes "<prefix>_<row>"
    so that rows from different datasets can be concatenated without collisions.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] {path.name}: could not read ({e})", file=sys.stderr)
        return None

    for col in ("y_true", "y_pred"):
        if col not in df.columns:
            print(f"[SKIP] {path.name}: missing column '{col}'", file=sys.stderr)
            return None

    df = df.dropna(subset=["y_true", "y_pred"]).copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)

    if "row" not in df.columns:
        df["row"] = range(len(df))
    else:
        df["row"] = df["row"].astype(int)

    if "sentence" not in df.columns:
        df["sentence"] = ""

    if index_prefix:
        df["row"] = [f"{index_prefix}_{r}" for r in df["row"]]

    return df[["row", "sentence", "y_true", "y_pred"]].set_index("row")


def short_name(path: Path) -> str:
    """Human-readable variant label from filename."""
    meta = parse_filename(path.name)
    if meta["dataset"] == "-":
        return path.stem
    return f"{meta['model']}-{meta['variant']}"


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_combined(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all variant predictions into one wide frame indexed by row.
    Columns: sentence, y_true, <variant>_pred, <variant>_correct, ...
    """
    names = list(frames.keys())
    base = frames[names[0]][["sentence", "y_true"]].copy()

    for name, df in frames.items():
        base[f"{name}_pred"]    = df["y_pred"]
        base[f"{name}_correct"] = (df["y_true"] == df["y_pred"]).astype(int)

    return base.dropna()


def agreement_matrix(combined: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Pairwise agreement: fraction of rows where both variants agree."""
    mat = {}
    for a in names:
        mat[a] = {}
        for b in names:
            agree = (combined[f"{a}_pred"] == combined[f"{b}_pred"]).mean()
            mat[a][b] = round(agree, 3)
    return pd.DataFrame(mat, index=names, columns=names)


def exclusive_wins(combined: pd.DataFrame, a: str, b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two sub-frames:
      a_wins: rows where A is correct and B is wrong
      b_wins: rows where B is correct and A is wrong
    """
    a_wins = combined[(combined[f"{a}_correct"] == 1) & (combined[f"{b}_correct"] == 0)]
    b_wins = combined[(combined[f"{b}_correct"] == 1) & (combined[f"{a}_correct"] == 0)]
    return a_wins, b_wins


def shared_mistakes(combined: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Rows that ALL variants get wrong."""
    mask = pd.Series(True, index=combined.index)
    for name in names:
        mask &= combined[f"{name}_correct"] == 0
    return combined[mask]


def shared_correct(combined: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Rows that ALL variants get right."""
    mask = pd.Series(True, index=combined.index)
    for name in names:
        mask &= combined[f"{name}_correct"] == 1
    return combined[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _pred_summary(row, names: list[str]) -> str:
    parts = []
    for n in names:
        pred = int(row[f"{n}_pred"])
        ok   = "✓" if row[f"{n}_correct"] else "✗"
        parts.append(f"{n}={pred}{ok}")
    return "  ".join(parts)


def print_examples(df: pd.DataFrame, names: list[str], show_text: bool,
                   max_rows: int, label_filter: int | None):
    if label_filter is not None:
        df = df[df["y_true"] == label_filter]

    if df.empty:
        print("  (none)")
        return

    shown = df.head(max_rows)
    for row_id, row in shown.iterrows():
        truth_name = LABEL_NAMES.get(int(row["y_true"]), str(int(row["y_true"])))
        preds = _pred_summary(row, names)
        if show_text:
            sentence = str(row["sentence"])
            print(f"  row {row_id:>4}  truth={truth_name:20s}  {preds}")
            print(f"            \"{sentence[:120]}{'...' if len(sentence) > 120 else ''}\"")
        else:
            print(f"  row {row_id:>4}  truth={truth_name:20s}  {preds}")

    if len(df) > max_rows:
        print(f"  ... ({len(df) - max_rows} more rows not shown; use --max-rows to increase)")


def section(title: str, width: int = 72):
    print()
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent hypothesis analysis
# ─────────────────────────────────────────────────────────────────────────────

MULTI_AGENT_COLS = [
    "row", "sentence", "y_true",
    "agent1_pred",
    "agent2_assessment", "agent2_issues", "agent2_suggestion",
    "agent3_pred", "agent3_rationale",
    "y_pred",
]


def load_multi_csv(path: Path, index_prefix: str = "") -> pd.DataFrame | None:
    """
    Load a multi-agent result CSV, preserving all agent columns.
    Returns None if the file doesn't look like a multi-agent result.

    If *index_prefix* is given (e.g. "K1"), each row index becomes "<prefix>_<row>".
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] {path.name}: could not read ({e})", file=sys.stderr)
        return None

    required = {"y_true", "y_pred", "agent1_pred", "agent2_assessment", "agent2_suggestion", "agent3_pred"}
    if not required.issubset(df.columns):
        return None  # not a multi-agent file

    if "agent2_assessment" not in df.columns or df["agent2_assessment"].isna().all():
        return None  # single-agent file with empty agent2 columns

    df = df.dropna(subset=["y_true", "agent1_pred"]).copy()
    df["y_true"]            = df["y_true"].astype(int)
    df["agent1_pred"]       = pd.to_numeric(df["agent1_pred"], errors="coerce")
    df["agent2_suggestion"] = pd.to_numeric(df["agent2_suggestion"], errors="coerce")
    df["agent3_pred"]       = pd.to_numeric(df["agent3_pred"], errors="coerce")
    df["y_pred"]            = pd.to_numeric(df["y_pred"], errors="coerce")

    if "row" not in df.columns:
        df["row"] = range(len(df))
    else:
        df["row"] = df["row"].astype(int)

    if "sentence" not in df.columns:
        df["sentence"] = ""

    if index_prefix:
        df["row"] = [f"{index_prefix}_{r}" for r in df["row"]]

    return df.set_index("row")


def _find_multi_single_pair(
    paths: list[Path],
) -> tuple[Path, Path] | None:
    """
    Among the loaded paths, find the first (multi, single) pair that share the same model.
    Returns (multi_path, single_path) or None.
    """
    by_model: dict[str, dict[str, Path]] = {}
    for p in paths:
        meta = parse_filename(p.name)
        model   = meta["model"]
        variant = meta["variant"]
        by_model.setdefault(model, {})[variant] = p

    for model, variants in by_model.items():
        multi_keys  = [v for v in variants if v == "multi"]
        single_keys = [v for v in variants if v == "single"]
        if multi_keys and single_keys:
            return variants[multi_keys[0]], variants[single_keys[0]]

    return None


def _truncate(text: str, max_len: int = 140) -> str:
    text = str(text)
    return text[:max_len] + "..." if len(text) > max_len else text


def _print_hypothesis_examples(
    rows: pd.DataFrame,
    show_text: bool,
    max_rows: int,
    label_filter: int | None,
    multi_name: str,
    single_name: str,
):
    if label_filter is not None:
        rows = rows[rows["y_true"] == label_filter]

    if rows.empty:
        print("  (none)")
        return

    shown = rows.head(max_rows)
    for row_id, row in shown.iterrows():
        truth_name = LABEL_NAMES.get(int(row["y_true"]), str(int(row["y_true"])))
        a1   = int(row["agent1_pred"])
        a2s  = int(row["agent2_suggestion"])
        a3   = int(row["agent3_pred"])
        ypred = int(row["y_pred"]) if pd.notna(row["y_pred"]) else "?"
        print(
            f"  row {row_id:>4}  truth={truth_name:20s} "
            f"A1={a1}  A2_assessment={row['agent2_assessment']:6s}  A2_suggests={a2s}  "
            f"A3={a3}  final={ypred}"
        )
        if show_text:
            print(f"            sentence:   \"{_truncate(row['sentence'])}\"")
            if pd.notna(row.get("agent2_issues", "")):
                print(f"            A2 issues:  \"{_truncate(row['agent2_issues'])}\"")
            if pd.notna(row.get("agent3_rationale", "")):
                print(f"            A3 reason:  \"{_truncate(row['agent3_rationale'])}\"")

    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more rows not shown; use --max-rows to increase)")


def analyze_multi_agent_hypothesis(
    raw: pd.DataFrame,
    combined: pd.DataFrame,
    multi_name: str,
    single_name: str,
    show_text: bool,
    max_rows: int,
    label_filter: int | None,
    save_hard_cases: bool = False,
    dataset_label: str = "",
):
    """
    Test the hypothesis:
      "Hard cases cause agent2 to find conflicting reasons,
       agent3 follows agent2, but agent1 already had the correct prediction."

    'Hard cases' here are operationalised as rows where single (= agent1 only)
    is correct but multi is wrong.

    *raw* must be a pre-loaded multi-agent DataFrame (from load_multi_csv or a
    concatenation of several such DataFrames with prefixed indices).
    """
    # Align on shared index
    idx = combined.index.intersection(raw.index)
    raw = raw.loc[idx].copy()
    comb = combined.loc[idx].copy()

    n_total = len(idx)

    # Hard cases: single correct, multi wrong
    hard_mask = (comb[f"{single_name}_correct"] == 1) & (comb[f"{multi_name}_correct"] == 0)
    hard = raw[hard_mask].copy()
    n_hard = len(hard)

    # Soft wins: multi correct, single wrong  (for contrast)
    multi_wins_mask = (comb[f"{multi_name}_correct"] == 1) & (comb[f"{single_name}_correct"] == 0)
    multi_wins = raw[multi_wins_mask].copy()

    # ── Optionally save hard-case / multi-win subsets ────────────────────────
    if save_hard_cases:
        suffix = f"_{dataset_label}" if dataset_label else ""
        out_cols_base = ["sentence", "y_true", "y_pred",
                         "agent1_pred", "agent2_assessment", "agent2_issues",
                         "agent2_suggestion", "agent3_pred", "agent3_rationale"]

        def _save_subset(df, label, fname):
            if df.empty:
                return
            cols = [c for c in out_cols_base if c in df.columns]
            df.index.name = "row"
            df[cols].reset_index().to_csv(fname, index=False)
            print(f"  Saved {label}: {len(df)} rows → {fname}")

        # Single correct, multi wrong (TP→FN and TN→FP)
        if n_hard > 0:
            _save_subset(hard[hard["y_true"] == 1], "TP→FN (single TP, multi FN)", f"hard_TP-FN{suffix}.csv")
            _save_subset(hard[hard["y_true"] == 0], "TN→FP (single TN, multi FP)", f"hard_TN-FP{suffix}.csv")

        # Multi correct, single wrong (FN→TP and FP→TN)
        if len(multi_wins) > 0:
            _save_subset(multi_wins[multi_wins["y_true"] == 1], "FN→TP (single FN, multi TP)", f"hard_FN-TP{suffix}.csv")
            _save_subset(multi_wins[multi_wins["y_true"] == 0], "FP→TN (single FP, multi TN)", f"hard_FP-TN{suffix}.csv")

    section(
        f"Multi-agent failure hypothesis  —  {multi_name}  vs  {single_name}",
        width=72,
    )
    print(f"\n  Total rows: {n_total}")
    print(f"  Hard cases (single correct, multi wrong): {n_hard}  ({n_hard/n_total:.1%})")
    print(f"  Multi wins (multi correct, single wrong): {len(multi_wins)}  ({len(multi_wins)/n_total:.1%})")

    if n_hard == 0:
        print("\n  No hard cases found; hypothesis cannot be tested.")
        return

    # ── Among hard cases, classify failure mode ───────────────────────────────
    #
    # Pattern A  (the hypothesis):
    #   agent1 was right  AND  agent2 flips  AND  agent3 follows agent2
    #
    # Pattern B  (agent3 resists):
    #   agent1 was right  AND  agent2 flips  AND  agent3 goes back to agent1
    #   (but final still wrong — shouldn't happen if A3 agreed with A1; skip)
    #
    # Pattern C  (agent1 was already wrong):
    #   agent1 was wrong  (agent2 may or may not flip)
    #
    # Pattern D  (agent2 sound, agent3 wrong anyway):
    #   agent2 says 'sound' but final is still wrong  (agent3 caused the flip)

    a1_right  = hard["agent1_pred"] == hard["y_true"]
    a2_flips  = (hard["agent2_assessment"] == "flawed") & (hard["agent2_suggestion"] != hard["agent1_pred"])
    a3_follows_a2 = hard["agent3_pred"] == hard["agent2_suggestion"]
    a3_resists    = hard["agent3_pred"] == hard["agent1_pred"]
    a2_sound  = hard["agent2_assessment"] == "sound"

    pat_A = a1_right & a2_flips & a3_follows_a2   # hypothesis pattern
    pat_B = a1_right & a2_flips & a3_resists       # agent3 resists agent2
    pat_C = ~a1_right                              # agent1 already wrong
    pat_D = a1_right & a2_sound                    # agent2 doesn't flip but outcome wrong

    counts = {
        "A – A1 right, A2 flips, A3 follows A2 (hypothesis)": pat_A.sum(),
        "B – A1 right, A2 flips, A3 resists (reverts to A1)": pat_B.sum(),
        "C – A1 already wrong":                                pat_C.sum(),
        "D – A1 right, A2 sound, A3 flips anyway":            pat_D.sum(),
        "Other":  n_hard - pat_A.sum() - pat_B.sum() - pat_C.sum() - pat_D.sum(),
    }

    print(f"\n  Breakdown of hard cases by failure mode:")
    for label, cnt in counts.items():
        bar = "█" * int(30 * cnt / n_hard) if n_hard else ""
        print(f"    {label:<50s}  {cnt:>4d}  ({cnt/n_hard:.1%})  {bar}")

    # ── Show pattern A examples (the hypothesis) ──────────────────────────────
    pat_A_rows = hard[pat_A]
    by_err_type = {}
    for err, mask in [("FN (missed deliberative)", pat_A_rows["y_true"] == 1),
                      ("FP (false alarm)",          pat_A_rows["y_true"] == 0)]:
        by_err_type[err] = pat_A_rows[mask]

    print(f"\n  Pattern A — agent2 flips agent1's correct answer, agent3 follows")
    print(f"  ({pat_A.sum()} cases; {pat_A.sum()/n_hard:.1%} of hard cases)")
    for err_label, subset in by_err_type.items():
        print(f"\n    {err_label}: {len(subset)} cases")
        _print_hypothesis_examples(subset, show_text, max_rows, label_filter,
                                   multi_name, single_name)

    # ── Show pattern B examples (agent3 resists) ─────────────────────────────
    pat_B_rows = hard[pat_B]
    print(f"\n  Pattern B — agent2 tries to flip, but agent3 resists")
    print(f"  ({pat_B.sum()} cases; note: final still wrong here because A3≠A1 override)")
    _print_hypothesis_examples(pat_B_rows, show_text, max_rows, label_filter,
                                multi_name, single_name)

    # ── Contrast: multi_wins — when does agent2 correctly fix agent1? ─────────
    if len(multi_wins) > 0:
        mw_a1_wrong = multi_wins["agent1_pred"] != multi_wins["y_true"]
        mw_a2_flips = (multi_wins["agent2_assessment"] == "flawed") & \
                      (multi_wins["agent2_suggestion"] != multi_wins["agent1_pred"])
        mw_a3_follows = multi_wins["agent3_pred"] == multi_wins["agent2_suggestion"]

        # breakdown by fix pattern (mirror of hard-case patterns)
        mw_a1_right  = ~mw_a1_wrong
        mw_a2_sound  = multi_wins["agent2_assessment"] == "sound"
        mw_a3_resists = multi_wins["agent3_pred"] == multi_wins["agent1_pred"]

        mw_pat_A = mw_a1_wrong & mw_a2_flips & mw_a3_follows    # A2 correctly fixes A1
        mw_pat_B = mw_a1_wrong & mw_a2_flips & mw_a3_resists    # A2 tries to fix, A3 resists
        mw_pat_C = mw_a1_right & mw_a2_sound                    # A1 always right, A2 agrees
        mw_pat_D = mw_a1_right & mw_a2_flips & mw_a3_follows    # A2 wrongly doubts A1 but A3 still right?
        n_mw = len(multi_wins)

        section(
            f"Contrast: multi correct, single wrong  —  {multi_name}  vs  {single_name}",
            width=72,
        )
        print(f"\n  {n_mw} rows where multi is correct and single is wrong")
        print(f"\n  Breakdown by correction mode:")
        mw_counts = {
            "A – A1 wrong, A2 flips, A3 follows A2 (A2 saves the day)": mw_pat_A.sum(),
            "B – A1 wrong, A2 flips, A3 resists A2 (A3 saves the day)": mw_pat_B.sum(),
            "C – A1 right, A2 sound (A1 never needed help)":             mw_pat_C.sum(),
            "D – A1 right, A2 wrongly doubts, A3 follows A2 (yet right)": mw_pat_D.sum(),
            "Other": n_mw - mw_pat_A.sum() - mw_pat_B.sum() - mw_pat_C.sum() - mw_pat_D.sum(),
        }
        for label, cnt in mw_counts.items():
            bar = "█" * int(30 * cnt / n_mw) if n_mw else ""
            print(f"    {label:<55s}  {cnt:>4d}  ({cnt/n_mw:.1%})  {bar}")

        # Show pattern A (A2 correctly catches A1's mistake)
        mw_pat_A_rows = multi_wins[mw_pat_A]
        for err, mask in [("FN fixed (was missing deliberative)", mw_pat_A_rows["y_true"] == 1),
                          ("FP fixed (was false alarm)",          mw_pat_A_rows["y_true"] == 0)]:
            subset = mw_pat_A_rows[mask]
            print(f"\n  Pattern A — {err}: {len(subset)} cases")
            _print_hypothesis_examples(subset, show_text, max_rows, label_filter,
                                       multi_name, single_name)

        # Show pattern C (A1 always right, early exit, A2 never needed)
        mw_pat_C_rows = multi_wins[mw_pat_C]
        for err, mask in [("deliberative", mw_pat_C_rows["y_true"] == 1),
                          ("not-deliberative", mw_pat_C_rows["y_true"] == 0)]:
            subset = mw_pat_C_rows[mask]
            if len(subset) == 0:
                continue
            print(f"\n  Pattern C — A1 already right, A2 confirms ({err}): {len(subset)} cases")
            _print_hypothesis_examples(subset, show_text, max_rows, label_filter,
                                       multi_name, single_name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Hypothesis support summary:")
    hyp_pct = pat_A.sum() / n_hard if n_hard else 0
    resist_pct = pat_B.sum() / n_hard if n_hard else 0
    print(f"    {pat_A.sum()}/{n_hard} ({hyp_pct:.1%}) hard-case errors match the hypothesis "
          f"(A1 right → A2 misleads → A3 follows).")
    print(f"    {pat_B.sum()}/{n_hard} ({resist_pct:.1%}) cases show agent3 resisting agent2's flip.")
    if hyp_pct >= 0.5:
        verdict = "SUPPORTED"
    elif hyp_pct >= 0.25:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    print(f"    Verdict: {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative comparison of FOIA classification variants on the same dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", "-d", nargs="+", metavar="KEY",
                     help="Auto-discover CSVs matching one or more dataset keys (e.g. K1, or K1 K2 K3 R4). "
                          "When multiple keys are given, files with the same model/variant are concatenated "
                          "into a single corpus before analysis.")
    src.add_argument("--files", "-f", nargs="+", metavar="CSV",
                     help="Explicit list of CSV files to compare")

    parser.add_argument("--dir", default=".", metavar="PATH",
                        help="Directory to scan when using --dataset (default: .)")
    parser.add_argument("--model", metavar="NAME",
                        help="Filter auto-discovered files to a single model (e.g. gemini)")
    parser.add_argument("--pair", nargs=2, metavar=("VARIANT_A", "VARIANT_B"),
                        help="Restrict pair analysis to these two variant names "
                             "(matched against the short name, e.g. simple fewshot-cot-hard)")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None,
                        help="Restrict example display to a specific ground-truth label "
                             "(0 = not-deliberative, 1 = deliberative)")
    parser.add_argument("--fn-only", action="store_true",
                        help="Shorthand for --label 1: focus only on deliberative sentences "
                             "(false negatives are missed deliberative cases)")
    parser.add_argument("--text", "-t", action="store_true",
                        help="Print the sentence text alongside each example")
    parser.add_argument("--max-rows", type=int, default=20, metavar="N",
                        help="Max example rows to print per section (default: 20)")
    parser.add_argument("--no-pairwise", action="store_true",
                        help="Skip the detailed pairwise breakdown, print only global sections")
    parser.add_argument("--hypothesis", action="store_true",
                        help="Test the multi-agent failure hypothesis: agent2 finds conflicting "
                             "reasons on hard cases, agent3 follows agent2, but agent1 already "
                             "had the correct prediction. Requires at least one multi + one single "
                             "CSV from the same model to be present.")
    parser.add_argument("--save-hard-cases", action="store_true",
                        help="When --hypothesis is active, save the FN and FP subsets of "
                             "'single correct, multi wrong' rows to hard_FN_<dataset>.csv "
                             "and hard_FP_<dataset>.csv")
    parser.add_argument("--out", metavar="CSV",
                        help="Write the combined per-example judgement table to this CSV")
    args = parser.parse_args()

    label_filter = 1 if args.fn_only else args.label

    # ── Collect files ─────────────────────────────────────────────────────────
    # paths_by_variant maps variant_key -> list of (Path, index_prefix)
    # index_prefix is the dataset tag used to make row indices unique across
    # datasets; it is empty when files are specified explicitly.
    paths_by_variant: dict[str, list[tuple[Path, str]]] = {}

    if args.files:
        for p_str in args.files:
            p = Path(p_str)
            name = short_name(p)
            paths_by_variant.setdefault(name, []).append((p, ""))
        dataset_label = "custom"
    else:
        scan_dir = Path(args.dir)
        dataset_keys = [d.upper() for d in args.dataset]
        for ds_key in dataset_keys:
            ds_paths = sorted(
                p for p in scan_dir.glob("*.csv")
                if p.name not in SKIP_FILES
                and parse_filename(p.name)["dataset"].upper() == ds_key
                and (args.model is None
                     or parse_filename(p.name)["model"].lower() == args.model.lower())
            )
            for p in ds_paths:
                meta = parse_filename(p.name)
                variant_key = f"{meta['model']}-{meta['variant']}"
                # Use dataset key as prefix only when combining multiple datasets
                prefix = ds_key if len(dataset_keys) > 1 else ""
                paths_by_variant.setdefault(variant_key, []).append((p, prefix))
        dataset_label = "+".join(dataset_keys)

    if not paths_by_variant:
        print("No matching CSV files found.", file=sys.stderr)
        sys.exit(1)

    # ── Load and merge ────────────────────────────────────────────────────────
    # Files with the same variant_key are concatenated into one frame.
    # When index_prefix is set, row indices become e.g. "K1_42" to avoid
    # collisions when rows from different datasets are joined.
    frames: dict[str, pd.DataFrame] = {}
    for variant_key, path_ds_list in paths_by_variant.items():
        dfs = []
        for p, prefix in path_ds_list:
            df = load_csv(p, index_prefix=prefix)
            if df is not None:
                dfs.append(df)
        if dfs:
            merged = pd.concat(dfs)
            if variant_key in frames:
                variant_key = path_ds_list[0][0].stem  # fallback on collision
            frames[variant_key] = merged

    if len(frames) < 2:
        print(f"Need at least 2 loadable files; only got {len(frames)}.", file=sys.stderr)
        sys.exit(1)

    names = list(frames.keys())
    combined = build_combined(frames)

    n_rows = len(combined)

    print(f"\nVariant comparison — dataset: {dataset_label}  ({n_rows} examples)")
    print(f"Variants ({len(names)}):")
    for n in names:
        correct = combined[f"{n}_correct"].sum()
        print(f"  {n:50s}  accuracy={correct/n_rows:.3f}  ({correct}/{n_rows} correct)")

    # ── Agreement matrix ──────────────────────────────────────────────────────
    section("Pairwise prediction agreement (fraction of rows with identical prediction)")
    mat = agreement_matrix(combined, names)
    print(mat.to_string())

    # ── All-correct / all-wrong ───────────────────────────────────────────────
    s_correct = shared_correct(combined, names)
    s_wrong   = shared_mistakes(combined, names)

    section(f"Examples ALL variants get RIGHT  ({len(s_correct)}/{n_rows})")
    print_examples(s_correct, names, args.text, args.max_rows, label_filter)

    section(f"Examples ALL variants get WRONG  ({len(s_wrong)}/{n_rows})")
    print_examples(s_wrong, names, args.text, args.max_rows, label_filter)

    # ── Exclusive wins per variant ────────────────────────────────────────────
    # Count how many examples each variant gets right that NO other variant gets right
    section("Examples only ONE variant gets right (exclusive wins)")
    for name in names:
        others_wrong = pd.Series(True, index=combined.index)
        for other in names:
            if other != name:
                others_wrong &= combined[f"{other}_correct"] == 0
        exclusive = combined[(combined[f"{name}_correct"] == 1) & others_wrong]
        label_str = ""
        if label_filter is not None:
            n_label = (exclusive["y_true"] == label_filter).sum()
            label_str = f"  (label={label_filter}: {n_label})"
        print(f"\n  {name}: {len(exclusive)} exclusive wins{label_str}")
        if len(exclusive) > 0:
            print_examples(exclusive, names, args.text, args.max_rows, label_filter)

    # ── Pairwise breakdown ────────────────────────────────────────────────────
    if not args.no_pairwise:
        pairs = list(combinations(names, 2))
        if args.pair:
            # Filter to the requested pair (partial match on short name)
            pa, pb = args.pair
            pairs = [
                (a, b) for a, b in pairs
                if (pa.lower() in a.lower() and pb.lower() in b.lower())
                or (pb.lower() in a.lower() and pa.lower() in b.lower())
            ]
            if not pairs:
                print(f"\n[WARN] --pair {args.pair} matched no variant pairs; "
                      "check the variant names printed above.", file=sys.stderr)

        for a, b in pairs:
            a_wins, b_wins = exclusive_wins(combined, a, b)

            # filter to label if requested (for counting)
            def label_count(df):
                return (df["y_true"] == label_filter).sum() if label_filter is not None else len(df)

            section(f"Pairwise: {a}  vs  {b}")

            print(f"\n  {a} correct, {b} wrong  ({len(a_wins)} rows"
                  + (f", label={label_filter}: {label_count(a_wins)}" if label_filter is not None else "")
                  + ")")
            print_examples(a_wins, [a, b], args.text, args.max_rows, label_filter)

            print(f"\n  {b} correct, {a} wrong  ({len(b_wins)} rows"
                  + (f", label={label_filter}: {label_count(b_wins)}" if label_filter is not None else "")
                  + ")")
            print_examples(b_wins, [a, b], args.text, args.max_rows, label_filter)

            both_wrong = combined[
                (combined[f"{a}_correct"] == 0) & (combined[f"{b}_correct"] == 0)
            ]
            print(f"\n  Both wrong  ({len(both_wrong)} rows"
                  + (f", label={label_filter}: {label_count(both_wrong)}" if label_filter is not None else "")
                  + ")")
            print_examples(both_wrong, [a, b], args.text, args.max_rows, label_filter)

    # ── Error-type breakdown per variant ─────────────────────────────────────
    section("Error breakdown per variant")
    header = f"  {'Variant':<50}  {'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in names:
        sub = combined[["y_true", f"{name}_pred"]].copy()
        sub.columns = ["y_true", "y_pred"]
        tp = int(((sub["y_true"] == 1) & (sub["y_pred"] == 1)).sum())
        tn = int(((sub["y_true"] == 0) & (sub["y_pred"] == 0)).sum())
        fp = int(((sub["y_true"] == 0) & (sub["y_pred"] == 1)).sum())
        fn = int(((sub["y_true"] == 1) & (sub["y_pred"] == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {name:<50}  {tp:>5}  {tn:>5}  {fp:>5}  {fn:>5}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")

    # ── Confusion-type overlap between all variant pairs ─────────────────────
    section("Shared false-negatives (FN) and false-positives (FP) between all variant pairs")
    for a, b in combinations(names, 2):
        a_fn = set(combined[(combined["y_true"] == 1) & (combined[f"{a}_pred"] == 0)].index)
        b_fn = set(combined[(combined["y_true"] == 1) & (combined[f"{b}_pred"] == 0)].index)
        shared_fn = len(a_fn & b_fn)
        only_a_fn = len(a_fn - b_fn)
        only_b_fn = len(b_fn - a_fn)

        a_fp = set(combined[(combined["y_true"] == 0) & (combined[f"{a}_pred"] == 1)].index)
        b_fp = set(combined[(combined["y_true"] == 0) & (combined[f"{b}_pred"] == 1)].index)
        shared_fp = len(a_fp & b_fp)
        only_a_fp = len(a_fp - b_fp)
        only_b_fp = len(b_fp - a_fp)

        print(f"\n  {a}  ↔  {b}")
        print(f"    FN: shared={shared_fn}  only-{a}={only_a_fn}  only-{b}={only_b_fn}")
        print(f"    FP: shared={shared_fp}  only-{a}={only_a_fp}  only-{b}={only_b_fp}")

    # ── Multi-agent failure hypothesis ───────────────────────────────────────
    if args.hypothesis:
        # Find all (multi, single) variant pairs by model name.
        # For each such pair, load and concatenate all their multi-agent CSVs
        # (with the same index prefixing used above) and run the hypothesis test.
        from collections import defaultdict
        multi_paths_by_model: dict[str, list[tuple[Path, str]]] = defaultdict(list)
        single_paths_by_model: dict[str, list[tuple[Path, str]]] = defaultdict(list)

        for variant_key, path_ds_list in paths_by_variant.items():
            # variant_key is "model-variant" or a stem fallback
            meta = parse_filename(path_ds_list[0][0].name)
            model   = meta["model"]
            variant = meta["variant"]
            if variant == "multi":
                multi_paths_by_model[model].extend(path_ds_list)
            elif variant == "single":
                single_paths_by_model[model].extend(path_ds_list)

        paired_models = set(multi_paths_by_model) & set(single_paths_by_model)
        if not paired_models:
            print(
                "\n[WARN] --hypothesis: could not find a (multi, single) pair from the "
                "same model among the loaded files. Provide e.g. qwen-multi_K2.csv and "
                "qwen-single_K2.csv together.",
                file=sys.stderr,
            )
        for model in sorted(paired_models):
            multi_name  = f"{model}-multi"
            single_name = f"{model}-single"
            if multi_name not in names or single_name not in names:
                print(
                    f"\n[WARN] --hypothesis: resolved pair ({multi_name}, {single_name}) "
                    "but one or both are not in the loaded frames.",
                    file=sys.stderr,
                )
                continue

            # Load and concatenate all multi-agent CSVs for this model
            raw_dfs = []
            for p, prefix in multi_paths_by_model[model]:
                df = load_multi_csv(p, index_prefix=prefix)
                if df is not None:
                    raw_dfs.append(df)
            if not raw_dfs:
                print(
                    f"\n[WARN] --hypothesis: no valid multi-agent CSVs found for model '{model}'.",
                    file=sys.stderr,
                )
                continue
            raw_multi = pd.concat(raw_dfs)

            analyze_multi_agent_hypothesis(
                raw=raw_multi,
                combined=combined,
                multi_name=multi_name,
                single_name=single_name,
                show_text=args.text,
                max_rows=args.max_rows,
                label_filter=label_filter,
                save_hard_cases=args.save_hard_cases,
                dataset_label=dataset_label,
            )

    # ── Optional CSV export ───────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        combined.to_csv(out_path)
        print(f"\nCombined judgement table written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
