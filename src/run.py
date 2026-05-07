"""
FOIA Sentence Classification
=============================
Classifies government sentences as always deliberative or not under FOIA Exemption 5.
Supports single-agent and multi-agent (three-agent) pipelines, with optional CoT reasoning
and few-shot prompting.

True label: <Always deliberative> column (1 = always deliberative, blank = not).
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Optional
import dotenv

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

dotenv.load_dotenv()


# ── Data ──────────────────────────────────────────────────────────────────────

FILE_REGISTRY: dict[str, str] = {
    "K1": "foia_K1-Final.txt",
    "K2": "foia_K2--Final.txt",
    "K3": "foia_K3-Final.txt",
    "K5": "foia_K5-Final.txt",
    "R4": "foia_R4-Final.txt",
}

COLUMNS = ["label", "sentence", "always_deliberative", "non_iia", "sentiment", "subjectivity"]


def load_data(data_dir: str = ".", file_keys: Optional[list[str]] = None) -> pd.DataFrame:
    """Load data files by key (e.g. ['K1', 'R4']). Loads all files if file_keys is None."""
    keys = file_keys or list(FILE_REGISTRY.keys())
    unknown = [k for k in keys if k not in FILE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown file keys: {unknown}. Valid keys: {list(FILE_REGISTRY)}")

    frames = []
    for key in keys:
        fname = FILE_REGISTRY[key]
        path = Path(data_dir) / fname
        df = pd.read_csv(
            path, sep="|", names=COLUMNS, header=0,
            dtype=str, keep_default_na=False,
        )
        df["source_file"] = fname
        df["file_key"] = key
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["sentence"].str.strip() != ""].copy()
    df["y_true"] = (df["always_deliberative"].str.strip() == "1").astype(int)
    return df.reset_index(drop=True)


# ── Prompts ───────────────────────────────────────────────────────────────────

from prompts import (
    FOIA_BACKGROUND,
    AGENT1_SYSTEM_SIMPLE,
    AGENT1_SYSTEM_COT,
    AGENT1_SYSTEM_COT_LABELED,
    AGENT1_COT_LABELED_USER_TEMPLATE,
    AGENT2_SYSTEM,
    AGENT2_SYSTEM_SIMPLE,
    AGENT3_SYSTEM,
    AGENT3_SYSTEM_SIMPLE,
    USER_TEMPLATE,
    AGENT2_USER_TEMPLATE,
    AGENT2_USER_TEMPLATE_SIMPLE,
    AGENT3_USER_TEMPLATE,
    AGENT3_USER_TEMPLATE_SIMPLE,
)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    text = text.split("```json")[-1]
    text = text.replace("```json", "").replace("```", "")
    return json.loads(text.strip())


# ── Few-shot helpers ──────────────────────────────────────────────────────────

def sample_few_shot_examples(
    df: pd.DataFrame,
    n: int,
    exclude_source_file: Optional[str] = None,
    exclude_sentences: Optional[set[str]] = None,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Return n balanced (sentence, label) pairs, excluding rows from exclude_source_file."""
    pool = df if exclude_source_file is None else df[df["source_file"] != exclude_source_file]
    if exclude_sentences:
        pool = pool[~pool["sentence"].str.strip().isin(exclude_sentences)]
    n_pos = (n + 1) // 2
    n_neg = n // 2
    pos = pool[pool["y_true"] == 1].sample(n=min(n_pos, (pool["y_true"] == 1).sum()), random_state=seed)
    neg = pool[pool["y_true"] == 0].sample(n=min(n_neg, (pool["y_true"] == 0).sum()), random_state=seed)
    examples = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    return [(r["sentence"].strip(), int(r["y_true"])) for _, r in examples.iterrows()]


def load_hard_examples(csv_paths: list[str]) -> pd.DataFrame:
    """Load wrong predictions from results CSVs as hard few-shot candidates.

    A hard example is one where the model predicted incorrectly (y_pred != y_true).
    The correct label (y_true) is kept so it can be shown in the prompt.
    """
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p, dtype=str)
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        wrong = df.dropna(subset=["y_true", "y_pred"])
        wrong = wrong[wrong["y_pred"] != wrong["y_true"]].copy()
        frames.append(wrong[["sentence", "y_true"]])
    if not frames:
        return pd.DataFrame(columns=["sentence", "y_true"])
    combined = pd.concat(frames, ignore_index=True)
    combined["y_true"] = combined["y_true"].astype(int)
    return combined.drop_duplicates("sentence").reset_index(drop=True)


def sample_hard_few_shot_examples(
    hard_df: pd.DataFrame,
    n: int,
    exclude_sentences: Optional[set[str]] = None,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Sample n balanced hard (wrong-prediction) examples, excluding given sentences."""
    pool = hard_df
    if exclude_sentences:
        pool = pool[~pool["sentence"].str.strip().isin(exclude_sentences)]
    n_pos = (n + 1) // 2
    n_neg = n // 2
    pos = pool[pool["y_true"] == 1]
    neg = pool[pool["y_true"] == 0]
    pos_sample = pos.sample(n=min(n_pos, len(pos)), random_state=seed)
    neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=seed)
    examples = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed)
    return [(r["sentence"].strip(), int(r["y_true"])) for _, r in examples.iterrows()]


FewShotExample = tuple[str, int, str, str]  # (sentence, label, step1, step2)


def generate_cot_for_examples(
    examples: list[tuple[str, int]],
    backend: "LLMBackend",
) -> list[FewShotExample]:
    """Generate label-consistent CoT reasoning for each few-shot example.

    Uses a label-conditioned prompt so the model writes step1/step2 that
    justify the known ground-truth label rather than freely re-classifying.
    """
    label_word = {0: "NOT ALWAYS DELIBERATIVE", 1: "ALWAYS DELIBERATIVE"}
    enriched: list[FewShotExample] = []
    for sent, label in examples:
        try:
            user_text = AGENT1_COT_LABELED_USER_TEMPLATE.format(
                sentence=sent, label=label, label_word=label_word[label]
            )
            text, _ = backend.call(AGENT1_SYSTEM_COT_LABELED, [("user", user_text)])
            result = _parse_json(text)
            step1 = result.get("step1", "")
            step2 = result.get("step2", "")
        except Exception as e:
            print(f"  [CoT pre-gen failed for few-shot example, using fallback: {e}]")
            step1 = step2 = ""
        enriched.append((sent, label, step1, step2))
    return enriched


def format_few_shot_agent1_response(label: int, step1: str, step2: str, use_cot: bool = True) -> str:
    if use_cot:
        return json.dumps({"step1": step1, "step2": step2, "deliberative": label})
    return json.dumps({"deliberative": label})


# ── Backend base & registry ───────────────────────────────────────────────────

BACKEND_REGISTRY: dict[str, "LLMBackend"] = {}


class LLMBackend:
    """Base class. Each subclass implements _call_api(system, contents) -> (text, usage).
    usage = {"input_tokens": int, "output_tokens": int}
    """

    def call(self, system: str, contents: list[tuple[str, str]]) -> tuple[str, dict]:
        """Returns (response_text, usage_dict)."""
        return self._call_api(system, contents)

    def _call_api(self, system: str, contents: list[tuple[str, str]]) -> tuple[str, dict]:
        raise NotImplementedError


class GeminiBackend(LLMBackend):
    def __init__(self, model: str):
        self.model = model

    def _call_api(self, system: str, contents: list[tuple[str, str]]) -> tuple[str, dict]:
        from google import genai
        from google.genai.types import GenerateContentConfig, Content, Part

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set.")

        client = genai.Client(api_key=api_key)
        gc_contents = [Content(role=role, parts=[Part(text=text)]) for role, text in contents]
        response = client.models.generate_content(
            model=self.model,
            contents=gc_contents,
            config=GenerateContentConfig(system_instruction=system, temperature=0),
        )
        usage = {
            "input_tokens":  response.usage_metadata.prompt_token_count or 0,
            "output_tokens": response.usage_metadata.candidates_token_count or 0,
        }
        return response.text, usage


# ── Qwen local backend (transformers) ────────────────────────────────────────

class QwenBackend(LLMBackend):
    """Runs a Qwen model locally via transformers. Model is loaded on first call."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model = model_name   # kept as `model` attr for cost/usage reporting
        self._model = None
        self._tokenizer = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {self.model} …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, torch_dtype="auto", device_map="auto"
        )

    def _call_api(self, system: str, contents: list[tuple[str, str]]) -> tuple[str, dict]:
        if self._model is None:
            self._load()

        role_map = {"user": "user", "model": "assistant"}
        messages = [{"role": "system", "content": system}]
        messages += [{"role": role_map[role], "content": text} for role, text in contents]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        model_inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        input_len = model_inputs.input_ids.shape[1]

        generated_ids = self._model.generate(**model_inputs, max_new_tokens=1024, do_sample=False, temperature=0.0, top_p=0.8, top_k=20)
        new_ids = generated_ids[0][input_len:]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        usage = {"input_tokens": input_len, "output_tokens": len(new_ids)}
        return response, usage


BACKEND_REGISTRY.update({
    "gemini-flash":      GeminiBackend("gemini-2.5-flash"),
    "qwen3.5-9b":        QwenBackend("Qwen/Qwen3.5-9B")
})

# ── Pricing ───────────────────────────────────────────────────────────────────
# (input $/1M tokens, output $/1M tokens) — update as rates change
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash":      (0.30,   2.50),
    # Qwen models are free to run locally
}


def compute_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    if model_id not in MODEL_PRICING:
        return 0.0
    in_price, out_price = MODEL_PRICING[model_id]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


# ── Three-agent pipeline ──────────────────────────────────────────────────────

def run_agent1(
    sentence: str,
    backend: LLMBackend,
    few_shot_examples: Optional[list[FewShotExample]] = None,
    use_cot: bool = False
) -> dict:
    """Predictor: prediction with optional CoT reasoning (--cot)."""
    system = AGENT1_SYSTEM_COT if use_cot else AGENT1_SYSTEM_SIMPLE
    contents = []
    for ex_sent, ex_label, ex_step1, ex_step2 in (few_shot_examples or []):
        contents.append(("user",  USER_TEMPLATE.format(sentence=ex_sent)))
        contents.append(("model", format_few_shot_agent1_response(ex_label, ex_step1, ex_step2, use_cot=use_cot)))
    user_text = USER_TEMPLATE.format(sentence=sentence.replace('"', "'"))
    contents.append(("user", user_text))

    text, usage = backend.call(system, contents)
    print(text)
    result = _parse_json(text)
    deliberative = result.get("deliberative") or result.get("delibrative", 0) or result.get("deliterative", 0)  # catch common misspelling
    return {
        "pred":  int(deliberative),
        "step1": result.get("step1", ""),
        "step2": result.get("step2", ""),
        "_usage": usage,
    }


def run_agent2(sentence: str, agent1: dict, backend: LLMBackend, use_cot: bool = True) -> dict:
    """Critic: reviews Agent 1's reasoning (CoT) or gives a second opinion (no CoT)."""
    label_str = {0: "NOT-DELIB", 1: "DELIBERATIVE"}
    if use_cot:
        system = AGENT2_SYSTEM
        user_text = AGENT2_USER_TEMPLATE.format(
            sentence=sentence,
            step1=agent1["step1"],
            step2=agent1["step2"],
            pred1=agent1["pred"],
            pred1_label=label_str[agent1["pred"]],
        )
    else:
        system = AGENT2_SYSTEM_SIMPLE
        user_text = AGENT2_USER_TEMPLATE_SIMPLE.format(
            sentence=sentence,
            pred1=agent1["pred"],
            pred1_label=label_str[agent1["pred"]],
        )
    text, usage = backend.call(system, [("user", user_text)])
    print(text)
    result = _parse_json(text)
    return {
        "assessment": result.get("assessment", ""),
        "issues":     result.get("issues", ""),
        "suggestion": int(result.get("suggestion", agent1["pred"])),
        "_usage": usage,
    }


def run_agent3(sentence: str, agent1: dict, agent2: dict, backend: LLMBackend, use_cot: bool = True) -> dict:
    """Judge: final verdict."""
    label_str = {0: "NOT-DELIB", 1: "DELIBERATIVE"}
    if use_cot:
        system = AGENT3_SYSTEM
        user_text = AGENT3_USER_TEMPLATE.format(
            sentence=sentence,
            step1=agent1["step1"],
            step2=agent1["step2"],
            pred1=agent1["pred"],
            pred1_label=label_str[agent1["pred"]],
            assessment=agent2["assessment"],
            issues=agent2["issues"],
            pred2=agent2["suggestion"],
            pred2_label=label_str[agent2["suggestion"]],
        )
    else:
        system = AGENT3_SYSTEM_SIMPLE
        user_text = AGENT3_USER_TEMPLATE_SIMPLE.format(
            sentence=sentence,
            pred1=agent1["pred"],
            pred1_label=label_str[agent1["pred"]],
            pred2=agent2["suggestion"],
            pred2_label=label_str[agent2["suggestion"]],
        )
    text, usage = backend.call(system, [("user", user_text)])
    result = _parse_json(text)
    return {
        "pred":      int(result["deliberative"]),
        "rationale": result.get("rationale", ""),
        "_usage": usage,
    }


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    df: pd.DataFrame,
    agent1_model: str,
    agent2_model: str,
    agent3_model: str,
    output_path: Optional[str] = None,
    delay: float = 0.2,
    max_rows: Optional[int] = None,
    n_few_shot: int = 0,
    few_shot_pool: Optional[pd.DataFrame] = None,
    hard_examples_df: Optional[pd.DataFrame] = None,
    early_exit: bool = True,
    single_agent: bool = False,
    use_cot: bool = False,
) -> dict:
    b1 = BACKEND_REGISTRY[agent1_model]
    b2 = BACKEND_REGISTRY[agent2_model] if not single_agent else None
    b3 = BACKEND_REGISTRY[agent3_model] if not single_agent else None

    subset = df if max_rows is None else df.head(max_rows)
    pool   = few_shot_pool if few_shot_pool is not None else df

    # ── Checkpoint: skip rows already saved ───────────────────────────────────
    checkpoint = Path(output_path) if output_path else None
    completed_indices: set[int] = set()
    checkpoint_data: dict[int, dict] = {}  # row_idx -> record (for in-place updates)
    if checkpoint and checkpoint.exists():
        done = pd.read_csv(checkpoint)
        for _, r in done.iterrows():
            checkpoint_data[int(r["row"])] = r.to_dict()
        valid_mask = done["y_pred"].isin([0, 1])
        completed_indices = set(done.loc[valid_mask, "row"].astype(int).tolist())
        n_invalid = int((~valid_mask).sum())
        print(
            f"Resuming from checkpoint: {len(completed_indices)} rows done"
            + (f", {n_invalid} invalid y_pred rows will be retried." if n_invalid else ".")
            + "\n"
        )

    label_str = {0: "NOT-DELIB", 1: "DELIBERATIVE"}
    records = []
    errors  = []
    n_early_exit = 0
    current_source_file = None
    few_shot_examples: Optional[list[FewShotExample]] = None

    # Token usage accumulators per agent
    usage: dict[str, dict] = {
        "agent1": {"model": BACKEND_REGISTRY[agent1_model].model if hasattr(BACKEND_REGISTRY[agent1_model], "model") else agent1_model, "input_tokens": 0, "output_tokens": 0},
        "agent2": {"model": BACKEND_REGISTRY[agent2_model].model if hasattr(BACKEND_REGISTRY[agent2_model], "model") else agent2_model, "input_tokens": 0, "output_tokens": 0},
        "agent3": {"model": BACKEND_REGISTRY[agent3_model].model if hasattr(BACKEND_REGISTRY[agent3_model], "model") else agent3_model, "input_tokens": 0, "output_tokens": 0},
    }

    for i, row in subset.iterrows():
        if i in completed_indices:
            continue

        sentence  = row["sentence"].strip()
        y_true_i  = int(row["y_true"])

        if n_few_shot > 0 and row["source_file"] != current_source_file:
            current_source_file = row["source_file"]
            # Sentences belonging to the current source file — excluded from hard examples
            source_sentences = set(pool[pool["source_file"] == current_source_file]["sentence"].str.strip())

            if hard_examples_df is not None and not hard_examples_df.empty:
                # Fill as many slots as possible with hard (wrong-prediction) examples,
                # then top up with regular balanced examples if needed.
                hard_raw = sample_hard_few_shot_examples(
                    hard_examples_df, n=n_few_shot, exclude_sentences=source_sentences,
                )
                n_remaining = n_few_shot - len(hard_raw)
                hard_sentences = {s for s, _ in hard_raw}
                if n_remaining > 0:
                    regular_raw = sample_few_shot_examples(
                        pool, n=n_remaining,
                        exclude_source_file=current_source_file,
                        exclude_sentences=hard_sentences,
                    )
                else:
                    regular_raw = []
                raw_examples = hard_raw + regular_raw
                n_hard_used = len(hard_raw)
            else:
                raw_examples = sample_few_shot_examples(
                    pool, n=n_few_shot, exclude_source_file=current_source_file,
                )
                n_hard_used = 0

            if use_cot:
                print(f"\n--- Generating CoT reasoning for few-shot examples ({current_source_file}) ---")
                few_shot_examples = generate_cot_for_examples(raw_examples, b1)
            else:
                few_shot_examples = [(s, l, "", "") for s, l in raw_examples]
            hard_sentences_set = {s for s, _ in (hard_raw if hard_examples_df is not None else [])}
            print(f"\n--- Few-shot examples for {current_source_file}"
                  + (f" ({n_hard_used} hard)" if n_hard_used else "") + " ---")
            for ex_sent, ex_label, ex_step1, _ in few_shot_examples:
                tag = " [HARD]" if ex_sent in hard_sentences_set else ""
                step_preview = f"  step1: {ex_step1[:60]}" if ex_step1 else ""
                print(f"  [{label_str[ex_label]:<13}]{tag}  {ex_sent[:90]}{step_preview}")
            print()

        print(f"[{i:>4}] {sentence[:90]}")
        try:
            a1 = run_agent1(sentence, b1, few_shot_examples, use_cot=use_cot)
            usage["agent1"]["input_tokens"]  += a1["_usage"]["input_tokens"]
            usage["agent1"]["output_tokens"] += a1["_usage"]["output_tokens"]
            match = "✓" if a1["pred"] == y_true_i else "✗"
            print(f"       Agent1  pred={label_str[a1['pred']]:<13}"
                  f"  step1: {a1['step1'][:60]}"
                  + (f"  true={label_str[y_true_i]}  {match}" if single_agent else ""))

            if single_agent:
                record = {
                    "row": i,
                    "sentence": sentence,
                    "y_true": y_true_i,
                    "agent1_pred": a1["pred"],
                    "agent1_step1": a1["step1"],
                    "agent1_step2": a1["step2"],
                    "agent2_assessment": None,
                    "agent2_issues": None,
                    "agent2_suggestion": None,
                    "agent3_pred": None,
                    "agent3_rationale": None,
                    "y_pred": a1["pred"],
                }
            else:
                a2 = run_agent2(sentence, a1, b2, use_cot=use_cot)
                usage["agent2"]["input_tokens"]  += a2["_usage"]["input_tokens"]
                usage["agent2"]["output_tokens"] += a2["_usage"]["output_tokens"]
                print(f"       Agent2  assessment={a2['assessment']:<8}"
                      f"  suggestion={label_str[a2['suggestion']]:<13}"
                      f"  issues: {a2['issues'][:50]}")

                if early_exit and a1["pred"] == a2["suggestion"]:
                    a3 = {"pred": a1["pred"], "rationale": "early exit: A1 and A2 agree", "_usage": {"input_tokens": 0, "output_tokens": 0}}
                    n_early_exit += 1
                    match = "✓" if a3["pred"] == y_true_i else "✗"
                    print(f"       Agent3  SKIPPED — A1=A2={label_str[a3['pred']]}  "
                          f"true={label_str[y_true_i]}  {match}")
                else:
                    a3 = run_agent3(sentence, a1, a2, b3, use_cot=use_cot)
                    usage["agent3"]["input_tokens"]  += a3["_usage"]["input_tokens"]
                    usage["agent3"]["output_tokens"] += a3["_usage"]["output_tokens"]
                    match = "✓" if a3["pred"] == y_true_i else "✗"
                    print(f"       Agent3  final={label_str[a3['pred']]:<13}"
                          f"  true={label_str[y_true_i]:<13}  {match}")
                    print(f"               rationale: {a3['rationale'][:80]}")

                record = {
                    "row": i,
                    "sentence": sentence,
                    "y_true": y_true_i,
                    "agent1_pred": a1["pred"],
                    "agent1_step1": a1["step1"],
                    "agent1_step2": a1["step2"],
                    "agent2_assessment": a2["assessment"],
                    "agent2_issues": a2["issues"],
                    "agent2_suggestion": a2["suggestion"],
                    "agent3_pred": a3["pred"],
                    "agent3_rationale": a3["rationale"],
                    "y_pred": a3["pred"],
                }

        except Exception as e:
            print(f"       ERROR: {e}")
            errors.append({"row": i, "error": str(e)})
            record = {
                "row": i, "sentence": sentence, "y_true": y_true_i,
                "y_pred": None,
                "agent1_pred": None, "agent2_suggestion": None, "agent3_pred": None,
            }

        records.append(record)

        # Write checkpoint immediately; rewrite full file to keep rows in order
        if checkpoint:
            checkpoint_data[i] = record
            sorted_records = [checkpoint_data[k] for k in sorted(checkpoint_data)]
            pd.DataFrame(sorted_records).to_csv(checkpoint, index=False)

        if delay > 0:
            time.sleep(delay)

    # Load full results from checkpoint (includes prior runs if resuming)
    if checkpoint and checkpoint.exists():
        results_df = pd.read_csv(checkpoint)
    else:
        results_df = pd.DataFrame(records)

    y_true = results_df["y_true"].tolist()
    y_pred = results_df["y_pred"].tolist()
    print(y_pred[:50])

    report   = classification_report(y_true, y_pred, labels=[0, 1], target_names=["not_deliberative", "deliberative"])
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1])
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro")

    mode_label = f"A1:{agent1_model}" if single_agent else f"A1:{agent1_model}  A2:{agent2_model}  A3:{agent3_model}"
    print(f"\n=== {'Single-agent' if single_agent else 'Agentic'} results  [{mode_label}] ===")
    print(f"Samples: {len(results_df)}  |  Errors: {len(errors)}")
    if not single_agent and early_exit:
        print(f"Early exits (A1=A2): {n_early_exit}/{len(results_df)}  ({n_early_exit/len(results_df):.1%})")
    print(f"Macro F1: {round(macro_f1, 4)}")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  {cm}")

    if not single_agent:
        # Agreement stats
        valid = results_df.dropna(subset=["agent1_pred", "agent2_suggestion", "agent3_pred"])
        a1_a2_agree = (valid["agent1_pred"] == valid["agent2_suggestion"]).mean()
        a1_a3_agree = (valid["agent1_pred"] == valid["agent3_pred"]).mean()
        a2_a3_agree = (valid["agent2_suggestion"] == valid["agent3_pred"]).mean()
        print(f"\nAgent agreement:  A1↔A2={a1_a2_agree:.1%}  A1↔A3={a1_a3_agree:.1%}  A2↔A3={a2_a3_agree:.1%}")

    # ── Cost report ───────────────────────────────────────────────────────────
    active_usage = {"agent1": usage["agent1"]} if single_agent else usage
    total_cost = 0.0
    print("\nCost breakdown:")
    print(f"  {'Agent':<10} {'Model':<28} {'Input tok':>10} {'Output tok':>11} {'Cost':>10}")
    print(f"  {'-'*73}")
    for agent_label, u in active_usage.items():
        cost = compute_cost(u["model"], u["input_tokens"], u["output_tokens"])
        total_cost += cost
        cost_str = f"${cost:.4f}" if cost > 0 else "free (local)"
        print(f"  {agent_label:<10} {u['model']:<28} {u['input_tokens']:>10,} {u['output_tokens']:>11,} {cost_str:>10}")
    print(f"  {'-'*73}")
    total_str = f"${total_cost:.4f}"
    print(f"  {'TOTAL':<10} {'':<28} {sum(u['input_tokens'] for u in active_usage.values()):>10,} {sum(u['output_tokens'] for u in active_usage.values()):>11,} {total_str:>10}")
    if not single_agent and n_early_exit > 0:
        saved_pct = n_early_exit / len(results_df)
        print(f"\n  Early exit saved ~{saved_pct:.0%} of Agent 3 calls.")

    if output_path:
        print(f"\nResults saved to {output_path}")

    return {
        "agent1_model": agent1_model,
        "agent2_model": agent2_model,
        "agent3_model": agent3_model,
        "n_samples": len(results_df),
        "n_errors": len(errors),
        "n_early_exit": n_early_exit,
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FOIA deliberative classifier — three-agent pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    backend_choices = list(BACKEND_REGISTRY.keys())

    parser.add_argument("--agent1-model", default="qwen3.5-9b", choices=backend_choices,
                        help="Agent 1 (Predictor) model")
    parser.add_argument("--agent2-model", default="qwen3.5-9b", choices=backend_choices,
                        help="Agent 2 (Critic) model")
    parser.add_argument("--agent3-model", default="qwen3.5-9b", choices=backend_choices,
                        help="Agent 3 (Judge) model")
    parser.add_argument(
        "--files", nargs="+", default=list(FILE_REGISTRY.keys()),
        choices=list(FILE_REGISTRY.keys()), metavar="KEY",
        help=f"Data files to evaluate. Available: {list(FILE_REGISTRY.keys())} (default: all)",
    )
    parser.add_argument("--data-dir", default=".", help="Directory with FOIA .txt files")
    parser.add_argument("--output", default=None, help="CSV path to save full results (optional)")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for quick testing")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Seconds between sentences (3 API calls each, default: 0.2)")
    parser.add_argument("--few-shot", type=int, default=0, metavar="N",
                        help="Few-shot examples for Agent 1 (balanced, sampled from other files)")
    parser.add_argument("--hard-results", nargs="+", default=None, metavar="CSV",
                        help="Path(s) to results CSVs from previous runs. Wrong predictions are used as\n"
                             "hard few-shot examples (correct label shown in prompt). Requires --few-shot.")
    parser.add_argument("--no-early-exit", action="store_true",
                        help="Always run all three agents, even when A1 and A2 agree")
    parser.add_argument("--single-agent", action="store_true",
                        help="Run Agent 1 only (skip Critic and Judge)")
    parser.add_argument("--cot", action="store_true",
                        help="Enable chain-of-thought reasoning for Agent 1 (step-by-step predecisional/deliberative analysis)")
    
    args = parser.parse_args()

    print(f"Files:    {args.files}")
    print(f"Agent 1:  {args.agent1_model}  (Predictor)")
    print(f"Agent 2:  {args.agent2_model}  (Critic)")
    print(f"Agent 3:  {args.agent3_model}  (Judge)")
    early_exit = not args.no_early_exit
    print(f"Agent 1 mode: {'CoT (chain-of-thought)' if args.cot else 'simple (base prompt)'}")
    print(f"Mode: {'single-agent (Agent 1 only)' if args.single_agent else '3-agent pipeline'}")
    if not args.single_agent:
        print(f"Early exit: {'enabled (Agent 3 skipped when A1=A2)' if early_exit else 'disabled'}")
    if args.few_shot > 0:
        print(f"Few-shot: {args.few_shot} examples for Agent 1 (excluding source file)")
    if args.hard_results:
        if args.few_shot == 0:
            parser.error("--hard-results requires --few-shot N > 0")
        print(f"Hard examples: {args.hard_results}")
    print()

    df = load_data(args.data_dir, file_keys=args.files)
    print(f"Loaded {len(df)} sentences.")
    print(df["y_true"].value_counts().rename({0: "not_deliberative", 1: "deliberative"}).to_string())
    print()

    all_files = list(FILE_REGISTRY.keys())
    few_shot_pool = load_data(args.data_dir, file_keys=all_files) if args.files != all_files else df
    if args.few_shot > 0 and args.files != all_files:
        extra = [k for k in all_files if k not in args.files]
        print(f"Few-shot pool includes all files (extra sources: {extra})\n")

    hard_examples_df = None
    if args.hard_results:
        hard_examples_df = load_hard_examples(args.hard_results)
        n_hard_pos = (hard_examples_df["y_true"] == 1).sum()
        n_hard_neg = (hard_examples_df["y_true"] == 0).sum()
        print(f"Hard examples loaded: {len(hard_examples_df)} wrong predictions "
              f"(deliberative={n_hard_pos}, not_deliberative={n_hard_neg})\n")

    run_experiment(
        df,
        agent1_model=args.agent1_model,
        agent2_model=args.agent2_model,
        agent3_model=args.agent3_model,
        output_path=args.output,
        delay=args.delay,
        max_rows=args.max_rows,
        n_few_shot=args.few_shot,
        few_shot_pool=few_shot_pool,
        hard_examples_df=hard_examples_df,
        early_exit=early_exit,
        single_agent=args.single_agent,
        use_cot=args.cot,
    )


if __name__ == "__main__":
    main()
