# To Redact or Not to Redact

Automated classification of government email sentences as *deliberative* or *non-deliberative* under FOIA Exemption 5 (deliberative process privilege). This project compares single-agent and multi-agent LLM pipelines across multiple prompting strategies. This repository includes code for running experiments, evaluating results, and analyzing linguistic features of the sentences for the paper 'To Redact, or not to Redact? A Local LLM Approach to Deliberative Process Privilege Classification'.

## Paper abstract

Government transparency laws, like the Freedom of Information (FOIA) acts in the United States and United Kingdom, and the Woo (Open Government Act) in the Netherlands, grant citizens the right to directly request documents from the government. As these documents might contain sensitive information, such as personal information or threats to national security, the laws allow governments to redact sensitive parts of the documents prior to release. 
We build on prior research to perform automatic sensitivity classification based on the FOIA Exemption 5 deliberative process privilege using Large Language Models (LLMs). 
However, processing documents not yet cleared for review via third-party cloud APIs is often legally or politically untenable. 
Therefore, in this work, we perform sensitivity classification with a small, local model, deployable on consumer-grade hardware (Qwen-3.5 9B). 
We compare eight variants of applying LLMs for sentence classification, using well-known prompting techniques, and find that a combination of Chain-of-Thought prompting and few-shot prompting with error-based examples outperforms classification models of earlier work in terms of recall and F2 score. 
This method also closely approaches the performance of a widely-used, cost-efficient commercial model (Gemini 2.5 Flash). 
In an additional analysis, we find that deliberative sentences contain more verbs that indicate the expression of opinions, and are more often phrased in in first-person. 
Above all, deliberativeness seems characterized by the presence of a combination of multiple indicators, in particular the combination of first-person words with a verb for expressing opinion.

## Background

Under the Freedom of Information Act (FOIA), agencies may withhold documents that are both **predecisional** (created before a final decision) and **deliberative** (reflect internal agency opinions or recommendations). Determining which sentences qualify is labor-intensive and inconsistent. This project explores whether LLMs can reliably automate this judgment.

**Task**: Binary classification — `1` = always deliberative (withheld), `0` = not deliberative (releasable).

## Architecture

The core pipeline chains up to three LLM agents:

```
Sentence ──► Agent 1 (Predictor) ──► Agent 2 (Critic) ──► Agent 3 (Judge) ──► Final label
                                          │
                              If Agents 1 & 2 agree, skip Agent 3 (early exit)
```

- **Agent 1** — predicts and optionally reasons step-by-step (CoT)
- **Agent 2** — critically reviews Agent 1's reasoning and prediction
- **Agent 3** — weighs both analyses and delivers a final verdict

## Variants

| Variant | Description |
|---|---|
| `simple` | Agent 1 only, no reasoning |
| `single` | Agent 1 only with chain-of-thought |
| `fewshot` | Few-shot examples (balanced labels) |
| `fewshot-hard` | Few-shot examples drawn from prior errors |
| `fewshot-cot-hard` | Hard few-shot + CoT |
| `multi-nocot` | Three-agent pipeline, no CoT |
| `multi` | Three-agent pipeline with CoT |
| `fewshot-cot-hard-multi` | Full pipeline: CoT + hard examples + three agents |

## Models

- **Gemini 2.5 Flash** — cloud API (requires `GOOGLE_API_KEY`)
- **Qwen 3.5 9B** — local inference via HuggingFace Transformers

## Repository Structure

```
to-redact-or-not-to-redact/
├── src/
│   ├── run.py            # Main experiment runner
│   ├── prompts.py        # System prompts and user templates for all agents
│   └── evaluate_all.py   # Metrics, comparison tables, and plots
├── analysis/
│   ├── verb_analysis.py  # Linguistic feature analysis (verb categories, tense, modality)
│   ├── count_verbs.py    # Verb frequency counter
│   └── compare_variants.py  # Qualitative cross-variant comparison
├── data/
│   ├── foia_K1-Final.txt  # 1,751 sentences
│   ├── foia_K2--Final.txt # 1,341 sentences
│   ├── foia_K3-Final.txt  # 2,043 sentences
│   ├── foia_K5-Final.txt  # 1,857 sentences
│   └── foia_R4-Final.txt  #   983 sentences
└── results/               # Output CSVs from all experiments
```

Data files are pipe-delimited with columns: `label | sentence | always_deliberative | non_iia | sentiment | subjectivity`.

## Setup

```bash
pip install pandas scikit-learn spacy google-genai transformers torch matplotlib python-dotenv
python -m spacy download en_core_web_sm
```

Create a `.env` file in the project root for the Gemini backend:

```
GOOGLE_API_KEY=your_key_here
```

## Usage

### Run experiments

```bash
# Single-agent baseline (Qwen, dataset K1)
python src/run.py --single-agent --files K1

# Three-agent pipeline with CoT on all datasets
python src/run.py --cot --files K1 K2 K3 K5 R4

# Few-shot with hard examples mined from prior runs
python src/run.py --few-shot 4 --hard-results results/gemini-simple_K1.csv --cot --files K1

# Use Gemini instead of local Qwen
python src/run.py --agent1-model gemini-2.5-flash --cot --files K1
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--agent1-model` | `qwen3.5-9b` | Model for Agent 1 |
| `--agent2-model` | `qwen3.5-9b` | Model for Agent 2 |
| `--agent3-model` | `qwen3.5-9b` | Model for Agent 3 |
| `--files` | all | Datasets to run (`K1 K2 K3 K5 R4`) |
| `--single-agent` | off | Skip Agents 2 & 3 |
| `--cot` | off | Enable chain-of-thought reasoning |
| `--few-shot N` | off | Use N few-shot examples per label |
| `--hard-results CSV` | — | CSVs to mine hard examples from |
| `--no-early-exit` | off | Always run all three agents |
| `--delay SECS` | 0.2 | Delay between API calls |
| `--max-rows N` | — | Limit rows (for testing) |

Experiments are **resumable**: re-running with the same output path picks up from the last completed row.

### Evaluate results

```bash
# Aggregated metrics table sorted by macro F1
python src/evaluate_all.py --aggregate --sort macro_f1 --markdown

# Per-dataset breakdown for Gemini runs only
python src/evaluate_all.py --model gemini --sort macro_f1
```

### Analyze variants qualitatively

```bash
# Show sentences where variants disagree, test Agent 2 confusion hypothesis
python src/analysis/compare_variants.py --dataset K1 K2 --hypothesis --text
```

### Linguistic analysis

```bash
# Verb frequency across deliberative vs. non-deliberative sentences
python src/analysis/count_verbs.py

# Full feature analysis (modality, tense, predecisional markers)
python src/analysis/verb_analysis.py
```

## Output Format

Each result CSV contains one row per sentence with columns:

| Column | Description |
|---|---|
| `row` | Row index in source file |
| `sentence` | Input sentence |
| `y_true` | Ground truth label |
| `agent1_pred` | Agent 1 prediction |
| `agent1_step1` | Agent 1 predecisional reasoning |
| `agent1_step2` | Agent 1 deliberative reasoning |
| `agent2_assessment` | Agent 2 critique summary |
| `agent2_issues` | Specific issues Agent 2 identified |
| `agent2_suggestion` | Agent 2 recommended label |
| `agent3_pred` | Agent 3 final prediction |
| `agent3_rationale` | Agent 3 reasoning |
| `y_pred` | Final prediction used for evaluation |

## License

Apache 2.0 — see [LICENSE](LICENSE).
