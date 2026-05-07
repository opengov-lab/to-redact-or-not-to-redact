"""
count_verbs.py — Count verbs in the 'sentence' column of a CSV file.

Usage:
    python count_verbs.py <input.csv> [output.json]

Output:
  <output>.json  — per-verb counts and sentence percentages

The input CSV must have a 'sentence' column.
"""

import sys
import json
import csv
from collections import Counter
import spacy

VERB_POS = {"VERB", "AUX"}


def count_verbs(csv_path: str) -> tuple[dict, int]:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    total_counts: Counter = Counter()
    sentence_counts: Counter = Counter()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sentences = [row["sentence"] for row in reader if row.get("sentence")]

    n = len(sentences)
    print(f"Processing {n} sentences...", file=sys.stderr)

    for doc in nlp.pipe(sentences, batch_size=256):
        seen_verbs = set()
        for token in doc:
            if token.pos_ in VERB_POS:
                lemma = token.lemma_.lower()
                total_counts[lemma] += 1
                seen_verbs.add(lemma)
        for lemma in seen_verbs:
            sentence_counts[lemma] += 1

    verb_results = {
        verb: {
            "count": total_counts[verb],
            "sentence_count": sentence_counts[verb],
            "sentence_pct": round(100 * sentence_counts[verb] / n, 2),
        }
        for verb in total_counts
    }
    verb_results = dict(sorted(verb_results.items(), key=lambda x: x[1]["sentence_count"], reverse=True))

    return verb_results, n


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace(".csv", "_verb_counts.json")

    verb_results, n_sentences = count_verbs(csv_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(verb_results, f, indent=2)

    print(f"Saved {len(verb_results)} unique verbs to {json_path} ({n_sentences} sentences total)")
    print(f"\n{'Rank':<5} {'Verb':<20} {'Count':>7}  {'Sentences':>10}  {'% sentences':>12}")
    print("-" * 60)
    for rank, (verb, stats) in enumerate(list(verb_results.items())[:30], 1):
        print(f"  {rank:<3} {verb:<20} {stats['count']:>7}  {stats['sentence_count']:>10}  {stats['sentence_pct']:>11.1f}%")


if __name__ == "__main__":
    main()
