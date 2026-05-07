"""
Analyse easy_sentences_{0,1}.csv and report what percentage of sentences
contain each of the following categories, plus sentence-length statistics.

Main categories (lemma-matched on VERB/AUX tokens via spaCy):
  stative   – verbs describing a state rather than an action (e.g. "be",
              "have", "seem", "contain", "belong"); includes perception and
              emotion verbs (e.g. "like", "fear", "trust").
  reporting – verbs that introduce speech or information (e.g. "say",
              "report", "argue", "confirm", "reveal", "emphasise").
  modal     – auxiliary verbs expressing possibility, necessity, or volition
              (e.g. "can", "could", "may", "should", "must", "would").
  cognitive – verbs describing mental processes (e.g. "think", "believe",
              "know", "consider", "expect", "decide", "doubt").
  person    – first-person pronouns matched on surface form across all tokens
              (i, we, me, us, my, our, mine, ours).

Predecisional indicators:
  future_temporal_adv – surface adverbs/phrases pointing to a later point in
                        time (e.g. "subsequently", "in due course"); single
                        tokens matched against FUTURE_TEMPORAL_ADVERBS,
                        multi-word phrases substring-matched against
                        FUTURE_TEMPORAL_PHRASES.
  past_tense          – any VERB/AUX token with Tense=Past (spaCy morph;
                        no lexicon).
"""

import sys
import pandas as pd
import spacy

# ---------------------------------------------------------------------------
# Verb lexicons  (lemma form)
# ---------------------------------------------------------------------------

STATIVE_VERBS = {
    "be", "have", "seem", "appear", "exist", "contain", "belong", "consist",
    "include", "involve", "lack", "mean", "matter", "owe", "own", "possess",
    "remain", "resemble", "suffice", "suit",
    # perception/emotion statives
    "like", "love", "hate", "prefer", "want", "need", "wish", "desire",
    "fear", "hope", "trust", "depend", "differ", "agree", "disagree",
    "fit", "equal",
}

REPORTING_VERBS = {
    "say", "tell", "report", "state", "claim", "argue", "suggest",
    "indicate", "note", "mention", "announce", "declare", "explain",
    "describe", "confirm", "deny", "assert", "allege", "state",
    "remark", "observe", "insist", "emphasise", "emphasize", "warn",
    "add", "conclude", "state", "propose", "recommend", "advise",
    "urge", "request", "ask", "acknowledge", "admit", "concede",
    "reveal", "disclose", "show", "demonstrate", "find", "stress",
    "write", "respond", "reply", "answer", "testify", "cite",
}

MODAL_VERBS = {
    "can", "could", "may", "might", "shall", "should", "will", "would",
    "must", "ought", "need", "dare",
}

COGNITIVE_VERBS = {
    "think", "believe", "know", "understand", "consider", "assume",
    "expect", "imagine", "suppose", "recognize", "recognise", "realise",
    "realize", "perceive", "feel", "find", "wonder", "remember", "forget",
    "recall", "anticipate", "conceive", "decide", "determine", "judge",
    "estimate", "predict", "suspect", "hypothesize", "hypothesise",
    "conclude", "infer", "learn", "discover", "doubt"
}

PERSON_WORDS = {"i", "we", "me", "us", "my", "our", "mine", "ours"}

# ---------------------------------------------------------------------------
# Predecisional indicators
# ---------------------------------------------------------------------------


# Adverbs pointing to a later point in time (single tokens, surface-matched).
FUTURE_TEMPORAL_ADVERBS = {
    "later", "subsequently", "eventually", "thereafter", "hereafter",
    "afterwards", "afterward", "ultimately", "soon", "imminently",
    "henceforth", "prospectively", "momentarily", "tomorrow", "next week",
    "next month", "next year", "future", "forthcoming", "pending", "upcoming"
}

# Multi-word temporal phrases referring to a later time (substring-matched).
FUTURE_TEMPORAL_PHRASES = {
    "in due course", "going forward", "at a later stage", "at a later date",
    "at a later time", "at a later point", "in the future",
    "in the near future", "at some future point", "at some point",
}


CATEGORIES = {
    "stative":              STATIVE_VERBS,
    "reporting":            REPORTING_VERBS,
    "modal":                MODAL_VERBS,
    "cognitive":            COGNITIVE_VERBS,
    "person":               PERSON_WORDS,
    "future_temporal_adv":  FUTURE_TEMPORAL_ADVERBS,
    # past_tense has no lexicon – detected via spaCy morphology
    "past_tense":           None,
}

MAIN_CATS    = ["stative", "reporting", "modal", "cognitive", "person"]
COOCCUR_CATS = MAIN_CATS + ["future_temporal_adv"]
PREDEC_CATS  = ["future_temporal_adv", "past_tense"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nlp():
    model = "en_core_web_sm"
    try:
        return spacy.load(model, disable=["ner", "parser"])
    except OSError:
        print(f"[ERROR] spaCy model '{model}' not found.")
        print(f"        Install it with:  python -m spacy download {model}")
        sys.exit(1)


def sentence_flags(doc) -> dict[str, bool]:
    """Return one bool per category: does this sentence contain a token from that category?

    Verb categories (stative, reporting, modal, cognitive)
    are matched via spaCy lemma on VERB/AUX tokens.
    Surface categories (person, future_temporal_adv) are matched by lowercased
    surface form across all tokens.
    Multi-word phrases (FUTURE_TEMPORAL_PHRASES) are matched as substrings of
    the lowercased sentence text.
    past_tense is detected via spaCy morphological Tense feature on
    VERB/AUX tokens (no lexicon); only Tense=Past is counted.
    """
    flags = {cat: False for cat in CATEGORIES}

    # Categories matched by surface form on any token (not just verbs).
    # past_tense uses None as its lexicon – handled separately via morphology.
    SURFACE_CATS = {"person", "future_temporal_adv"}
    verb_cats = [c for c in CATEGORIES
                 if c not in SURFACE_CATS and CATEGORIES[c] is not None]

    # Phrase-level matching (substring of full sentence).
    sentence_lower = doc.text.lower()
    for phrase in FUTURE_TEMPORAL_PHRASES:
        if phrase in sentence_lower:
            flags["future_temporal_adv"] = True
            break

    for token in doc:
        lower = token.text.lower()
        # Surface-matched categories: check every token.
        for cat in SURFACE_CATS:
            if CATEGORIES[cat] and lower in CATEGORIES[cat]:
                flags[cat] = True
        # Verb-based categories: lemma match restricted to VERB/AUX.
        if token.pos_ in ("VERB", "AUX"):
            lemma = token.lemma_.lower()
            for cat in verb_cats:
                if lemma in CATEGORIES[cat]:
                    flags[cat] = True
            # Past tense via morphological analysis.
            tense = token.morph.get("Tense")
            if tense and tense[0] == "Past":
                flags["past_tense"] = True

    return flags


def sentence_counts(doc) -> dict[str, int]:
    """Return the number of matching tokens per COOCCUR_CATS category in this sentence.

    Mirrors the matching logic of sentence_flags but accumulates a count
    instead of a boolean, so e.g. two stative verbs contribute 2.
    MAIN_CATS are counted (stative, reporting, modal, cognitive, person) plus
    future_temporal_adv (surface tokens and phrases).
    """
    counts = {cat: 0 for cat in COOCCUR_CATS}
    SURFACE_CATS_MAIN = {"person", "future_temporal_adv"}
    verb_cats_main = [c for c in MAIN_CATS if c not in SURFACE_CATS_MAIN]

    # Phrase-level matching for future_temporal_adv
    sentence_lower = doc.text.lower()
    for phrase in FUTURE_TEMPORAL_PHRASES:
        if phrase in sentence_lower:
            counts["future_temporal_adv"] += 1

    for token in doc:
        lower = token.text.lower()
        if lower in PERSON_WORDS:
            counts["person"] += 1
        if lower in FUTURE_TEMPORAL_ADVERBS:
            counts["future_temporal_adv"] += 1
        if token.pos_ in ("VERB", "AUX"):
            lemma = token.lemma_.lower()
            for cat in verb_cats_main:
                if lemma in CATEGORIES[cat]:
                    counts[cat] += 1
    return counts


def analyse_file(path: str, nlp) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sentence" not in df.columns:
        raise ValueError(f"No 'sentence' column found in {path}")

    sentences = df["sentence"].fillna("").tolist()
    docs = list(nlp.pipe(sentences, batch_size=64))

    results = [sentence_flags(doc) for doc in docs]
    flags_df = pd.DataFrame(results)

    # Per-category token counts for MAIN_CATS (used in co-occurrence report)
    counts_list = [sentence_counts(doc) for doc in docs]
    counts_df = pd.DataFrame(counts_list).rename(columns={c: f"{c}_n" for c in COOCCUR_CATS})
    flags_df = pd.concat([flags_df, counts_df], axis=1)

    # Length columns: token count (alpha tokens only) and character count
    flags_df["len_tokens"] = [sum(1 for t in doc if not t.is_space) for doc in docs]
    flags_df["len_chars"]  = [len(s.strip()) for s in sentences]

    return pd.concat([df.reset_index(drop=True), flags_df], axis=1)


def report(label: str, df: pd.DataFrame):
    """Count/pct for main verb and person categories."""
    n = len(df)
    print(f"\n{'='*60}")
    print(f"  {label}  (n = {n} sentences)")
    print(f"{'='*60}")
    for cat in MAIN_CATS:
        count = df[cat].sum()
        pct   = 100 * count / n if n else 0.0
        print(f"  {cat:<20}  {count:>4}  /  {n}   ({pct:.1f} %)")


def predecisional_report(label: str, df: pd.DataFrame):
    """Count/pct and co-occurrence for predecisional indicator categories."""
    import itertools

    focus = PREDEC_CATS
    n = len(df)

    print(f"\n{'='*60}")
    print(f"  PREDECISIONAL INDICATORS — {label}  (n = {n})")
    print(f"{'='*60}")

    # Per-category counts
    for cat in focus:
        count = df[cat].sum()
        pct   = 100 * count / n if n else 0.0
        print(f"  {cat:<20}  {count:>4}  /  {n}   ({pct:.1f} %)")
    any_predec = df[focus].any(axis=1)
    count = any_predec.sum()
    pct   = 100 * count / n if n else 0.0
    print(f"  {'any predecisional':<20}  {count:>4}  /  {n}   ({pct:.1f} %)")

    # Number of predecisional categories present per sentence
    per_sentence = df[focus].sum(axis=1)
    print(f"  Predecisional categories present per sentence:")
    print(f"    mean   = {per_sentence.mean():.3f}")
    print(f"    median = {per_sentence.median():.1f}")
    print(f"    std    = {per_sentence.std():.3f}")
    print(f"    min    = {int(per_sentence.min())}")
    print(f"    max    = {int(per_sentence.max())}")

    # Distribution
    print(f"  Distribution (# predecisional categories present):")
    for k in range(len(focus) + 1):
        cnt = (per_sentence == k).sum()
        pct = 100 * cnt / n if n else 0.0
        print(f"    {k} categories:  {cnt:>4}  ({pct:.1f} %)")

    # Sentences with ≥2 predecisional categories (genuine co-occurrence)
    cooccur = (per_sentence >= 2).sum()
    pct = 100 * cooccur / n if n else 0.0
    print(f"  Sentences with ≥2 predecisional categories: {cooccur}  ({pct:.1f} %)")

    # Pairwise co-occurrence counts
    print(f"  Pairwise co-occurrence counts:")
    for a, b in itertools.combinations(focus, 2):
        cnt = (df[a] & df[b]).sum()
        pct = 100 * cnt / n if n else 0.0
        print(f"    {a:<20} + {b:<20}  {cnt:>4}  ({pct:.1f} %)")

    # 5 example sentences that hit ≥1 predecisional indicator
    hit_rows = df[any_predec].dropna(subset=["sentence"]).head(5)
    agent2_cols = [c for c in ("agent2_assessment", "agent2_issues", "agent2_suggestion") if c in df.columns]
    print(f"  Example sentences with ≥1 predecisional indicator:")
    for i, (_, row) in enumerate(hit_rows.iterrows(), 1):
        cats_present = [c for c in focus if row[c]]
        print(f"    [{i}] {row['sentence'].strip()}")
        print(f"        indicators: {', '.join(cats_present)}")
        for col in agent2_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                lbl = col.replace("agent2_", "").replace("_", " ")
                print(f"        agent2 {lbl}: {str(val).strip()}")


def cooccurrence_report(label: str, df: pd.DataFrame):
    """Report co-occurrence statistics for stative, reporting, cognitive, and person categories."""
    import itertools

    focus = ["stative", "reporting", "modal", "cognitive", "person", "future_temporal_adv"]
    n = len(df)

    # Number of focus categories present per sentence
    per_sentence = df[focus].sum(axis=1)  # 0..4

    print(f"\n{'='*60}")
    print(f"  CO-OCCURRENCE (stative / reporting / cognitive / person)")
    print(f"  {label}  (n = {n} sentences)")
    print(f"{'='*60}")
    print(f"  Categories present per sentence:")
    print(f"    mean   = {per_sentence.mean():.3f}")
    print(f"    median = {per_sentence.median():.1f}")
    print(f"    std    = {per_sentence.std():.3f}")
    print(f"    min    = {int(per_sentence.min())}")
    print(f"    max    = {int(per_sentence.max())}")

    # Distribution: how many sentences have 0, 1, 2, 3, 4 categories
    print(f"  Distribution (# focus categories present):")
    for k in range(len(focus) + 1):
        cnt = (per_sentence == k).sum()
        pct = 100 * cnt / n if n else 0.0
        print(f"    {k} categories:  {cnt:>4}  ({pct:.1f} %)")

    # Sentences with ≥2 focus categories (genuine co-occurrence)
    cooccur = (per_sentence >= 2).sum()
    pct = 100 * cooccur / n if n else 0.0
    print(f"  Sentences with ≥2 categories: {cooccur}  ({pct:.1f} %)")

    # Total indicator token counts (counts multiple matches within a sentence)
    count_cols = [f"{c}_n" for c in focus]
    total_per_sentence = df[count_cols].sum(axis=1)
    print(f"  Total indicator tokens per sentence:")
    print(f"    sum    = {int(total_per_sentence.sum())}")
    print(f"    mean   = {total_per_sentence.mean():.3f}")
    print(f"    median = {total_per_sentence.median():.1f}")
    print(f"    std    = {total_per_sentence.std():.3f}")
    print(f"    max    = {int(total_per_sentence.max())}")
    print(f"  Per-category token totals:")
    for cat, col in zip(focus, count_cols):
        print(f"    {cat:<12}  {int(df[col].sum())}")

    # Pairwise co-occurrence counts
    print(f"  Pairwise co-occurrence counts:")
    for a, b in itertools.combinations(focus, 2):
        cnt = (df[a] & df[b]).sum()
        pct = 100 * cnt / n if n else 0.0
        print(f"    {a:<10} + {b:<10}  {cnt:>4}  ({pct:.1f} %)")

    # One example sentence per category count (1..5)
    has_ytrue = "y_true" in df.columns
    print(f"  Example sentences by number of categories present:")
    for k in range(0, len(focus) + 1):
        subset = df[per_sentence == k].dropna(subset=["sentence"])
        if subset.empty:
            continue

        for l in range(min(5, len(subset))):
            row = subset.sample(n=1, random_state=l).iloc[0]
            cats_present = [c for c in focus if row[c]]
            if has_ytrue and pd.notna(row.get("y_true")):
                lbl = "deliberative" if int(row["y_true"]) == 1 else "not deliberative"
            else:
                lbl = "N/A"
            print(f"    [{k} cat] label={lbl}")
            print(f"           {row['sentence'].strip()}")
            print(f"           categories: {', '.join(cats_present)}")


def length_report(label: str, df: pd.DataFrame):
    n = len(df)
    print(f"\n{'='*60}")
    print(f"  LENGTH STATS — {label}  (n = {n})")
    print(f"{'='*60}")
    for col, unit in [("len_tokens", "tokens"), ("len_chars", "chars")]:
        s = df[col]
        print(f"  {unit:<8}  mean={s.mean():.1f}  median={s.median():.1f}"
              f"  std={s.std():.1f}  min={s.min()}  max={s.max()}")
    # Distribution buckets (by token count)
    bins   = [0, 10, 20, 30, 50, float("inf")]
    labels = ["≤10", "11-20", "21-30", "31-50", ">50"]
    bucketed = pd.cut(df["len_tokens"], bins=bins, labels=labels, right=True)
    counts   = bucketed.value_counts().reindex(labels, fill_value=0)
    print(f"  token buckets:")
    for lbl, cnt in counts.items():
        pct = 100 * cnt / n if n else 0.0
        print(f"    {lbl:<6}  {cnt:>4}  ({pct:.1f} %)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    files = {
        "easy_sentences_0.csv": "easy_sentences_0.csv",
        "easy_sentences_1.csv": "easy_sentences_1.csv",
    }

    print("Loading spaCy model …")
    nlp = load_nlp()

    frames = {}
    for label, path in files.items():
        print(f"Processing {path} …")
        frames[label] = analyse_file(path, nlp)

    # Per-file reports
    for label, df in frames.items():
        report(label, df)
        length_report(label, df)
        cooccurrence_report(label, df)
        predecisional_report(label, df)

    # Combined report
    combined = pd.concat(frames.values(), ignore_index=True)
    report("COMBINED (all files)", combined)
    length_report("COMBINED (all files)", combined)
    predecisional_report("COMBINED (all files)", combined)
    cooccurrence_report("COMBINED (all files)", combined)


if __name__ == "__main__":
    main()
