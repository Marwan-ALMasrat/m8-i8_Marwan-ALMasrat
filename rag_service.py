"""Module 8 — Integration Task: RAG Service."""

import json
import os
import re
import string
from typing import List

import weaviate
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from index_helpers import bm25_search, dense_search, hybrid_search  # noqa: F401

# ---------------------------------------------------------------------------
# Constants — DO NOT MODIFY
# ---------------------------------------------------------------------------

CLASS_NAME = "Post"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
GENERATOR_MODEL = "google/flan-t5-base"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

ABSTAIN_PHRASES = [
    "i don't know",
    "i do not know",
    "not in the context",
    "the context does not",
    "cannot be answered",
    "no information",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "him", "his", "i", "if", "in", "into", "is",
    "it", "its", "just", "may", "me", "might", "my", "no", "nor", "not",
    "of", "on", "or", "our", "out", "over", "she", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "under", "until", "up", "was",
    "we", "were", "what", "when", "where", "which", "while", "who",
    "will", "with", "would", "you", "your",
}

# ---------------------------------------------------------------------------
# Module-level model loading — loaded ONCE per process
# ---------------------------------------------------------------------------

_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
_embedder = SentenceTransformer(EMBEDDER_MODEL)

_client: weaviate.Client | None = None


def _get_client() -> weaviate.Client:
    global _client
    if _client is None:
        _client = weaviate.Client(WEAVIATE_URL)
    return _client


# ---------------------------------------------------------------------------
# Helpers (provided)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize_for_groundedness(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t and t not in STOPWORDS}


def _whole_word_match(keyword: str, answer: str) -> bool:
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, answer, flags=re.IGNORECASE) is not None


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 5) -> List[dict]:
    """Retrieve top-k contexts using hybrid_search with alpha=0.5."""

    client = _get_client()

    # Call hybrid search — returns list of doc_ids
    doc_ids = hybrid_search(client, query, k, _embedder, alpha=0.5)

    # Resolve each doc_id to full record with title and answer_text
    contexts = []
    for doc_id in doc_ids:
        result = (
            client.query
            .get("Post", ["doc_id", "title", "answer_text"])
            .with_where({
                "path": ["doc_id"],
                "operator": "Equal",
                "valueText": doc_id
            })
            .with_limit(1)
            .do()
        )
        hits = result.get("data", {}).get("Get", {}).get("Post", [])
        if hits:
            contexts.append({
                "doc_id":      hits[0]["doc_id"],
                "title":       hits[0]["title"],
                "answer_text": hits[0]["answer_text"],
            })

    return contexts


def build_prompt(query: str, contexts: List[dict]) -> str:
    """Build the canonical prompt with exact template."""

    # Build context lines — truncate answer_text to first 80 tokens
    context_lines = []
    for i, ctx in enumerate(contexts, start=1):
        tokens = ctx["answer_text"].split()
        truncated = " ".join(tokens[:80])
        context_lines.append(f"[{i}] {ctx['title']}: {truncated}")

    context_block = "\n".join(context_lines)

    prompt = (
        'Answer the question using only the context. '
        'If the context does not contain the answer, say "I don\'t know."\n\n'
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    return prompt


def generate(prompt: str) -> str:
    """Call flan-t5-base with greedy decoding."""

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = _model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=1
    )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_pipeline(query: str, k: int = 5) -> dict:
    """Compose retrieve -> build_prompt -> generate."""

    contexts = retrieve(query, k)
    prompt   = build_prompt(query, contexts)
    answer   = generate(prompt)

    return {
        "query":    query,
        "answer":   answer,
        "contexts": contexts,
        "prompt":   prompt,
    }


def groundedness_score(answer: str, contexts: List[dict]) -> float:
    """Content-word overlap between answer and concatenated contexts."""

    # Handle empty answer
    answer_tokens = _tokenize_for_groundedness(answer)
    if not answer_tokens:
        return 0.0

    # Concatenate all context answer_texts
    context_text = " ".join(ctx["answer_text"] for ctx in contexts)
    context_tokens = _tokenize_for_groundedness(context_text)

    # |answer ∩ context| / |answer|
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def evaluate_rag(eval_path: str) -> dict:
    """Run the pipeline over the 30-pair eval set."""

    # Load eval rows
    eval_rows = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            eval_rows.append(json.loads(line))

    # Accumulators
    answerable_recall,  answerable_grd  = [], []
    borderline_abstain, borderline_grd  = [], []
    per_question = []

    for i, row in enumerate(eval_rows):
        question   = row["question"]
        difficulty = row["difficulty"]
        keywords   = row.get("expected_answer_keywords", [])

        # Run pipeline
        result  = rag_pipeline(question)
        answer  = result["answer"]
        contexts = result["contexts"]

        # Groundedness
        grd = groundedness_score(answer, contexts)

        if difficulty in ("single_fact", "single_doc_synthesis"):
            # Answerable row — compute keyword recall
            if keywords:
                matched = [kw for kw in keywords if _whole_word_match(kw, answer)]
                recall  = len(matched) / len(keywords)
            else:
                matched = []
                recall  = 0.0

            answerable_recall.append(recall)
            answerable_grd.append(grd)

            per_question.append({
                "row_index":        i,
                "difficulty":       difficulty,
                "question":         question,
                "answer":           answer,
                "groundedness":     grd,
                "matched_keywords": matched,
            })

        elif difficulty == "borderline":
            # Borderline row — check abstention
            answer_lower = answer.lower()

            phrase_abstain = any(p in answer_lower for p in ABSTAIN_PHRASES)
            short_and_low  = len(answer.strip()) <= 20 and grd <= 0.2
            abstained      = phrase_abstain or short_and_low

            borderline_abstain.append(1 if abstained else 0)
            borderline_grd.append(grd)

            per_question.append({
                "row_index":    i,
                "difficulty":   difficulty,
                "question":     question,
                "answer":       answer,
                "groundedness": grd,
                "abstained":    abstained,
            })

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "answer_keyword_recall_main":   mean(answerable_recall),
        "borderline_abstain_rate":      mean(borderline_abstain),
        "mean_groundedness_main":       mean(answerable_grd),
        "mean_groundedness_borderline": mean(borderline_grd),
        "per_question":                 per_question,
    }


# ---------------------------------------------------------------------------
# Main — run evaluation and print results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running RAG evaluation...")
    results = evaluate_rag("data/rag_eval.jsonl")

    print(f"\nanswer_keyword_recall_main:   {results['answer_keyword_recall_main']:.4f}")
    print(f"borderline_abstain_rate:      {results['borderline_abstain_rate']:.4f}")
    print(f"mean_groundedness_main:       {results['mean_groundedness_main']:.4f}")
    print(f"mean_groundedness_borderline: {results['mean_groundedness_borderline']:.4f}")