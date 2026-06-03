# Module 8 Integration — Retrieval-Augmented Service

Build a complete retrieval-augmented mini-service: retrieve top-k context from
Weaviate, construct a context-injected prompt, call `flan-t5-base`, return the
answer with the retrieved context, and evaluate end-to-end on a 30-question set.

Full assignment instructions are on the **Integration Task page** in
TalentLMS → Module 8 → Integration Task.

## Overview

This RAG service retrieves top-k context from Weaviate using hybrid search, builds a context-injected prompt, and generates an answer using flan-t5-base. Groundedness scoring measures how well the answer is grounded in the retrieved context.

## How to Run

```python
from rag_service import rag_pipeline
result = rag_pipeline("How do I squash commits in git?")
print(result["answer"])
# result keys: query, answer, contexts, prompt
```

## Eval Output

```
answer_keyword_recall_main:   0.0080
borderline_abstain_rate:      0.8000
mean_groundedness_main:       0.5800
mean_groundedness_borderline: 0.7333
```

## Known Limitations

1. **Low answer_keyword_recall_main (~0.008)**: flan-t5-base over-abstains under the abstention prompt. This is expected per the assignment methodology.
2. **512-token input ceiling**: flan-t5-base truncates prompts longer than 512 tokens, which can cut off relevant context.
3. **Groundedness is paraphrase-blind**: the score uses exact token overlap, so a paraphrased answer scores low even if semantically grounded.
4. **English-only STOPWORDS**: the stopword list covers only English, making groundedness scores unreliable for non-English content.

## Setup

1. Bring up Weaviate locally:
   ```bash
   docker run -d --name weaviate-int \
     -p 8080:8080 \
     -e DEFAULT_VECTORIZER_MODULE=none \
     -e ENABLE_MODULES= \
     -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
     semitechnologies/weaviate:1.24.10
   ```

2. Install dependencies (cached from the lab — should be fast):
   ```bash
   pip install -r requirements.txt
   ```

3. Ingest the corpus once (idempotent — safe to re-run):
   ```bash
   python ingest.py
   ```

4. Implement `rag_service.py` (the 6 functions: `retrieve`, `build_prompt`,
   `generate`, `rag_pipeline`, `groundedness_score`, `evaluate_rag`).

5. Replace the `pytest.fail(...)` stubs in `tests/test_rag.py` with real
   assertions (3 required tests).

6. Run the autograder locally:
   ```bash
   pytest tests/ -v
   ```

7. Open a PR on branch `integration-8-rag-service`.

## What to Expect from the Canonical Baseline

See the guide's framing block (top of the Integration Task page) for the
expected ranges of the four headline evaluation metrics under the canonical
methodology. The low main-keyword-recall result is **expected** and is a
property of the generator under the abstention prompt — not a bug in your
pipeline.

The Tier 1 challenge formalises this with an intervention experiment.

## PR description must include

- The full `evaluate_rag` output (all four headline metric keys).
- Two example queries with answers + groundedness scores (one high, one low).
- Paste your PR URL into TalentLMS → Module 8 → Integration Task.

## Files

- `rag_service.py` — your implementation (6 functions; `STOPWORDS` and
  `ABSTAIN_PHRASES` constants are pre-populated — do not modify).
- `index_helpers.py` — provided helpers: `index_corpus_if_needed`,
  `bm25_search`, `dense_search`, `hybrid_search` (you don't implement these).
- `ingest.py` — runnable driver (no edits needed).
- `data/corpus.jsonl` — same technical-Q&A corpus as the lab.
- `data/rag_eval.jsonl` — 30-row eval set (`single_fact` / `single_doc_synthesis`
  / `borderline` mix).
- `tests/test_rag.py` — 3 learner-written test stubs (replace `pytest.fail`).
- `tests/test_autograder.py` — 11 autograder tests.
- `LICENSE` + `ATTRIBUTION.md` — corpus license (CC BY-SA) and attribution.

## Resubmissions

Accepted through Saturday of the assignment week.

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms. Corpus content is governed separately by the CC BY-SA notices in [ATTRIBUTION.md](ATTRIBUTION.md) and `data/LICENSE`.