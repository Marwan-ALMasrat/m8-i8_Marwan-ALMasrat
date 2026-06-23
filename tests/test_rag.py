"""Learner-written tests for the RAG pipeline.

Replace each `pytest.fail("Not implemented...")` body with a real test that
uses `assert` statements. The autograder includes an AST meta-check
(test_learner_tests_complete) that verifies (a) at least 3 test functions
exist, (b) each has at least 1 ast.Assert node, and (c) no `pass` body and
no `pytest.fail("Not implemented")` placeholder remain.

Hint: import the functions you want to test from rag_service. The Weaviate
service must be running locally (the autograder workflow brings it up; for
local runs, `docker run` Weaviate first and `python ingest.py`).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rag_service import retrieve, build_prompt, groundedness_score


def test_retrieve_returns_k_or_fewer():
    """retrieve(query, k=N) should return at most N items, each a dict."""
    results = retrieve("how to use git", k=5)
    assert isinstance(results, list)
    assert len(results) <= 5
    for item in results:
        assert "doc_id" in item
        assert "title" in item
        assert "answer_text" in item


def test_build_prompt_includes_context():
    """build_prompt(query, contexts) should include all context titles in the
    rendered prompt and the literal string 'Question:' before the query."""
    query = "What is Python?"
    contexts = [
        {"doc_id": f"id{i}", "title": f"Title{i}", "answer_text": f"Answer text {i}"}
        for i in range(1, 6)
    ]
    prompt = build_prompt(query, contexts)
    assert "Answer the question using only the context" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "[1] Title1:" in prompt
    assert "[5] Title5:" in prompt


def test_groundedness_zero_for_unrelated_answer():
    """groundedness_score(unrelated_answer, contexts) should be ~ 0.0."""
    answer = "xyzzy plugh quux"
    contexts = [{"answer_text": "the quick brown fox jumps"}]
    score = groundedness_score(answer, contexts)
    assert score <= 0.1