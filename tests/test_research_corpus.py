"""Tests for the AI Research Agent + Regression Memory corpus."""

from __future__ import annotations

import json
import os

import pytest

from cvconform.research import ResearchAgent
from cvconform.corpus import ConformanceCorpus, stable_id


def _evidence(mechanism="precision", backend="coreml"):
    return {
        "backend": backend,
        "mechanism": mechanism,
        "cause": "Precision change (FP32 -> FP16 semi / bf16 / int8)",
        "confidence": 0.9,
        "impact": "Tensor drift on low-level op",
    }


def test_research_agent_produces_summary():
    agent = ResearchAgent(offline_ok=True)
    ev = _evidence("precision", "coreml")
    summary = agent.investigate(ev)
    assert summary.backend == "coreml"
    assert summary.mechanism == "precision"
    assert summary.recommendation, "should always have a recommendation"


def test_research_agent_online_path_uses_sources():
    def fake_search(query):
        return [
            {"title": "coremltools issue #123 FP16 drift",
             "url": "https://github.com/apple/coremltools/issues/123"},
        ]

    agent = ResearchAgent(web_search_fn=fake_search, offline_ok=True)
    summary = agent.investigate(_evidence("precision", "coreml"))
    assert summary.matched_issue == "https://github.com/apple/coremltools/issues/123"
    assert not summary.offline
    assert summary.sources


def test_research_multiple_mechanisms():
    agent = ResearchAgent(offline_ok=True)
    for mech in ["precision", "fusion", "quant_scale", "nms", "padding", "fallback"]:
        summary = agent.investigate(_evidence(mech, "onnx"))
        assert summary.mechanism == mech
        assert summary.recommendation


def test_corpus_roundtrip(tmp_path):
    corpus = ConformanceCorpus(root=str(tmp_path))
    entry = {
        "id": stable_id("test-failure"),
        "backend": "coreml",
        "fault": "precision_fp16",
        "created": "2026-01-01T00:00:00Z",
        "signal": {"score": 87.22, "divergences": [{"metric": "max_abs_err"}]},
    }
    path = corpus.record_failure("coreml", "fp16_conv_bug_001", entry)
    assert os.path.exists(path)

    entries = corpus.list_failures("coreml")
    assert len(entries) == 1
    assert entries[0]["id"] == entry["id"]

    summary = corpus.summarize()
    assert summary["total_entries"] == 1
    assert summary["by_backend"]["coreml"] == 1


def test_corpus_keyed_by_backend(tmp_path):
    corpus = ConformanceCorpus(root=str(tmp_path))
    corpus.record_failure("onnx", "resize_004", {"backend": "onnx", "id": "a"})
    corpus.record_failure("tensorrt", "nms_012", {"backend": "tensorrt", "id": "b"})
    by_onnx = corpus.list_failures("onnx")
    assert len(by_onnx) == 1
    assert by_onnx[0]["backend"] == "onnx"
    s = corpus.summarize()
    assert s["by_backend"] == {"onnx": 1, "tensorrt": 1}
