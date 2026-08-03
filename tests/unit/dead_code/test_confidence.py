"""Unit tests for DeadCodeAnalyzer."""
# NOTE: test_confidence_summary_low_count_not_always_zero is a regression
# test for a bug where confidence_summary["low"] was permanently 0 because
# the bucket counters ran *after* the min_confidence filter instead of before.

from __future__ import annotations

from datetime import timedelta

import pytest

from repowise.core.analysis.dead_code import (
    DeadCodeAnalyzer,
)
from tests.unit.dead_code._helpers import _build_graph, _now, _old_date


def test_confidence_low_for_recent_files():
    """Unreachable file with commit_count_90d > 0 should have confidence 0.4."""
    g = _build_graph(
        nodes={
            "pkg/recent.py": {
                "is_entry_point": False,
                "is_test": False,
                "is_api_contract": False,
                "symbol_count": 5,
                "symbols": [],
            },
        },
    )

    git_meta = {
        "pkg/recent.py": {
            "commit_count_90d": 3,
            "last_commit_at": _now() - timedelta(days=10),
            "age_days": 100,
            "primary_owner_name": "dev@example.com",
        },
    }

    analyzer = DeadCodeAnalyzer(g, git_meta_map=git_meta)
    report = analyzer.analyze(
        {
            "detect_unused_exports": False,
            "detect_zombie_packages": False,
            "min_confidence": 0.0,
        }
    )

    findings = [f for f in report.findings if f.file_path == "pkg/recent.py"]
    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(0.4)


def test_confidence_high_for_stale_unreachable():
    """Unreachable file with no commits in 90d and last commit > 6 months ago -> confidence 1.0."""
    g = _build_graph(
        nodes={
            "pkg/stale.py": {
                "is_entry_point": False,
                "is_test": False,
                "is_api_contract": False,
                "symbol_count": 5,
                "symbols": [],
            },
        },
    )

    git_meta = {
        "pkg/stale.py": {
            "commit_count_90d": 0,
            "last_commit_at": _old_date(days=365),
            "age_days": 730,
            "primary_owner_name": "dev@example.com",
        },
    }

    analyzer = DeadCodeAnalyzer(g, git_meta_map=git_meta)
    report = analyzer.analyze(
        {
            "detect_unused_exports": False,
            "detect_zombie_packages": False,
        }
    )

    findings = [f for f in report.findings if f.file_path == "pkg/stale.py"]
    assert len(findings) == 1
    assert findings[0].confidence == pytest.approx(1.0)


def test_confidence_summary_low_count_not_always_zero():
    """confidence_summary['low'] must reflect deprecated findings even when
    min_confidence (default 0.4) filters them out of the returned findings list.

    Regression: the bucket counters previously ran *after* the min_confidence
    filter, making low permanently 0 on default analyze() calls.
    """
    # A deprecated public symbol in a file that has an importer gets
    # confidence = 0.3 (< 0.4) from _detect_unused_exports. With the default
    # min_confidence=0.4 it is stripped from report.findings, but it should
    # still appear in confidence_summary["low"].
    g = _build_graph(
        nodes={
            "pkg/importer.py": {
                "is_entry_point": False,
                "is_test": False,
                "is_api_contract": False,
                "symbol_count": 0,
                "symbols": [],
            },
            "pkg/utils.py": {
                "is_entry_point": False,
                "is_test": False,
                "is_api_contract": False,
                "symbol_count": 1,
                "symbols": [
                    {
                        "name": "process_data_DEPRECATED",
                        "kind": "function",
                        "visibility": "public",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 5,
                    }
                ],
            },
        },
        # importer.py imports utils.py as a module but NOT the deprecated
        # symbol by name, so has_importers for the symbol is False.
        edges=[
            ("pkg/importer.py", "pkg/utils.py", {"edge_type": "imports", "imported_names": []}),
        ],
    )

    analyzer = DeadCodeAnalyzer(g, git_meta_map={})

    # Default call: min_confidence=0.4 filters out the 0.3-confidence finding.
    report = analyzer.analyze({"detect_unreachable_files": False, "detect_zombie_packages": False})

    # The finding must NOT appear in the returned findings list (filtered out).
    deprecated_findings = [
        f for f in report.findings if f.symbol_name == "process_data_DEPRECATED"
    ]
    assert deprecated_findings == [], (
        "Deprecated finding should be filtered from report.findings at default min_confidence=0.4"
    )

    # But it MUST be counted in the low bucket of the summary.
    assert report.confidence_summary["low"] >= 1, (
        "confidence_summary['low'] was 0 — bucket counters ran after the min_confidence "
        "filter and the deprecated finding was silently dropped from the summary"
    )
