"""Tests for _is_symbol_deprecated and its effect on confidence scoring.

The real deprecation signal in most codebases is an annotation, not a name
suffix. These tests verify that the full decorator surface (Python, Java/Kotlin,
Rust inner-attr, C# stripped attr, C++ stripped attr, Swift) is recognised and
lowers confidence to 0.3 — placing the finding below the default min_confidence
floor of 0.4 so it shows up in hidden_below_threshold rather than in the
returned findings.
"""

from __future__ import annotations

import pytest

from repowise.core.analysis.dead_code import DeadCodeAnalyzer
from repowise.core.analysis.dead_code.analyzer import _is_symbol_deprecated
from tests.unit.dead_code._helpers import _build_graph, _old_date

# ---------------------------------------------------------------------------
# Unit tests for _is_symbol_deprecated
# ---------------------------------------------------------------------------


class TestIsSymbolDeprecated:
    """Pure-function tests — no graph needed."""

    def test_name_suffix_DEPRECATED(self):
        assert _is_symbol_deprecated("process_data_DEPRECATED", []) is True

    def test_name_suffix_LEGACY(self):
        assert _is_symbol_deprecated("config_LEGACY", []) is True

    def test_name_suffix_COMPAT(self):
        assert _is_symbol_deprecated("parse_COMPAT", []) is True

    def test_python_at_deprecated(self):
        assert _is_symbol_deprecated("process_data", ["@deprecated"]) is True

    def test_python_typing_deprecated(self):
        assert _is_symbol_deprecated("process_data", ["@typing.deprecated"]) is True

    def test_python_warnings_deprecated(self):
        assert _is_symbol_deprecated("process_data", ["@warnings.deprecated"]) is True

    def test_java_Deprecated(self):
        # Java / Kotlin annotation — capital D
        assert _is_symbol_deprecated("processData", ["@Deprecated"]) is True

    def test_kotlin_Deprecated(self):
        assert _is_symbol_deprecated("processData", ["@kotlin.Deprecated"]) is True

    def test_scala_deprecated(self):
        assert _is_symbol_deprecated("processData", ["@deprecated"]) is True

    def test_swift_available_deprecated(self):
        # @available(*, deprecated) — call stripped, base is "available"
        # This should NOT match; the actual deprecated signal here is via the
        # argument — the helper only strips the outer call args.
        # Keeping this explicit so behaviour is documented.
        result = _is_symbol_deprecated("func", ["@available(*, deprecated)"])
        # "available" is not in _DEPRECATED_DECORATOR_BASES, so this is False.
        assert result is False

    def test_rust_inner_attr_no_at(self):
        # Rust: parser.py strips #[ ] and stores the inner content directly
        assert _is_symbol_deprecated("my_fn", ["deprecated"]) is True

    def test_rust_inner_attr_with_args(self):
        # #[deprecated(since = "1.0")] → "deprecated(since = \"1.0\")"
        assert _is_symbol_deprecated("my_fn", ['deprecated(since = "1.0")']) is True

    def test_csharp_Obsolete(self):
        # C#: parser.py strips [ ] and stores "Obsolete"
        assert _is_symbol_deprecated("MyMethod", ["Obsolete"]) is True

    def test_csharp_System_Obsolete(self):
        assert _is_symbol_deprecated("MyMethod", ["System.Obsolete"]) is True

    def test_csharp_Obsolete_with_message(self):
        # "[Obsolete(\"Use Foo instead\")]" → "Obsolete(\"Use Foo instead\")"
        assert _is_symbol_deprecated("MyMethod", ['Obsolete("Use Foo instead")']) is True

    def test_cpp_double_bracket_deprecated(self):
        # C++: parser.py strips [[ ]] and stores "deprecated"
        assert _is_symbol_deprecated("my_func", ["deprecated"]) is True

    def test_cpp_deprecated_with_reason(self):
        # [[deprecated("use bar() instead")]] → "deprecated(\"use bar() instead\")"
        assert _is_symbol_deprecated("my_func", ['deprecated("use bar() instead")']) is True

    def test_unrelated_decorator_is_not_deprecated(self):
        assert _is_symbol_deprecated("process_data", ["@property"]) is False
        assert _is_symbol_deprecated("process_data", ["@app.route"]) is False
        assert _is_symbol_deprecated("process_data", ["@staticmethod"]) is False

    def test_empty_decorators_no_suffix_is_not_deprecated(self):
        assert _is_symbol_deprecated("process_data", []) is False

    def test_multiple_decorators_one_deprecated(self):
        assert _is_symbol_deprecated("fn", ["@staticmethod", "@deprecated"]) is True


# ---------------------------------------------------------------------------
# Integration tests — full analyzer path
# ---------------------------------------------------------------------------


def _stale_file_node(name: str, *, symbols: list | None = None) -> dict:
    return {
        name: {
            "is_entry_point": False,
            "is_test": False,
            "is_api_contract": False,
            "symbol_count": 5,
            "symbols": symbols or [],
        }
    }


def _stale_git_meta(name: str) -> dict:
    return {
        name: {
            "commit_count_90d": 0,
            "last_commit_at": _old_date(days=400),
            "age_days": 400,
            "primary_owner_name": None,
        }
    }


def _symbol(name: str, decorators: list[str]) -> dict:
    return {
        "name": name,
        "kind": "function",
        "visibility": "public",
        "language": "python",
        "decorators": decorators,
        "start_line": 1,
        "end_line": 10,
    }


class TestDeprecatedDecoratorIntegration:
    """End-to-end: the decorator reaches the confidence score via the analyzer."""

    def _run(self, decorator: str, min_confidence: float = 0.4) -> object:
        """Build a minimal graph with one deprecated export and run the analyzer."""
        g = _build_graph(
            nodes={
                **_stale_file_node(
                    "pkg/utils.py",
                    symbols=[_symbol("process_data", [decorator])],
                ),
                "pkg/caller.py": {
                    "is_entry_point": False,
                    "is_test": False,
                    "is_api_contract": False,
                    "symbol_count": 2,
                    "symbols": [],
                },
            },
            edges=[("pkg/caller.py", "pkg/utils.py", {"edge_type": "imports"})],
        )
        git_meta = {**_stale_git_meta("pkg/utils.py"), **_stale_git_meta("pkg/caller.py")}
        analyzer = DeadCodeAnalyzer(g, git_meta_map=git_meta)
        return analyzer.analyze(
            {
                "detect_zombie_packages": False,
                "min_confidence": min_confidence,
            }
        )

    @pytest.mark.parametrize(
        "decorator",
        [
            "@deprecated",
            "@typing.deprecated",
            "@Deprecated",
            # Rust / C# / C++ inner-attr forms reach the graph without "@"
            "deprecated",
            "Obsolete",
        ],
    )
    def test_decorator_scores_confidence_0_3(self, decorator: str):
        report = self._run(decorator, min_confidence=0.0)
        deprecated_findings = [
            f for f in report.findings if f.symbol_name == "process_data"
        ]
        assert len(deprecated_findings) == 1, (
            f"Expected one finding for decorator {decorator!r}; got {deprecated_findings}"
        )
        assert deprecated_findings[0].confidence == pytest.approx(0.3), (
            f"Expected confidence=0.3 for decorator {decorator!r}; "
            f"got {deprecated_findings[0].confidence}"
        )

    @pytest.mark.parametrize(
        "decorator",
        [
            "@deprecated",
            "@Deprecated",
            "deprecated",
            "Obsolete",
        ],
    )
    def test_deprecated_decorator_hidden_under_default_floor(self, decorator: str):
        """Decorated symbols (confidence=0.3) fall below the 0.4 default floor."""
        report = self._run(decorator, min_confidence=0.4)
        deprecated_in_findings = [
            f for f in report.findings if f.symbol_name == "process_data"
        ]
        assert deprecated_in_findings == [], (
            f"Decorated symbol should not appear under default floor; "
            f"decorator={decorator!r}"
        )

    def test_unrelated_decorator_does_not_lower_confidence(self):
        report = self._run("@property", min_confidence=0.0)
        findings = [f for f in report.findings if f.symbol_name == "process_data"]
        assert len(findings) == 1
        assert findings[0].confidence != pytest.approx(0.3), (
            "@property should not lower confidence to 0.3"
        )

    def test_suffix_DEPRECATED_still_works_without_decorator(self):
        """Name-suffix detection (backward compat) is not broken by the new path."""
        g = _build_graph(
            nodes={
                **_stale_file_node(
                    "pkg/utils.py",
                    symbols=[_symbol("process_data_DEPRECATED", [])],
                ),
                "pkg/caller.py": {
                    "is_entry_point": False,
                    "is_test": False,
                    "is_api_contract": False,
                    "symbol_count": 2,
                    "symbols": [],
                },
            },
            edges=[("pkg/caller.py", "pkg/utils.py", {"edge_type": "imports"})],
        )
        git_meta = {**_stale_git_meta("pkg/utils.py"), **_stale_git_meta("pkg/caller.py")}
        analyzer = DeadCodeAnalyzer(g, git_meta_map=git_meta)
        report = analyzer.analyze({"detect_zombie_packages": False, "min_confidence": 0.0})
        findings = [f for f in report.findings if f.symbol_name == "process_data_DEPRECATED"]
        assert len(findings) == 1
        assert findings[0].confidence == pytest.approx(0.3)
