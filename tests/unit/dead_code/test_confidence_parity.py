"""Cross-language contract test for dead-code confidence threshold alignment.

Verifies that TypeScript's DEAD_CODE_CONFIDENCE.MEDIUM in packages/types/src/dead-code.ts
matches Python's RISK_CAP_CONFIDENCE in repowise.core.analysis.dead_code.risk_factors.
"""

from __future__ import annotations

import re
from pathlib import Path

from repowise.core.analysis.dead_code.risk_factors import (
    RISK_CAP_CONFIDENCE,
    SAFE_CONFIDENCE_THRESHOLD,
)


def test_ts_confidence_floor_matches_python_risk_cap() -> None:
    """Ensure DEAD_CODE_CONFIDENCE.MEDIUM in TypeScript matches RISK_CAP_CONFIDENCE (0.4)."""
    repo_root = Path(__file__).parents[3]
    ts_file = repo_root / "packages" / "types" / "src" / "dead-code.ts"
    assert ts_file.exists(), f"TypeScript types file not found at {ts_file}"

    content = ts_file.read_text(encoding="utf-8")

    medium_match = re.search(
        r"DEAD_CODE_CONFIDENCE\s*=\s*\{[\s\S]*?MEDIUM:\s*([0-9.]+)", content
    )
    assert medium_match is not None, "Could not parse DEAD_CODE_CONFIDENCE.MEDIUM from dead-code.ts"
    ts_medium = float(medium_match.group(1))

    assert ts_medium == RISK_CAP_CONFIDENCE, (
        f"Cross-language drift detected! TypeScript DEAD_CODE_CONFIDENCE.MEDIUM is {ts_medium}, "
        f"but Python RISK_CAP_CONFIDENCE is {RISK_CAP_CONFIDENCE}."
    )


def test_ts_confidence_high_matches_python_safe_threshold() -> None:
    """Ensure DEAD_CODE_CONFIDENCE.HIGH in TypeScript matches SAFE_CONFIDENCE_THRESHOLD (0.7)."""
    repo_root = Path(__file__).parents[3]
    ts_file = repo_root / "packages" / "types" / "src" / "dead-code.ts"
    assert ts_file.exists(), f"TypeScript types file not found at {ts_file}"

    content = ts_file.read_text(encoding="utf-8")

    high_match = re.search(
        r"DEAD_CODE_CONFIDENCE\s*=\s*\{[\s\S]*?HIGH:\s*([0-9.]+)", content
    )
    assert high_match is not None, "Could not parse DEAD_CODE_CONFIDENCE.HIGH from dead-code.ts"
    ts_high = float(high_match.group(1))

    assert ts_high == SAFE_CONFIDENCE_THRESHOLD, (
        f"Cross-language drift detected! TypeScript DEAD_CODE_CONFIDENCE.HIGH is {ts_high}, "
        f"but Python SAFE_CONFIDENCE_THRESHOLD is {SAFE_CONFIDENCE_THRESHOLD}."
    )
