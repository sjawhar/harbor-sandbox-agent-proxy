"""Tests for the test-assertion parity command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "tools" / "compare_assertions.py"


def _write_test(directory: Path, source: str) -> None:
    directory.mkdir()
    (directory / "test_subject.py").write_text(source, encoding="utf-8")


def _run_compare(
    reference: Path, candidate: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            _SCRIPT,
            "--reference",
            reference,
            "--candidate",
            candidate,
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_reports_fewer_assertions_and_fails(tmp_path: Path) -> None:
    """Fails if a candidate silently drops a reference assertion."""
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_test(
        reference,
        """
def test_preserves_all_checks():
    assert first_check()
    assert second_check()
    assert third_check()
""",
    )
    _write_test(
        candidate,
        """
def test_preserves_all_checks():
    assert first_check()
    assert second_check()
""",
    )

    result = _run_compare(reference, candidate)

    assert result.returncode == 1
    assert "test_preserves_all_checks: FEWER 3→2" in result.stdout


def test_allows_a_named_fewer_assertions_exception(tmp_path: Path) -> None:
    """Permits only the explicitly documented dropped assertion exception."""
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_test(
        reference,
        """
def test_host_specific_gateway_check():
    assert gateway_is_available()
""",
    )
    _write_test(candidate, "def test_host_specific_gateway_check():\n    pass\n")
    allow = tmp_path / "allow.txt"
    allow.write_text(
        "# host behavior stays in hawk\ntest_host_specific_gateway_check\n",
        encoding="utf-8",
    )

    result = _run_compare(reference, candidate, "--allow", str(allow))

    assert result.returncode == 0
    assert "test_host_specific_gateway_check: FEWER 1→0" in result.stdout


def test_compares_pytest_context_manager_checks(tmp_path: Path) -> None:
    """Counts pytest.raises calls that protect an expected failure."""
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    source = """
def test_rejects_undeclared_model():
    with pytest.raises(RuntimeError, match="undeclared"):
        run_undeclared_model()
"""
    _write_test(reference, source)
    _write_test(candidate, source)

    result = _run_compare(reference, candidate)

    assert result.returncode == 0
    assert "test_rejects_undeclared_model: SAME" in result.stdout


def test_reports_a_missing_reference_test(tmp_path: Path) -> None:
    """Fails if a reference test disappears from the candidate suite."""
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_test(reference, "def test_route_is_pinned():\n    assert route()\n")
    _write_test(candidate, "def test_unrelated_behavior():\n    assert True\n")

    result = _run_compare(reference, candidate)

    assert result.returncode == 1
    assert "test_route_is_pinned: MISSING" in result.stdout


def test_normalizes_renamed_identifiers(tmp_path: Path) -> None:
    """Treats a mechanically renamed assertion as the same check."""
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_test(
        reference,
        """
def test_forwards_model_call():
    assert hawk_relay.forward("model") == "ok"
""",
    )
    _write_test(
        candidate,
        """
def test_forwards_model_call():
    assert sandbox_agent_proxy.forward("model") == "ok"
""",
    )

    result = _run_compare(
        reference,
        candidate,
        "--rename",
        "hawk_relay=sandbox_agent_proxy",
    )

    assert result.returncode == 0
    assert "test_forwards_model_call: SAME" in result.stdout
