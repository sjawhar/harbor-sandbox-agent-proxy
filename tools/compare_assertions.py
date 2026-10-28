"""Compare test assertion sources between a reference suite and a candidate suite."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestChecks:
    """The normalized checks made by one test function."""

    name: str
    checks: Counter[str]


def _is_pytest_check(call: ast.Call) -> bool:
    function = call.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "pytest"
        and function.attr in {"raises", "warns"}
    )


def _normalized(source: str, replacements: Sequence[tuple[str, str]]) -> str:
    for old, new in replacements:
        source = source.replace(old, new)
    return source


def _checks(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    replacements: Sequence[tuple[str, str]],
) -> Counter[str]:
    checks: Counter[str] = Counter()
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            checks[_normalized(ast.unparse(node.test), replacements)] += 1
        elif isinstance(node, ast.Call) and _is_pytest_check(node):
            checks[_normalized(ast.unparse(node), replacements)] += 1
    return checks


def _tests(
    directory: Path, replacements: Sequence[tuple[str, str]]
) -> list[TestChecks]:
    tests: list[TestChecks] = []
    for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                tests.append(TestChecks(node.name, _checks(node, replacements)))
    return tests


def _read_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        name
        for line in path.read_text(encoding="utf-8").splitlines()
        if (name := line.partition("#")[0].strip())
    }


def _parse_rename(value: str) -> tuple[str, str]:
    old, separator, new = value.partition("=")
    if not separator or not old or not new:
        raise argparse.ArgumentTypeError("renames must use OLD=NEW")
    return old, new


def _status(reference: TestChecks, candidate: TestChecks | None) -> str:
    if candidate is None:
        return "MISSING"
    if reference.checks == candidate.checks:
        return "SAME"

    reference_count = reference.checks.total()
    candidate_count = candidate.checks.total()
    if candidate_count < reference_count:
        return f"FEWER {reference_count}→{candidate_count}"
    return f"DIFF {reference_count}→{candidate_count}"


def _candidate_by_name(tests: Iterable[TestChecks]) -> dict[str, TestChecks]:
    return {test.name: test for test in tests}


def compare(
    reference_directory: Path,
    candidate_directory: Path,
    replacements: Sequence[tuple[str, str]],
    allowed: set[str],
) -> int:
    """Print each reference comparison and return the parity-gate status."""
    candidates = _candidate_by_name(_tests(candidate_directory, replacements))
    failed = False
    for reference in _tests(reference_directory, replacements):
        status = _status(reference, candidates.get(reference.name))
        print(f"{reference.name}: {status}")
        if (
            status.split(maxsplit=1)[0] in {"FEWER", "MISSING"}
            and reference.name not in allowed
        ):
            failed = True
    return int(failed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--rename", action="append", default=[], type=_parse_rename)
    parser.add_argument("--allow", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the assertion parity comparison command."""
    parsed = _parser().parse_args(arguments)
    return compare(
        parsed.reference, parsed.candidate, parsed.rename, _read_allowlist(parsed.allow)
    )


if __name__ == "__main__":
    raise SystemExit(main())
