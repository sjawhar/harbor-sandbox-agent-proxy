"""Independent offline oracle, never imported by production accounting.

Input is JSONL. G1 requires each row's frozen model/dialect/profile/credit and
joined committed_charge. Arithmetic-only exploration cannot pass G1. Measurement
modes consume frozen export rows, not canned SDK fixtures. Exit zero means this
artifact passed that check; it never enables a deployment or changes a profile.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _key(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _message(value, dialect):
    if not isinstance(value, dict):
        return value
    value = dict(value)
    if dialect in ("anthropic", "openai") and isinstance(value.get("content"), str):
        value["content"] = [{"type": "text", "text": value["content"]}]
    elif dialect == "anthropic" and isinstance(value.get("content"), list):
        value["content"] = [{k: v for k, v in x.items() if k != "cache_control"} if isinstance(x, dict) else x for x in value["content"]]
    return value


def _root(value, dialect):
    if dialect != "anthropic" or not isinstance(value, dict):
        return value
    value = dict(value)
    for field in ("system", "tools"):
        if isinstance(value.get(field), list):
            value[field] = [{k: v for k, v in item.items() if k != "cache_control"}
                            if isinstance(item, dict) else item for item in value[field]]
    return value


def replay(rows, credit="eligible", *, arithmetic_only=False):
    checkpoints, profile_policies = {}, {}
    charges, differences, comparisons = [], [], []
    aliases = {"messages": "anthropic", "chat-completions": "openai", "responses": "openai-responses", "gemini": "google"}
    for expected, row in enumerate(sorted(rows, key=lambda x: x["admit_seq"]), 1):
        if type(row["admit_seq"]) is not int or row["admit_seq"] != expected:
            raise ValueError("admission sequence must be complete and unique")
        dialect = aliases.get(row.get("dialect"), row.get("dialect"))
        identity_ok = all(isinstance(row.get(k), str) and row[k].strip() for k in ("model", "dialect", "profile"))
        identity_ok = identity_ok and dialect in ("anthropic", "openai", "openai-responses", "google") and row.get("credit") in ("none", "eligible")
        if not identity_ok:
            differences.append(dict(admit_seq=expected, reason="missing_frozen_profile"))
        policy = row.get("credit", credit)
        if policy not in ("none", "eligible"):
            policy = "none"
        profile_key = (row.get("model"), dialect, row.get("profile")) if identity_ok else None
        if profile_key is not None:
            previous = profile_policies.setdefault(profile_key, policy)
            if previous != policy:
                differences.append(dict(admit_seq=expected, reason="profile_policy_changed"))
        root = _key([row.get("model"), dialect, row.get("profile"), _root(row["root"], dialect)])
        messages = tuple(_key(_message(m, dialect)) for m in row["messages"])
        n, o, w = row["n"], row["o"], row.get("w", 0)
        if any(type(x) is not int or x < 0 for x in (n, o, w)):
            raise ValueError("oracle requires measured counts, never invented zeros")
        matches = [(len(prefix), count) for (r, prefix), count in checkpoints.items()
                   if r == root and messages[:len(prefix)] == prefix]
        k = max(matches, default=(0, 0))[1] if policy == "eligible" else 0
        if n < k:
            k = 0
        charge = n - k + o + w
        charges.append(charge)
        checkpoints.setdefault((root, messages), n)
        if type(row.get("committed_charge")) is not int or row["committed_charge"] < 0:
            differences.append(dict(admit_seq=expected, reason="missing_committed_charge"))
        elif row["committed_charge"] != charge:
            differences.append(dict(admit_seq=expected, expected=charge, actual=row["committed_charge"]))
        if "old_charge" in row and row["old_charge"] != charge:
            if not row.get("difference_cause"):
                differences.append(dict(admit_seq=expected, reason="unclassified_old_ledger_difference"))
            comparisons.append(dict(admit_seq=expected, old=row["old_charge"], expected=charge,
                                    cause=row.get("difference_cause")))
    return dict(passed=bool(rows) and not differences and not arithmetic_only,
                mode="arithmetic_only" if arithmetic_only else "g1", charges=charges, spent=sum(charges),
                differences=differences, old_ledger_differences=comparisons)


def stability(rows, tolerance=0):
    failures, comparable = [], []
    if type(tolerance) is not int or tolerance < 0:
        failures.append("invalid_framing_tolerance")
        tolerance = 0
    if len(rows) < 30:
        failures.append("fewer_than_30_calls")
    for row in rows:
        if type(row.get("n")) is not int or row["n"] <= 0 or type(row.get("tail_tokens")) is not int or row["tail_tokens"] < 0:
            failures.append("unknown_count")
        if any(type(row.get(k)) is not bool for k in ("extends_previous", "thinking_present", "thinking_dropped")):
            failures.append("invalid_coverage_flags")
    for previous, current in zip(rows, rows[1:]):
        if current.get("extends_previous") is True:
            if any(type(row.get("n")) is not int or row["n"] <= 0 for row in (previous, current)) or type(current.get("tail_tokens")) is not int or current["tail_tokens"] < 0:
                continue
            if current["tail_tokens"] > 0:
                comparable.append(current)
            if current["n"] < previous["n"] or current["n"] - previous["n"] < current["tail_tokens"] - tolerance:
                failures.append("prefix_rendering_changed")
    if not comparable:
        failures.append("no_comparable_extensions")
    if not all(any(row.get(key) is True for row in comparable) for key in ("thinking_present", "thinking_dropped")):
        failures.append("missing_comparable_thinking_coverage")
    return dict(passed=not failures, calls=len(rows), comparable_pairs=len(comparable),
                credit="none" if failures else "eligible", failures=failures)


def shadow_bound(rows):
    failures, profiles = [], {}
    if not rows:
        failures.append("no_calls")
    for row in rows:
        if any(row.get(key) is not True for key in ("real_route", "successful", "terminal")) or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("profile", "bound_source", "bound_version", "route_bound_source", "route_bound_version")):
            failures.append("missing_real_usage_or_bound_identity")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in ("n", "o", "r_in", "final_body_bytes")):
            failures.append("missing_counts")
        elif row["n"] <= 0:
            failures.append("no_nonzero_prompt_measurement")
        elif row["n"] > row["r_in"] or row["final_body_bytes"] <= 0 or row["r_in"] < row["final_body_bytes"]:
            failures.append("input_bound_violated")
        if row.get("opaque_reasoning") and not row.get("effective_input_contract"):
            failures.append("unbounded_opaque_input")
        profile = row.get("profile")
        if isinstance(profile, str):
            profiles.setdefault(profile, []).append(row)
    for group in profiles.values():
        if not all(any(row.get(key) is True for row in group) for key in ("tools_transition", "thinking_transition")):
            failures.append("missing_profile_transition_coverage")
    return dict(passed=not failures, calls=len(rows), failures=failures)


def comparison(rows):
    failures = []
    attempts = {}
    residuals = []
    for row in rows:
        attempts.setdefault(row["attempt"], []).append(row)
        if not row.get("selection_frozen") or not row.get("versions_frozen") or row.get("worker_attribution") != "matched":
            failures.append("unfrozen_or_unmatched")
        if any(type(row.get(k)) is not int for k in ("platform_charge", "candidate_charge")):
            failures.append("unidentifiable_component")
            continue
        residuals.append(row["candidate_charge"] - row["platform_charge"])
    groups = list(attempts.values())
    if sum(any(r.get("reset") for r in g) for g in groups) < 5:
        failures.append("missing_reset_coverage")
    if sum(any(r.get("workers") for r in g) for g in groups) < 2:
        failures.append("missing_worker_coverage")
    if sum(any(r.get("non_anthropic") for r in g) for g in groups) < 2:
        failures.append("missing_dialect_coverage")
    heldout = [g for g in groups if all(r.get("held_out") for r in g)]
    if not groups or len(heldout) * 3 < len(groups):
        failures.append("missing_heldout_third")
    for group in heldout:
        platform = sum(r.get("platform_charge", 0) for r in group)
        candidate = sum(r.get("candidate_charge", 0) for r in group)
        if platform <= 0 or abs(candidate - platform) > .01 * platform:
            failures.append("heldout_disagreement")
    if residuals and (all(r >= 0 for r in residuals) or all(r <= 0 for r in residuals)) and any(residuals):
        failures.append("systematic_residual_sign")
    return dict(passed=bool(rows) and not failures, residuals=residuals, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "m1", "m2", "m3"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--credit", choices=("none", "eligible"), default="none", help="arithmetic-only fallback; G1 uses each row's frozen credit policy")
    parser.add_argument("--arithmetic-only", action="store_true", help="calculate without certifying G1; always exits nonzero")
    parser.add_argument("--framing-tolerance", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    fn = {"replay": lambda: replay(rows, args.credit, arithmetic_only=args.arithmetic_only), "m1": lambda: comparison(rows),
          "m2": lambda: stability(rows, args.framing_tolerance), "m3": lambda: shadow_bound(rows)}[args.mode]
    result = fn()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
