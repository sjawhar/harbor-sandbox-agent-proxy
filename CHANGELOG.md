# Changelog

## Unreleased

- Adds optional per-trial `harbor-growth/1` token accounting, validated whole-request reservations, ordered commits, and durable journal recovery.
- Adds trusted task budget requirements, result classification that retains grades, and offline replay and measurement-gate tools. No budget is enabled by default.
- Refuses structured non-text tool outputs before confinement, preserves stripped server declarations, and keeps close/recovery diagnostics consistent.
- Requires per-profile joined-charge evidence in replay gates and carries closed refusal codes through native errors with a strict message encoding.
- Keeps shutdown independent of close-caller cancellation and credits measured empty-message prefixes consistently with replay.

## 0.1.2

- Records each installed agent caller key before setup and preserves it through relay shutdown so host redaction covers setup-only trials.

## 0.1.1

- The wheel ships a `py.typed` marker, so type checkers read the package's own annotations instead of reporting missing stubs.
- `uv.lock` records the package's own version.

## 0.1.0

- Adds a Harbor relay that keeps model API keys in the host process while an agent runs in a sandbox.
- Provides `RelayAgent` and `RelayEnvironment`, including host methods for routing calls, protecting values, opening an event sink, and selecting bridged MCP tool servers.
- Adds the in-sandbox model proxy, which suspends task tools after the agent turn and records relay activity.
- Adds the `sandbox-agent-proxy` Harbor observer plugin for redacted per-trial outcomes.
- Supports Harbor 0.23.0 through 0.24.x and Inspect AI 0.3.278 or later.
