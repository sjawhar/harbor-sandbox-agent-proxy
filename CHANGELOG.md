# Changelog

## 0.1.0

- Adds a Harbor relay that keeps model API keys in the host process while an agent runs in a sandbox.
- Provides `RelayAgent` and `RelayEnvironment`, including host methods for routing calls, protecting values, opening an event sink, and selecting bridged MCP tool servers.
- Adds the in-sandbox model proxy, which suspends task tools after the agent turn and records relay activity.
- Adds the `sandbox-agent-proxy` Harbor observer plugin for redacted per-trial outcomes.
- Supports Harbor 0.23.0 through 0.24.x and Inspect AI 0.3.278 or later.
