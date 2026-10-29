# harbor-sandbox-agent-proxy

`harbor-sandbox-agent-proxy` is a public [Harbor](https://github.com/harbor-framework/harbor)
extension for running an agent inside a Harbor sandbox with no model API key in the
sandbox: the agent's model calls are relayed through a host-side process that holds
the key. It provides:

- `RelayAgent` and `RelayEnvironment` import-path wrappers for a Harbor job;
- a sandbox-side model proxy that uses Inspect AI's existing bridge interfaces; and
- a `sandbox-agent-proxy` Harbor observer plugin, which records a redacted
  per-trial outcome at `sandbox-agent-proxy/outcome.json`.

Everything a host decides is implemented by subclassing `RelayEnvironment`. The
package supplies defaults that work for a plain `harbor run`, so it has no dependency
on a particular host.

The package supports `inspect-ai>=0.3.278`. It centralizes every private Inspect
import in `_inspect_compat.py`; a moved private name raises an import error that names
the incompatible symbol. It is tested with Inspect AI 0.3.278 and 0.3.285, and with
Harbor 0.23.0 and 0.24.0.

## Status

Pre-release. Not yet on PyPI.

## Install

```bash
uv add 'harbor-sandbox-agent-proxy[harbor,providers] @ git+https://github.com/sjawhar/harbor-sandbox-agent-proxy'
```

## Hosting the relay

Configure both wrappers with the same JSON `relay` object. Its values are data only:
the host names the wrapper classes with import paths, but does not pass
`module:attribute` collaborators. Use the package's `RelayEnvironment` for the
default host, or name a subclass that overrides the host seam:

```json
{
  "agents": [
    {
      "import_path": "harbor_sandbox_agent_proxy.agent:RelayAgent",
      "kwargs": {
        "agent": {"import_path": "my_package.agent:MyAgent"},
        "relay": {"declared_models": ["anthropic/claude-sonnet-4-5"]}
      }
    }
  ],
  "environment": {
    "import_path": "my_package.host:MyRelayEnvironment",
    "kwargs": {
      "environment": {
        "import_path": "harbor.environments.docker.docker:DockerEnvironment"
      },
      "relay": {"declared_models": ["anthropic/claude-sonnet-4-5"]}
    }
  }
}
```

`RelayEnvironment` defines four methods a host can override:

* `route(call)` selects the permitted call's `Route`. The default matches a declared
  model, reads `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or the first available Google
  key (`GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_GENERATIVE_AI_API_KEY`), and
  forwards to the first configured provider base URL
  (`ANTHROPIC_BASE_URL`; `OPENAI_BASE_URL` or `OPENAI_API_BASE`; or
  `GOOGLE_BASE_URL`) or its public endpoint. It raises `RouteRefused` for a model that
  is not declared and `RuntimeError` when its provider has no host key. A host may
  raise `RouteRefused` to deny a call; other route errors propagate.
* `protected_values()` returns values that must neither enter the sandbox nor appear
  in records. The default returns the configured default-route keys.
* `open_event_sink(trial)` opens the destination for the trial's Inspect events. The
  default writes the redacted JSON outcome at
  `sandbox-agent-proxy/outcome.json`; sink I/O failures propagate.
* `bridged_tool_servers()` returns task-local MCP servers. The default reads
  `[metadata.sandbox_agent_proxy].bridged_mcp_servers` from the task's `task.toml`;
  malformed TOML or metadata raises `ValueError`.

The environment instance holds all per-trial state. `caller_keys` is the set of minted
caller keys collected when the relay closes, and `outcome` is its `RelayOutcome`. A
custom event sink can read both when it closes the outcome. Inspect event metadata is
stored under the `sandbox_agent_proxy` key.

```toml
[metadata.sandbox_agent_proxy]
bridged_mcp_servers = [
  { name = "task-tools", command = "python3", args = ["/opt/tools.py"] }
]
```

An agent whose own loop runs on the host constructs its model client before Harbor
creates the environment. Its `api_base` is therefore the job composer’s responsibility
to set; the package does not derive it from the relay.

Run Harbor with the observer plugin:

```bash
harbor run --config job.json --yes --plugin sandbox-agent-proxy
```

The included e2e suite drives this command against Harbor's Docker environment and
checks the recorded outcome.

## Development

```bash
uv sync --all-extras --group dev
uv run pytest -n 4 -q
uv run pytest tests/e2e --e2e -m e2e -n 2 --durations=0
```
