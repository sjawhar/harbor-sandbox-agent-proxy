# harbor-sandbox-agent-proxy

`harbor-sandbox-agent-proxy` is a public [Harbor](https://github.com/harbor-framework/harbor)
extension for running an agent inside a Harbor sandbox with no model API key in the
sandbox: the agent's model calls are relayed through a host-side process that holds
the key. It provides:

- `RelayAgent` and `RelayEnvironment` import-path wrappers for a Harbor job;
- a sandbox-side model proxy that uses Inspect AI's existing bridge interfaces; and
- a `sandbox-agent-proxy` Harbor observer plugin, which records a redacted
  per-trial outcome at `sandbox-agent-proxy/outcome.json`.

Everything a host decides (where calls go, which key they use, where per-call records
are written, which values are secret) is an injected collaborator with a default that
works for a plain `harbor run`. The package depends on no particular host.

The package supports `inspect-ai>=0.3.278`. It centralizes every private Inspect
import in `_inspect_compat.py`; a moved private name raises an import error that
names the incompatible symbol. It is tested with Inspect AI 0.3.278 and 0.3.285.

## Status

Pre-release. Not yet on PyPI.

## Install

```bash
uv add 'harbor-sandbox-agent-proxy[harbor,providers] @ git+https://github.com/sjawhar/harbor-sandbox-agent-proxy'
```

## Job configuration

Configure both wrappers with the same `relay` value. The package's default
`Upstream` resolves the public provider endpoint on every request and reads only the
corresponding host environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
`GEMINI_API_KEY`). The wrapper removes protected values from the environment it gives
Harbor, refuses to run a trial whose started sandbox environment contains one, and never
writes one to the JSON outcome.

Hosts that use another routing policy provide collaborators through the wrappers'
JSON kwargs. Each collaborator can be an object or a `module:attribute` import path:

- `upstream` implements `async resolve(*, dialect, model) -> UpstreamTarget`;
- `secrets` returns host values that must be excluded and redacted;
- `EventSinkFactory` opens a per-trial record sink; and
- `BridgedToolServerProvider` supplies per-trial MCP server specifications.

The package keeps no process-wide trial registry. A host integration owns its route
policy and any transformation of the package's records into its own log format.

```json
{
  "agents": [
    {
      "import_path": "harbor_sandbox_agent_proxy.agent:RelayAgent",
      "kwargs": {
        "agent": {"import_path": "my_package.agent:MyAgent"},
        "relay": {"declared_models": ["anthropic/claude-sonnet-4-5"]},
        "upstream": "my_package.routing:upstream",
        "secrets": "my_package.routing:secrets"
      }
    }
  ],
  "environment": {
    "import_path": "harbor_sandbox_agent_proxy.environment:RelayEnvironment",
    "kwargs": {
      "environment": {"import_path": "harbor.environments.docker.docker:DockerEnvironment"},
      "relay": {"declared_models": ["anthropic/claude-sonnet-4-5"]},
      "secrets": "my_package.routing:secrets"
    }
  },
  "metadata": {
    "sandbox_agent_proxy": {
      "bridged_mcp_servers": []
    }
  }
}
```


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
