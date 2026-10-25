# harbor-sandbox-agent-proxy

`harbor-sandbox-agent-proxy` is a public [Harbor](https://github.com/METR/harbor)
extension for running an agent inside a Harbor sandbox while relaying its model calls
through a host-side gateway. It provides:

- `RelayAgent` and `RelayEnvironment` import-path wrappers for a Harbor job;
- a sandbox-side model proxy that uses Inspect AI's existing bridge interfaces;
- a `sandbox-agent-proxy` Harbor observer plugin, which records a redacted
  per-trial outcome at `sandbox-agent-proxy/outcome.json`; and
- reusable Docker e2e fixtures for Messages- and Responses-dialect agents.

The package supports `inspect-ai>=0.3.278`. It centralizes every private Inspect
import in `_inspect_compat.py`; a moved private name raises an import error that
names the incompatible symbol. Version 0.1.0 is tested with Inspect AI 0.3.278
through 0.3.285.

## Install

```bash
uv add 'harbor-sandbox-agent-proxy[harbor,providers]'
```

## Job configuration

Configure both wrappers with the same `relay` value. Each declared model's
provider (or provider/service) needs a gateway route. The API key is read from the
host process environment when a model call is made and is never passed to the
sandbox.

```json
{
  "agents": [
    {
      "import_path": "harbor_sandbox_agent_proxy.agent:RelayAgent",
      "kwargs": {
        "agent": {"import_path": "my_package.agent:MyAgent"},
        "relay": {
          "declared_models": ["anthropic/claude-sonnet-4-5"],
          "gateway": {
            "routes": {
              "anthropic": {
                "base_url": "https://gateway.example/v1",
                "api_key_env": "GATEWAY_API_KEY"
              }
            }
          }
        }
      }
    }
  ],
  "environment": {
    "import_path": "harbor_sandbox_agent_proxy.environment:RelayEnvironment",
    "kwargs": {
      "environment": {"import_path": "harbor.environments.docker.docker:DockerEnvironment"},
      "relay": {
        "declared_models": ["anthropic/claude-sonnet-4-5"],
        "gateway": {
          "routes": {
            "anthropic": {
              "base_url": "https://gateway.example/v1",
              "api_key_env": "GATEWAY_API_KEY"
            }
          }
        }
      }
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
checks the recorded outcome and verifier results.

## Development

```bash
uv sync --all-extras --group dev
uv run pytest -n 4 -q
uv run pytest tests/e2e --e2e -m e2e -n 2 --durations=0
```
