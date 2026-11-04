# Token-budget operating gates

## State and limits

M1, M2, M3 and G1–G5 have **not been run** by this implementation. No live model
calls, deployment, enforced canary, sync-sidecar isolation test, or platform parity
measurement is authorized by passing the offline suite.

The cap assumes the selected provider/gateway contracts actually bound input,
all candidates' output and outside-output thinking. Measured counterexamples
fence the ledger; they do not undo an overrun. E1 remains unconfirmed: the stored
Claude fixture has a 1,576-byte first request and reports 9,838 prompt tokens.
It cannot establish a bound on the final wire body of the deployed route. Never
fit an injection allowance to those observations.

Native Google caller authentication (F3) remains open on the inspected bridge.
This package does not repair its dropped request headers. Do not claim it fixed
from an offline direct-handler test. Do not remove a lane's old relay until that
lane passes G1–G5. M1 alone permits a platform-comparison claim. M2 alone permits an
empirical credit-eligible profile for its frozen model/mode.

## Offline regression commands

Run in this checkout, with its frozen optional dependencies installed. Each Python
process is memory-capped and niced. `DOCKER_CONFIG` is a fresh empty directory.
The native renderer controls additionally require the design-captured Inspect
0.3.285 `proxy.py` as `HSAP_NATIVE_PROXY_SOURCE`. Its SHA256 must be
`7c1ea740e18e3c3c7e8599afcf2f4a5e8c51e96460f7f3f0c3f06dcdeee3a449`.
Without that artifact the two native-loopback controls skip; a skipped control
is not a passed native-encoding check. They exercise actual HTTP/SSE rendering,
not a deployed CLI or sandbox-service queue.

```sh
( ulimit -v 6291456; nice -n 19 uv sync --all-extras --group dev --frozen )
: "${HSAP_NATIVE_PROXY_SOURCE:?design-pinned Inspect 0.3.285 proxy.py}"
( ulimit -v 6291456; DOCKER_CONFIG=$(mktemp -d) HSAP_NATIVE_PROXY_SOURCE="$HSAP_NATIVE_PROXY_SOURCE" nice -n 19 uv run pytest )
( ulimit -v 6291456; DOCKER_CONFIG=$(mktemp -d) HSAP_NATIVE_PROXY_SOURCE="$HSAP_NATIVE_PROXY_SOURCE" nice -n 19 uv run pytest tests/test_budget*.py )
( ulimit -v 6291456; DOCKER_CONFIG=$(mktemp -d) BUDGET_RULES=v2 nice -n 19 uv run pytest tests/test_budget.py -k '71 or 72 or 74 or 79' )
```

The last command is an intentional negative control: it must exit 1 with **four
behavioral failures**, not import/config/collection errors. The test-only v2 stub
omits paid roots, admits negative holds, requires an exhaustion witness for
natural success, and commits in response order. The same four tests without
`BUDGET_RULES=v2` must pass. The descending duplicate example is **600**, not 300.

These are offline controls. Original design tests 54, 56, 58, 64, 66–67, 69–70
require the actual lane CLI, builder, native queue, result reader, Job or complete
real trajectory; their lane-facing acceptance remains unexecuted here. Test 88's
synthetic bound detector does not execute real M3. Deleted/superseded tests (15,
26, 39–40, 44, 65) follow the v3 replacements, not the v2 oracle.
Test coverage is established by the behavior each test executes, not by a repeated
design-number parameter. Concrete retry/fresh-meter, backoff cancellation,
cancellation-boundary and recorder-failure controls are in
`tests/test_budget_failures.py`; partial-settlement fsync/phase-end failure,
ordered record joins and concurrent close are in `tests/test_budget_review.py`.
The pinned native-loopback controls are in `tests/test_budget_native.py`.
`tests/test_budget_delta.py` covers cancellation of the first/all close callers,
proxy/writer cleanup, shared real close failures, and measured empty-message
checkpoints with literal charges and independent replay agreement.
`tests/test_budget_shadow.py` compares forwarded bytes/headers, responses and
both ordinary recording modes against no-budget trials, covering refusals,
policy/meter/write failures and phase swaps. Only clocks and opaque ids are fixed.
`tests/test_budget_shadow_io.py` uses unfrozen clocks and blocked storage to
check delivery, response completion, heartbeat, bounded overflow and shutdown.
`tests/test_budget_shadow_review.py` retains the 13 reviewer controls, adapted to
await asynchronous initialization before installing ledger hooks and to require
the enforcing cap's refusal rather than assert its old exclusion.

Shadow writes use one daemon and a 256-operation FIFO, outside the sender's
completion dependency. Full queues drop and count observations. Shutdown drains
for at most one second after the baseline proxy stops. Shadow accounting is
exempt from the baseline recording cap; enforcing journals/snapshots count
alongside ordinary recording in both modes. Final `observation_status` events
report coverage loss/errors/timeouts. Missing or incomplete summaries cannot
establish coverage; host-side `ShadowBudget.observation_status()` also reports a
pending writer or unavailable log without reading storage. Diagnostic I/O and
host-log handlers run on the writer, never on the forwarding event loop.

## Artifact contracts

Freeze selections, model/API/CLI versions, route/profile and tolerance **before**
measuring. Keep exports in a private location outside this public source tree.
The commands below read JSONL artifacts; no command makes a model call itself.
A failure or empty export is not a passed gate.

- `REPLAY_JSONL`: one row per admitted attempt, with `admit_seq`, `root` (the full
  semantic root), complete request `messages`, frozen `model`, `dialect`, `profile`
  (profile id) and `credit` (`eligible` or `none`), raw `n`, `o`, optional iteration
  `w`, and integer `committed_charge` joined from the journal. Every row needs its
  own frozen profile: one trial may use several models with different credit
  policies. Missing identity, policy or joined charge fails G1; arithmetic-only
  exploration cannot certify it. Include `old_charge` where a lane has an old
  relay, and a nonempty `difference_cause` for every old-ledger difference. Do not
  derive expected charges using production identity/credit helpers. The oracle
  drops cache-control only at Anthropic envelope paths; Chat/Responses payloads
  retain it. Message string/text-block equivalence applies to Anthropic and Chat.
- `M1_JSONL`: per aligned turn: `attempt`, `platform_charge`, `candidate_charge`,
  `selection_frozen`, `versions_frozen`, `worker_attribution: matched`, `reset`,
  `workers`, `non_anthropic`, `held_out`. Derive platform deltas from consecutive
  `TokenBudgetBlock.precalculated_total_tokens_remaining` values; reconcile the
  initial allocation and final output with `tokens_consumed`. Missing system,
  tools, thinking or worker attribution is unidentifiable, not zero. Compare
  candidate rules that differ and retain per-turn residuals. The tool's residual
  sign check is a minimum screen, not a substitute for examining systematic
  per-turn/reset bias.
- `M2_JSONL`: at least 30 real-route calls of one frozen model/mode, with `n`,
  `extends_previous`, provider-counted `tail_tokens`, `thinking_present` and
  `thinking_dropped`. Include retained and omitted earlier reasoning/thinking.
  Every row's counts must be measured plain integers (`n > 0`, `tail_tokens >= 0`).
  At least one real extension must compare a positive measured tail; retained and
  dropped thinking must both occur among comparable extensions, not only on rows
  that skip the comparison. A nonempty but incomparable trajectory fails M2.
- `M3_JSONL`: actual successful shadow calls with `final_body_bytes`, `r_in`, raw
  `n`, `o`, `terminal`, `successful`, `real_route`, `profile`, `bound_source`,
  `bound_version`, `route_bound_source`, `route_bound_version`,
  `tools_transition`, `thinking_transition`; opaque-reasoning rows additionally
  name `effective_input_contract`. In shadow mode, export raw counts, terminal
  and response-success flags, `forwarded_bytes`, bounds, model and profile from
  the separate shadow log's `usage` events; map `forwarded_bytes` to
  `final_body_bytes`. The body measured is the actual baseline body after model
  normalization/confinement, not the copy with hypothetical limit/usage injection.
  Join bound-source versions from the frozen launch policy. Missing bounds or
  profiles and would-refuse/error events are not evidence to discard to make M3
  pass. Bounds and injection allowance come from trusted configuration, not
  sample fitting. Stand-in usage must set `real_route: false` and cannot pass M3.
  Successful raw prompt evidence must be nonzero. Transition coverage is required
  for each profile, not borrowed from another profile. Source/provenance review
  remains necessary: a JSON boolean does not authenticate a measurement.

## M1 — comparative reporting only (unrun)

Coverage: at least five attempts with reset/auto-compaction, at least two with
workers, at least two non-Anthropic attempts where available, and a held-out third
never used for fitting. Held-out attempt agreement must be within 1%; inspect
per-turn residuals for systematic sign and reset errors. Disable workers for a
matched measurement or report the comparison unavailable. A passing growth
fixture, or a +12.3% conversion, is not an identification of the platform.

```sh
: "${M1_JSONL:?frozen aligned real platform/Harbor measurements}"
( ulimit -v 6291456; nice -n 19 uv run python -m tests.budget_replay m1 "$M1_JSONL" )
```

## M2 — per-model/mode credit eligibility (unrun)

A single prefix-rendering violation keeps the profile at `credit: none`.
Choose and record the framing tolerance before examining the result. A passing
sample is empirical support for this route/model/mode, not a universal proof.

```sh
: "${M2_JSONL:?real-route retained/dropped-thinking trajectory}"
: "${FRAMING_TOLERANCE:?predeclared nonnegative token tolerance}"
( ulimit -v 6291456; nice -n 19 uv run python -m tests.budget_replay m2 "$M2_JSONL" --framing-tolerance "$FRAMING_TOLERANCE" )
```

## M3 — per-profile shadow input-bound check (unrun)

First run a lane shadow with both wrappers using `enforce: false` and the actual
bound/injection source versions. Shadow mode preserves baseline traffic and
recording. Before exporting evidence, require a final `observation_status` event
with `complete: true` and zero dropped observations: a stalled/lost/erroring
writer cannot establish coverage. Export all final-body/raw-usage evidence under
the schema above, including unknown/error observations. Require nonzero
successful usage, representative tool/thinking transitions, and `n <= r_in` on
every qualifying call. Ordinary text measurements do not enable opaque reasoning
without an independent effective-input contract. Keep unbounded Gemini modes
disabled. A counterexample or missing coverage blocks the canary.

```sh
: "${SHADOW_JOB:?lane-built Harbor shadow job JSON}"
# LIVE MODEL CALLS: do not execute as part of implementation verification.
( ulimit -v 6291456; nice -n 19 uv run harbor run --config "$SHADOW_JOB" --yes --plugin sandbox-agent-proxy )
: "${M3_JSONL:?trusted final-body/raw-usage export from that shadow}"
( ulimit -v 6291456; nice -n 19 uv run python -m tests.budget_replay m3 "$M3_JSONL" )
```

## G1 — independent replay per lane (unrun)

Replay whole recorded trials, including real auto-compaction. Join committed
facts by ledger/call/attempt, not pending usage records' null charges. Classify
every difference from the old scalar; do not demand equality with it. Report
new-root first-use charges, same-root resets, re-sent output, branch/iteration
work, credit invalidations, bound/partial charges and the graded cutoff. Do not
size a limit by multiplying a diagnostic.

```sh
: "${REPLAY_JSONL:?complete frozen lane trial export}"
( ulimit -v 6291456; nice -n 19 uv run python -m tests.budget_replay replay "$REPLAY_JSONL" )
```

## G2 — native refusal and grade preservation per lane (unrun)

Use the lane's actual pinned CLI (Claude Code 2.1.309 or agy 1.4.12 for the design's
controls), native sandbox-service queue, and final verifier. Exercise unary 400
and post-200 SSE errors. An unchanged conversation, changed-content near miss,
exhaustion control and known-good natural submission must give the distinct
expected outcomes. Keep the actual submission and grade after exhaustion. A
queue client disconnect must still leave measured accounting. Exercise native
Google correct/missing/wrong/foreign keys; missing/wrong keys must never reach
routing before F3 can be called closed. Pin the bridge build in that evidence.
The native code is encoded in the message suffix
` [hsap-budget-code:<closed_code>]`, decoded by `budget.native_refusal_code`.
The pinned Inspect Anthropic renderer does not emit `error.code`; do not assert
that it does. Both native forms must preserve the exact encoded terminal message
and bind it to the durable refusal event. Decoding an unrelated message is not
budget-ending acceptance.

```sh
: "${G2_JOB:?lane-built native-CLI control job, verifier enabled}"
( ulimit -v 6291456; nice -n 19 uv run harbor run --config "$G2_JOB" --yes --plugin sandbox-agent-proxy )
```

Repeat that exact invocation for each lane-built positive and negative control
configuration. The lane must run its result reader over those outputs. Its
builder, native-terminal extraction and grader are not implemented in this
repository; no made-up generic shell flag substitutes for those lane commands.

## G3 — real builder/result-reader contract (unrun)

The lane supplies its **existing exact** build and reader commands, including
farm-tunnel and stale-pin/tarball paths. They must refuse omitted/shadow/wrong
budgets unless explicitly stamped as shadow experiments; wrappers must agree.
The reader uses `read_budget` and `classify_ending`, binds trusted launch identity,
configuration hash, source commit, limit and profiles, and keeps every available
grade. Test natural success, ordinary agent failure, real runner failure, limit,
insufficient, process-death/nonclosed journals, and material uncertainty. Copied
snapshots, malformed/unrelated terminal errors and wrong identities cannot
supply the exhaustion witness.

```sh
: "${LANE_BUILD_COMMAND:?exact approved lane builder invocation}"
: "${LANE_READER_COMMAND:?exact approved lane result-control invocation}"
bash -euc "$LANE_BUILD_COMMAND"
bash -euc "$LANE_READER_COMMAND"
```

These variables are prerequisites, not claims that a missing lane integration is
implemented here. Local wrapper tests exercise only this package's boundary.

## G4 — real Job and sync/sentinel isolation (unrun)

The lane's real Job must use a Secret reference in `main` only; never fetch a real
key for this test. Use a fresh sentinel. Sync mounts only the export subpath,
not runner inputs, the Docker root or its socket. Test both the sandbox agent
surface and sync. The runner, egress sidecar, operators and principals able to
read/mount the Secret remain trusted.

```sh
: "${LANE_SENTINEL_COMMAND:?exact approved real-Job sentinel/isolation test invocation}"
bash -euc "$LANE_SENTINEL_COMMAND"
```

This test changes a deployment and is explicitly outside implementation work.
An offline manifest inspection does not pass it.

## G5 — bounded enforced canary (unrun)

Only after that lane's G1–G4 **and real M3** pass: one enforced trial per lane,
limit chosen to reach exhaustion, old relay retained as backstop where present.
Check a complete durable record, native refusal binding, clean runner outcome
and retained final grade. Only G1–G5 together allow production enablement and
removal of that lane's old enforcing relay. No deployment correction or parity
claim follows merely from running the command.

```sh
: "${CANARY_JOB:?reviewed bounded enforcing lane job after G1-G4/M3}"
( ulimit -v 6291456; nice -n 19 uv run harbor run --config "$CANARY_JOB" --yes --plugin sandbox-agent-proxy )
: "${LANE_READER_COMMAND:?exact lane reader invocation for this canary}"
bash -euc "$LANE_READER_COMMAND"
```
