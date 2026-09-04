# Performance gate

`.github/workflows/perf.yml` measures go-llama's throughput and a native
llama.cpp build (the engine's own llama.cpp commit) back to back on the
same runner, for arm64 and amd64, and compares the **go/native ratio**
of two metrics against `baseline.json`:

- `tg` — token generation (decode), `BenchmarkDecode` vs llama-bench `tg`
- `pp` — prompt processing, `BenchmarkPromptEval` vs llama-bench `pp512`

A metric regresses when its ratio drops more than the threshold (10%)
below the recorded baseline ratio; the job fails. Ratios, not raw
tok/s, are gated because the CI pool mixes CPU generations — native and
go-llama always run on the same runner, so their ratio is comparable
across runs while raw numbers are not. Both sides use best-of-N.

## Updating the baseline

The baseline moves only by a reviewed commit. After a run whose numbers
should become the new bar (an engine bump, an accepted kernel change):

```sh
gh run download <run-id> -n perf-arm64 -D /tmp/perf-arm64
gh run download <run-id> -n perf-amd64 -D /tmp/perf-amd64
go run ./internal/cmd/perfgate baseline -baseline bench/baseline.json -result /tmp/perf-arm64/result.json
go run ./internal/cmd/perfgate baseline -baseline bench/baseline.json -result /tmp/perf-amd64/result.json
```

Entries are keyed by architecture **and runner CPU model**
(`amd64/AMD EPYC 7763 64-Core Processor`, `arm64/Neoverse-N2`): the
hosted pool mixes CPU generations whose native and go-llama throughputs
do not scale alike, so a ratio recorded on one model is not a bar for
another. A run on a model with no entry is recorded but not judged (the
job passes and prints the command above), which is how a new model —
or an empty file — gets seeded; add an entry for a model once it shows
up often enough to be worth gating.

When the engine's llama.cpp moves (a `llamawasm2go` bump), update
`LLAMA_CPP_COMMIT` in the workflow to llama-wasm's submodule commit for
that release, and re-seed: the native reference changed.
