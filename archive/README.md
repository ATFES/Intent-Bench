# Historical material

`experiments/` and `reports/` preserve prior observations, original ratio
definitions and experiment instructions. Their commands describe the historical
setup; use the repository README and `intent-bench` for current runs.

`legacy-tools/` retains the replaced paired executors and provider factories for
historical interpretation. They are not included in the installed package and
are not imported by the tool. Active implementations live in `src/intent_bench`.

`agent-candidates/` contains only candidate source files referenced by the retained
tables, plus adjacent measurement/result files when available. It excludes Agent
configuration, authentication, logs and model state. `sources.json` records missing
external paths as well as preserved candidates; absence is not a failed benchmark.

`migration.json` accounts for every original non-cache file. `unchanged` means a
direct byte comparison succeeded; `refactored` means the file was moved and edited,
so its active replacement must be read with the current API. No content hashes are
used. New runs and compilation products remain outside the repository history.
