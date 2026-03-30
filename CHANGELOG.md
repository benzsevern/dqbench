# Changelog

## v1.1.0 — 2026-03-29

### Added
- **ER (Entity Resolution) benchmark category** — deduplicate and link records across three difficulty tiers
- **Pipeline benchmark category** — end-to-end pipeline orchestration and quality gate benchmarks
- GoldenMatch ER benchmark results (95.30 with LLM, 77.21 without)
- `EntityResolutionAdapter` interface for custom ER tool benchmarking
- CLI commands: `dqbench run goldenmatch`, `dqbench run goldenpipe`
- Dataset generation flags: `--er`, `--pipeline`, `--all`
- Built-in adapters for GoldenMatch (ER) and GoldenPipe (Pipeline)

### Changed
- Expanded from 2 categories (Detect, Transform) to 4 (Detect, Transform, ER, Pipeline)
- Total test count: 161 across 12 tiers
- Updated PyPI keywords to include entity-resolution, deduplication, record-linkage, pipeline
- Updated GitHub topics and repository description

## v1.0.0

- Initial release
- Detect benchmark: 3 tiers, 83 tests
- Transform benchmark category (experimental)
- Built-in adapters for GoldenCheck, Great Expectations, Pandera, Soda Core
- DQBench Score: weighted F1 across tiers
