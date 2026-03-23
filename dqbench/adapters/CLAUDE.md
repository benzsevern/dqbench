# Adapters

Bridge between DQBench and any data validation tool.

## ABC Contract

```python
class DQBenchAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...        # shown in scorecard / comparison table

    @property
    @abstractmethod
    def version(self) -> str: ...     # tool version string

    @abstractmethod
    def validate(self, csv_path: Path) -> list[DQBenchFinding]: ...
```

## DQBenchFinding Fields

| Field | Type | Notes |
|-------|------|-------|
| `column` | `str` | Column name; comma-separate for cross-column checks |
| `severity` | `str` | `"ERROR"`, `"WARNING"`, or `"INFO"` |
| `check` | `str` | Machine-readable check name (matched against `ISSUE_KEYWORDS`) |
| `message` | `str` | Human-readable description (also keyword-matched) |
| `confidence` | `float` | 0-1, default 1.0 |

Severity matters: only `ERROR`/`WARNING` count as false positives on clean columns.
`INFO` findings are never penalised.

## Three Modes per Tool

Implement separate adapter classes for each mode to keep results comparable:

- **zero-config** — run the tool with no schema or config, pure auto-detection
- **auto-profiled** — use the tool's built-in profiling/auto-schema feature
- **best-effort** — hand-tuned config, full knowledge of the dataset schema

## Built-in Registry

Adapters are registered by short name in `cli.py` `BUILTIN_ADAPTERS`:

```python
"mytool-zero": "dqbench.adapters.mytool_adapter:MyToolZeroConfigAdapter",
"mytool-auto": "dqbench.adapters.mytool_adapter:MyToolAutoProfileAdapter",
"mytool-best": "dqbench.adapters.mytool_adapter:MyToolBestEffortAdapter",
```

Also add the names to `ALL_ADAPTER_NAMES` for inclusion in `dqbench run all`.

## Adding a New Adapter

1. Create `dqbench/adapters/<toolname>_adapter.py`.
2. Subclass `DQBenchAdapter` and implement all three abstract members.
3. Return `list[DQBenchFinding]` — do not raise; catch tool errors internally.
4. Register in `cli.py` (see above).

## Custom Adapter (No Registration Needed)

```bash
dqbench run mytool --adapter path/to/my_adapter.py
```

The file is scanned for any `DQBenchAdapter` subclass and instantiated
automatically — no changes to the DQBench codebase required.
