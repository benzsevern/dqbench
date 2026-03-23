from pathlib import Path
from dqbench.adapters.base import DQBenchAdapter
from dqbench.models import DQBenchFinding


class MockAdapter(DQBenchAdapter):
    @property
    def name(self): return "MockTool"

    @property
    def version(self): return "1.0"

    def validate(self, csv_path):
        return [DQBenchFinding(column="test", severity="error", check="test", message="test")]


def test_mock_adapter():
    adapter = MockAdapter()
    assert adapter.name == "MockTool"
    findings = adapter.validate(Path("fake.csv"))
    assert len(findings) == 1
