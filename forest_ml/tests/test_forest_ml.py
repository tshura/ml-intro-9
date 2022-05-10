from click.testing import CliRunner
import pytest

from forest_ml.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()

def test_error_for_invalid_criterion(
    runner: CliRunner
) -> None:
    """It fails when criterion is not in ['gini', 'entropy']"""
    result = runner.invoke(
        train,
        [
            "--criterion",
            'foo',
        ],
    )
    assert result.exit_code == 2
    assert "criterion takes values 'gini' or 'entropy'" in result.output

    