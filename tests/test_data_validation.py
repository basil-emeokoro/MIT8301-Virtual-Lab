from pathlib import Path
import pandas as pd

from mit8301_credit_risk.data_validation import validate_and_restore


ROOT = Path(__file__).resolve().parents[1]


def test_exact_target_reconciliation(tmp_path):
    output = tmp_path / "restored.csv"
    report = tmp_path / "report.md"
    result = validate_and_restore(ROOT / "data/raw/german_credit_data.csv", ROOT / "data/raw/target_source/german_credit_with_risk.csv", output, report)
    restored = pd.read_csv(output)
    assert result["validation_passed"]
    assert result["total_mismatched_cells"] == 0
    assert restored.shape == (1000, 10)
    assert restored.credit_risk.value_counts().to_dict() == {0: 700, 1: 300}


def test_mismatch_stops_reconciliation(tmp_path):
    supplied = pd.DataFrame({"Age": [20], "Sex": ["male"]})
    candidate = pd.DataFrame({"Age": [21], "Sex": ["male"], "Risk": ["good"]})
    supplied.to_csv(tmp_path / "a.csv", index=False); candidate.to_csv(tmp_path / "b.csv", index=False)
    try:
        validate_and_restore(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "o.csv", tmp_path / "r.md")
    except ValueError as exc:
        assert "mismatched" in str(exc)
    else:
        raise AssertionError("A predictor mismatch must stop target restoration")
