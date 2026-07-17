import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_pipeline_outputs_are_complete():
    results = json.loads((ROOT / "reports/tables/pipeline_results.json").read_text(encoding="utf-8"))
    assert results["validation"]["validation_passed"]
    assert len(results["metrics"]) == 4
    for model in results["metrics"]:
        assert {"Accuracy", "Precision", "Recall", "F1", "ROC_AUC"} <= model.keys()
    assert (ROOT / "submission/Basil_Oforbuike_Emeokoro_MIT8301_CA.pdf").stat().st_size > 100_000
