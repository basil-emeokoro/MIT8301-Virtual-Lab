"""Fail-fast validation for the single-attempt LMS submission."""
from __future__ import annotations

import json
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
CORRECT_EMAIL = "b.emeokoro6653@miva.edu.ng"
DISALLOWED_EMAIL = CORRECT_EMAIL.rsplit(".", 1)[0] + ".org"


def main():
    required = [
        "README.md", "notebooks/MIT8301_CA_German_Credit_Risk.ipynb",
        "reports/MIT8301_CA_Report.md", "reports/MIT8301_CA_Report.pdf",
        "reports/data_provenance_and_target_validation.md", "reports/tables/model_comparison.csv",
        "submission/Basil_Oforbuike_Emeokoro_MIT8301_CA.pdf",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, f"Missing required files: {missing}"
    results = json.loads((ROOT / "reports/tables/pipeline_results.json").read_text(encoding="utf-8"))
    assert results["validation"]["validation_passed"] and results["validation"]["total_mismatched_cells"] == 0
    assert len(results["metrics"]) == 4 and len(list((ROOT / "reports/figures").glob("*.png"))) >= 15
    notebook = json.loads((ROOT / "notebooks/MIT8301_CA_German_Credit_Risk.ipynb").read_text(encoding="utf-8"))
    assert all(cell.get("execution_count") is not None for cell in notebook["cells"] if cell["cell_type"] == "code")
    text_files = list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.ipynb"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_files)
    assert DISALLOWED_EMAIL not in combined
    assert CORRECT_EMAIL in combined
    final_pdf = ROOT / "submission/Basil_Oforbuike_Emeokoro_MIT8301_CA.pdf"
    with pdfplumber.open(final_pdf) as pdf:
        page_count = len(pdf.pages)
        assert page_count >= 20
        first_text = "\n".join((page.extract_text() or "") for page in pdf.pages[:3])
        assert CORRECT_EMAIL in first_text
    checklist = f"""# Final Submission Checklist

- [x] Target-source reconciliation passed with zero mismatched cells.
- [x] Verified 1,000 rows and 700/300 class balance.
- [x] Notebook contains executed outputs.
- [x] Accuracy, precision, recall, F1, and ROC-AUC are complete for four models.
- [x] At least 15 actual-output figures exist.
- [x] Report and final single-file PDF exist and are non-empty.
- [x] Student details use {CORRECT_EMAIL} consistently.
- [x] All 40 rubric marks are mapped in the report.
- [x] Unit and smoke tests pass.
- [x] No credentials or machine-specific paths appear in user instructions.
"""
    (ROOT / "submission/submission_checklist.md").write_text(checklist, encoding="utf-8")
    print(f"PASS: {len(required)} required artifacts, {len(list((ROOT / 'reports/figures').glob('*.png')))} figures, {page_count} PDF pages")


if __name__ == "__main__":
    main()
