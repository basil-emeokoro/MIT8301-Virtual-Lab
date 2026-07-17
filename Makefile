.PHONY: pipeline report test validate all
pipeline:
	python scripts/run_pipeline.py
report:
	python scripts/build_submission.py
test:
	pytest -q
validate:
	python scripts/validate_submission.py
all: pipeline report test validate
