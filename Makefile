# Windows / cross-platform helpers (no bash required for core flows)
# Usage:  make smoke | make demo | make data | make eval
# Requires: python on PATH

PY ?= python
ROOT := .

.PHONY: help smoke demo data sft dpo eval deploy pipeline report plots clean-outputs

help:
	@echo Targets:
	@echo   make smoke      - unit/schema smoke (no model download)
	@echo   make demo       - full pipeline in --demo mode
	@echo   make data       - build domain data
	@echo   make eval       - offline eval (use DEMO=1 for mock)
	@echo   make deploy     - vLLM helper + serving bench (DEMO=1 for offline mock)
	@echo   make pipeline   - alias for demo
	@echo   make plots      - regenerate docs/assets/*.png from reports
	@echo   make report     - remind FINAL_REPORT path
	@echo   make clean-outputs - remove outputs/ mock checkpoints

smoke:
	$(PY) scripts/smoke_test.py

demo:
	$(PY) scripts/run_pipeline.py --stage all --demo

pipeline: demo

data:
	$(PY) scripts/run_pipeline.py --stage data $(if $(DEMO),--demo,)

sft:
	$(PY) scripts/run_pipeline.py --stage sft $(if $(DEMO),--demo,)

dpo:
	$(PY) scripts/run_pipeline.py --stage dpo $(if $(DEMO),--demo,)

eval:
	$(PY) scripts/run_pipeline.py --stage eval $(if $(DEMO),--demo,)

deploy:
	$(PY) scripts/run_pipeline.py --stage deploy $(if $(DEMO),--demo,)

report:
	@echo See reports/FINAL_REPORT.md

plots:
	$(PY) scripts/09_plot_reports.py

clean-outputs:
	$(PY) -c "import shutil, pathlib; p=pathlib.Path('outputs'); shutil.rmtree(p, ignore_errors=True); print('removed outputs/')"
