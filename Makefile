PYTHON := uv run python

.PHONY: experiment
experiment:
	@echo "💡 Running experiments"
	@$(PYTHON) src/main.py --runs 5