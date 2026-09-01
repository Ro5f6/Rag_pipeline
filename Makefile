.PHONY: help install test eval eval-gen serve ingest ask docker-build docker-run clean

help:
	@echo "make install       Install dependencies into the active environment"
	@echo "make test          Run the unit test suite"
	@echo "make eval          Run the retrieval evaluation harness"
	@echo "make eval-gen      Run the RAGAS generation evaluation (needs a judge key)"
	@echo "make serve         Start the API + web UI on http://localhost:8000"
	@echo "make ingest        Build and persist the indexes from data/sample_docs"
	@echo "make ask Q='...'   Ask a single question from the command line"
	@echo "make docker-build  Build the container image"
	@echo "make docker-run    Run the container on http://localhost:8000"
	@echo "make clean         Remove caches and persisted indexes"

install:
	pip install -r requirements.txt

test:
	pytest

eval:
	python -m evaluation.run_eval

eval-gen:
	python -m evaluation.eval_generation

serve:
	uvicorn main:app --host 0.0.0.0 --port 8000

ingest:
	python -m pipeline.orchestrator --ingest-only

ask:
	python -m pipeline.orchestrator --query "$(Q)"

docker-build:
	docker build -t rag-pipeline .

docker-run:
	docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) rag-pipeline

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage data/index
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
