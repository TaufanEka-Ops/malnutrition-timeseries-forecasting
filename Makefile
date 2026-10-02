.PHONY: install check test holdout walkforward validate reproduce all clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

check:
	$(PY) -m src.data --check

test:
	$(PY) -m pytest tests/ -q

holdout:
	$(PY) scripts/run_holdout.py

walkforward:
	$(PY) scripts/run_walk_forward.py

validate:
	$(PY) scripts/run_2021_validation.py

reproduce:
	$(PY) scripts/run_thesis_reproduction.py

all: check test reproduce holdout walkforward validate

clean:
	rm -rf results/figures/*.png .pytest_cache **/__pycache__
