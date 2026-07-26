.PHONY: install test lint run profile clean

install:
	python -m pip install -e ".[dev,api]"

install-spark:
	python -m pip install -e ".[full]"

test:
	python -m pytest -q

lint:
	ruff check src tests

run:
	python -m synthetic_data_platform.cli run --input data/raw/customer_events.csv --output artifacts --model vae --epochs 120 --samples 500

profile:
	python -m synthetic_data_platform.cli profile --input data/raw/customer_events.csv

spark-profile:
	python -m synthetic_data_platform.cli spark-profile --input data/raw/customer_events.csv

spark-evaluate:
	python -m synthetic_data_platform.cli spark-evaluate --real data/raw/customer_events.csv --synthetic artifacts/latest/synthetic.csv

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
