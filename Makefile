# Define variables for Poetry and Python executable
POETRY = poetry
PYTHON = $(POETRY) run python

# Define the targets
.PHONY: all format typecheck run

all: format typecheck run

format:
	$(POETRY) run black ./ec_data_analysis/

typecheck:
	$(POETRY) run mypy ./ec_data_analysis

run:
	$(POETRY) run python ec_data_analysis/main.py

# target to install dependencies
install:
	$(POETRY) install

# target to clean up
clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf ./out/*
