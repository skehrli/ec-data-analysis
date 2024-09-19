POETRY = poetry

# Define the targets
.PHONY: all format typecheck run

all: format typecheck run

format:
	$(POETRY) run black ./ec_data_analysis/

typecheck:
	$(POETRY) run mypy ./ec_data_analysis

run:
	$(POETRY) run python main.py

# target to install dependencies
install:
	$(POETRY) install

# target to clean up
clean:
	rm -rf ec_data_analysis/__pycache__
	rm -rf .mypy_cache
