.PHONY: build test publish-test publish clean

build:
	uv build

check:
	uvx ty check && uvx ruff check && uvx ruff format

test:
	uv run pytest --cov=buse --cov-report=term-missing

publish-test:
	uv pip install twine
	uv run python -m twine upload --repository testpypi dist/*

publish:
	uv pip install twine
	uv run python -m twine upload dist/*

clean:
	rm -rf dist build *.egg-info
