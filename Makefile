.PHONY: help venv install build serve lint clean deploy

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
MKDOCS := $(VENV)/bin/mkdocs

help:
	@echo "make install  - create venv and install dependencies"
	@echo "make build    - build the static site into ./site (strict)"
	@echo "make serve    - live-reload dev server at http://127.0.0.1:8000"
	@echo "make lint     - check for broken internal links / nav issues"
	@echo "make deploy   - publish to the gh-pages branch"
	@echo "make clean    - remove build artifacts"

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

build:
	$(MKDOCS) build --strict

serve:
	$(MKDOCS) serve

# --strict turns broken links and nav warnings into errors, so build IS the lint.
lint: build

deploy:
	$(MKDOCS) gh-deploy --force

clean:
	rm -rf site
