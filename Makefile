.PHONY: help venv install build serve lint clean deploy deploy-cf

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
MKDOCS := $(VENV)/bin/mkdocs

CF_PROJECT := inference-engineering-deep-dive

help:
	@echo "make install    - create venv and install dependencies"
	@echo "make build      - build the static site into ./site (strict)"
	@echo "make serve      - live-reload dev server at http://127.0.0.1:8000"
	@echo "make lint       - check for broken internal links / nav issues"
	@echo "make deploy-cf  - build and deploy to Cloudflare Pages (direct upload)"
	@echo "make deploy     - publish to the gh-pages branch (alternative host)"
	@echo "make clean      - remove build artifacts"

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

# Direct-upload the built site to Cloudflare Pages.
# First run triggers `wrangler login` (browser) and creates the project.
deploy-cf: build
	npx wrangler pages deploy site --project-name=$(CF_PROJECT) --branch=main

deploy:
	$(MKDOCS) gh-deploy --force

clean:
	rm -rf site
