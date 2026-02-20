#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = multimodal-lm-eap-ig
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
RELEASE_VERSION ?= v1.0.0-rc1

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Run release gate (lint + type + tests + build + twine check)
.PHONY: release_gate
release_gate:
	$(PYTHON_INTERPRETER) scripts/release/release.py gate --clean-dist


## Render release notes draft
.PHONY: release_notes
release_notes:
	$(PYTHON_INTERPRETER) scripts/release/release.py notes --version $(RELEASE_VERSION)


## Create release tag (no push). Override with RELEASE_VERSION=vX.Y.Z
.PHONY: release_tag
release_tag:
	$(PYTHON_INTERPRETER) scripts/release/release.py tag --version $(RELEASE_VERSION)


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
