# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ''
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

conda-setup: ## Creates the conda environment that is used for developing this library.
	@conda env create -f .conda/examples.yml && conda env update -n ml4pdm --file .conda/development.yml --prune && conda env update -n ml4pdm --file .conda/ml4pdm.yml --prune

conda-update: ## Updates the conda environment that is used for developing this library.
	@conda env update -n ml4pdm --file .conda/ml4pdm.yml --prune && conda env update -n ml4pdm --file .conda/examples.yml --prune && conda env update -n ml4pdm --file .conda/development.yml --prune

test: ## Run all tests for the ml4pdm library.
	@python -m pytest -v --cov=ml4pdm --cov-report term --cov-report xml:test-results/coverage.xml --junitxml=test-results/results.xml

clean: ## Cleans Sphinx build files and Python caches.
	@rm -rf `find . \( -name "__pycache__" -o -name "_build" -o -name ".pytest_cache" \)`

gitlab-builder: ## Builds the Docker image that is being used in the GitLab pipeline.
	docker login hub.cs.upb.de
	docker build -t hub.cs.upb.de/machine-learning-for-predictive-maintenance/code/conda-builder -f Dockerfile.builder .
	docker push hub.cs.upb.de/machine-learning-for-predictive-maintenance/code/conda-builder

.PHONY: help conda-setup conda-update test clean Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
