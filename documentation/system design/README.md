# System design

This folder contains the system design of the Predictive Maintenance library.

## Latest pdf file

The most recent pdf can be found here: [View](https://git.cs.uni-paderborn.de/machine-learning-for-predictive-maintenance/documentation/-/jobs/artifacts/master/file/system%20design/main_sys.pdf?job=build_system_design) / [Download](https://git.cs.uni-paderborn.de/machine-learning-for-predictive-maintenance/documentation/-/jobs/artifacts/master/raw/system%20design/main_sys.pdf?job=build_system_design).

The project is rendered automatically on every commit by the [GitLab CI pipeline](https://git.cs.uni-paderborn.de/machine-learning-for-predictive-maintenance/documentation/-/blob/master/.gitlab-ci.yml).

## Manual build
```bash
pdflatex main_sys.tex
biber main_sys
pdflatex main_sys.tex
pdflatex main_sys.tex
```