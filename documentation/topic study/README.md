# Topic Study

This folder contains a topic study on Predictive Maintenance approaches.

## Latest pdf file

The most recent pdf can be found here: [View](https://git.cs.uni-paderborn.de/machine-learning-for-predictive-maintenance/documentation/-/jobs/artifacts/master/file/topic%20study/main.pdf?job=build_topic_study) / [Download](https://git.cs.uni-paderborn.de/machine-learning-for-predictive-maintenance/documentation/-/jobs/artifacts/master/raw/topic%20study/main.pdf?job=build_topic_study).

The project is rendered automatically on every commit by the [GitLab CI pipeline](https://git.cs.uni-paderborn.de/machine-learning-for-predictive-maintenance/documentation/-/blob/master/.gitlab-ci.yml).

## Manual build
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```