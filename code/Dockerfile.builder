# Docker image used in GitLab CI pipeline

FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y \
    make latexmk texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra
RUN apt-get -y dist-upgrade
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

ADD Makefile .
ADD .conda/ .conda/

RUN make conda-setup
RUN rm -rf .conda Makefile
