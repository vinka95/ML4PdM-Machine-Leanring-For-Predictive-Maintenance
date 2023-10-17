# Content

This repository contains all important documentation for the "Machine Learning for Predictive Maintenance" project proup. It also contains a topic study for some state-of-the-art approaches of Predictive Maintenance.

# Git Concept

![GitFlow example workflow](https://wac-cdn.atlassian.com/dam/jcr:b5259cce-6245-49f2-b89b-9871f9ee3fa4/03%20(2).svg?cdnVersion=1427)
Image by Atlassian ([source](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow))

## Branches
* Simplified form of the common [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) practice
* Master branch for tested and working implementations only
* Develop branch as a base for all feature branches
* Feature branch for every approach to implement
  * Name example: `feature/ci-pipeine`
  * Will not be deleted to be able to apply fixes later on

## Merge Requests
* Reviewed by at least two people (author and team member)
  * Use GitLab approval feature
* No squashing and no deletion of source branch
* Write a changelog in the description with the "types" from [keepachangelog.com](https://keepachangelog.com/en/1.0.0/)
  * Types:  Added, Changed, Deprecated, Removed, Fixed, Security 
  * Are already included in Merge Request Template

## Commit messages
* Labels: `[TFE]`, `[HSC]`, `[HIE]`, `[RUL]`, `[GEN]`
* Example: `[RUL] Added motivation part.`
* Provide short and meaningful commit messages
* Split work into two commits if it affects multiple topics

## Tags
* Use [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning#Sequence-based_identifiers)
* Created only on master branch
* `1.0.0` means all featured are successfully implemented
* `0.m.0` means that `m` features are implemented
* Will be defined further in next milestone

## CI Pipeline
* Automated tests on ever commit (only master/develop branches + merge requests)
* Will be specified in next milestone