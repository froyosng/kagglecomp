# The Analytics Edge Data Competition 2026

This repository contains our team’s R code, analysis, modelling experiments, and submission workflow for **The Analytics Edge Data Competition 2026**.
Our team consists of: Imelda Lee, Woon Zee Ning, Clarence Elvareta, Sng Zhenhao

## Competition Overview

The objective of the competition is to predict which bundle of car safety features a customer will choose.

Each observation contains four alternative safety-feature bundles, with exactly one bundle selected. Our model must generate a predicted probability for each of the four alternatives.

* **Competition start:** 24 July 2026, 12:00 PM SGT
* **Competition end:** 1 August 2026, 12:00 PM SGT
* **Report deadline:** 10 August 2026, 12:00 PM SGT
* **Kaggle competition:** https://www.kaggle.com/t/42a0b591922c450d99cb01ac7a662b66
* **Programming language:** R only

## Current status

Best model, both by CV and public leaderboard: an 80/20 ensemble of a conditional
logit (dummy-coded attributes, Price recoded as a 12-level factor instead of linear,
price/opt-out heterogeneity by covariates/segment/task/region/parking, plus
choice-set context effects -- is this alternative the cheapest/dearest in its task,
and by how much) with an XGBoost multiclass model. CV log loss **1.145094**, public
leaderboard **1.202** (`submission_ensemble_v11_pricegap.csv`, submitted 2026-07-26).
Note: the CV-to-public gap has grown across every complexity increase this project
has made (mod1 0.034 -> ensemble_v9 0.052 -> ensemble_v11 0.057) -- see
`cleaning_log.md` / `submissions_log.csv` for the full discussion, relevant for the
report's public-vs-private section. Full model history, findings, and current next
steps live in
[`AGENTS.md`](AGENTS.md) and [`cleaning_log.md`](cleaning_log.md) -- read those first
before starting new work. Every model tried (submitted or not) is tracked with its
validation/public log loss in [`submissions_log.csv`](submissions_log.csv).

## Data

The competition provides the following files:

```text
csv files/
├── train.csv
├── test.csv
└── sample_submission.csv
```

* `train.csv`: 21,565 labelled observations, 1,135 respondents x 19 choice tasks
* `test.csv`: 4,997 unlabelled observations, 263 respondents (**entirely disjoint** from
  the training respondents -- see `cleaning_log.md`, this matters for validation design)
* `sample_submission.csv`: Required submission format

The competition data is not included in this repository (`csv files/` is gitignored).
Team members should download the files from Kaggle and place them inside a local
`csv files/` directory at the repo root.

## Evaluation Metric

Submissions are evaluated using multiclass log loss:

[
\text{LogLoss}
==============

-\frac{1}{n}
\sum_{i=1}^{n}
\sum_{j=1}^{4}
y_{ij}\log(p_{ij})
]

where:

* (n) is the number of observations;
* (y_{ij}=1) when alternative (j) is selected for observation (i), and (0) otherwise;
* (p_{ij}) is the predicted probability that alternative (j) is selected.

A lower log-loss score indicates better predictive performance.

The benchmark predicts a probability of `0.25` for every alternative and has a log-loss score of:

```text
1.38629
```

Predicted probabilities should sum to `1` across the four alternatives for every observation.

## Repository Structure

```text
.
├── R/                       # Reusable R functions (log_loss.R, model scripts)
├── notebooks/experiments/   # Individual exploratory notebooks (one per person)
├── csv files/               # Competition data, local only (gitignored)
├── data_processed/          # Cached derived objects e.g. the train/val split (gitignored)
├── AGENTS.md                # Project memory: state, findings, next steps -- read first
├── cleaning_log.md          # Narrative log of every data/modelling finding
├── submissions_log.csv      # Every model tried: validation + public log loss
├── model_experiments_log.csv# Imelda's per-notebook experiment tracker
├── competition_report.qmd   # Source for the final report
├── submission_*.csv         # Generated Kaggle submissions (root level)
├── models/, figures/, report/, submissions/  # Placeholders for individual workflows
├── README.md
├── .gitignore
└── kagglecomp.Rproj
```

## Project Workflow

1. Inspect and understand the training and test data.
2. Conduct exploratory data analysis.
3. Establish a validation strategy.
4. Build a benchmark model.
5. Test alternative model specifications.
6. Perform feature engineering.
7. Compare models using validation log loss.
8. Generate predicted probabilities for the test set.
9. Submit predictions to Kaggle.
10. Record public leaderboard results and model details.

## Notebooks

Individual exploratory work lives under `notebooks/experiments/<name>_*.Rmd`, one
person at a time, so we don't step on each other's cells. The shared, validated
pipeline (feature builders, final model specs) lives in `R/` and `competition_report.qmd`
-- fold a working result from your own notebook into those once it's confirmed on the
canonical respondent-level split (see below), and log it in `submissions_log.csv`.

## Submission Tracking

Every model anyone tries -- submitted to Kaggle or not -- is recorded in
[`submissions_log.csv`](submissions_log.csv) with its validation log loss, public
leaderboard score (if submitted), the gap between them, and a source citation for
anything beyond class material. Check it before re-trying something; it's the single
source of truth for what's already been tried and what worked.

Teams may make a maximum of **two Kaggle submissions per day** -- use validation log
loss to decide what's worth a submission slot rather than testing everything on Kaggle.

**Validation must be respondent-level**, not row-level or task-level: each respondent
answered 19 correlated tasks, and the real test set is 263 respondents who never appear
in training. Split on `Case` (respondent id), not on individual rows or on `Task`
position -- a `Task`-based split leaks a respondent's other tasks into both sides and
will make your local score look better than it really is. The canonical split (seed
7402, 80/20 by respondent) is what every logged number above should use.

## Leaderboard

The test observations are divided into:

* **70% public leaderboard**
* **30% private leaderboard**

The final competition ranking is determined using the private leaderboard. The team’s best-scoring public submission will automatically be evaluated on the private leaderboard after the competition closes.

## Team Git Workflow

The `main` branch should contain the stable version of the project.

Each team member should work on a separate branch:

```bash
git switch -c <branch-name>
```

After making changes:

```bash
git add .
git commit -m "Describe the changes made"
git push
```

Completed work should be merged into `main` through a pull request.

Before beginning new work, update your branch with the latest changes from `main`.

## Competition Rules

* Only R may be used for the competition.
* Any R package may be used.
* Online resources and AI-assisted tools may be used.
* Work must be completed only within the assigned team.
* Competition data must not be privately shared outside the team.
* Only one Kaggle account may be used per team.
* The team may submit no more than two predictions per day.

## Report

The final report may contain a maximum of eight pages and should discuss:

1. The best-performing model on the public leaderboard.
2. Alternative models tested during the competition.
3. Differences between public and private leaderboard performance.
4. Insights obtained from the model.
5. Limitations of the modelling approach.

No executive summary or appendix is required.
