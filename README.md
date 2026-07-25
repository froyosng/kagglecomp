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

## Data

The competition provides the following files:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

* `train.csv`: 21,565 labelled observations
* `test.csv`: 4,997 unlabelled observations
* `sample_submission.csv`: Required submission format

The competition data is not included in this repository. Team members should download the files from Kaggle and place them inside the local `data/` directory.

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
├── R/                  # Reusable R functions
├── data/               # Competition data stored locally
├── notebooks/          # EDA and modelling notebooks
├── models/             # Saved model objects
├── submissions/        # Generated Kaggle submissions
├── figures/            # Plots and visualisations
├── report/             # Final report files
├── README.md
├── .gitignore
└── DataComp2026.Rproj
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

| File                       | Description                           |
| -------------------------- | ------------------------------------- |
| `01_eda.Rmd`               | Exploratory data analysis             |
| `02_baseline.Rmd`          | Initial benchmark model               |
| `03_model_experiments.Rmd` | Alternative model experiments         |
| `04_final_model.Rmd`       | Final model and submission generation |

## Submission Tracking

Each submission should be recorded in a submission log.

| Submission           | Model                       | Validation Log Loss | Public Log Loss | Notes                                    |
| -------------------- | --------------------------- | ------------------: | --------------: | ---------------------------------------- |
| `submission_001.csv` | Equal-probability benchmark |             1.38629 |             TBD | Probability of 0.25 for each alternative |
| `submission_002.csv` | TBD                         |                 TBD |             TBD | TBD                                      |

Teams may make a maximum of **two Kaggle submissions per day**.

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
