# ContextCheck

### When toxicity models disagree

![ContextCheck: classic models, LLM assessments, and a test macro F1 of 0.725 for Logistic Regression]({cover})

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})
[![Tests](https://github.com/{repo}/actions/workflows/tests.yml/badge.svg)](https://github.com/{repo}/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Logistic Regression achieved {lr_f1:.3f} macro F1**, compared with {rf_f1:.3f} for Random Forest and {baseline_f1:.3f} for a majority baseline on 3,682 held-out tweets. ContextCheck investigates where word-based classifiers and LLM assessments disagree, combining reproducible evaluation, six error case studies, feature interpretation, and a Colab demo. The LLM comparison is in progress: **Gemini {gemini_n}/30; OpenRouter {router_n}/30**.

[Explore the notebook](notebooks/contextcheck_toxicity_detector.ipynb) · [Run the demo](#try-the-demo) · [Inspect the results](results/) · [Reproduce the analysis](RUN_GUIDE.md)

## The question

Can a classifier distinguish identity-based hate, everyday profanity, and a joke that quotes an insult? These categories share vocabulary. A model that always predicts the largest class already reaches **77.5% accuracy** in this test set, so the central comparison uses macro F1: each class receives equal weight.

One result makes the problem concrete: Logistic Regression reaches **85.8% accuracy**, but only **33.8% precision for hate speech**. Studying these errors is part of the project, alongside the scores.

## One post, two LLM assessments

![Recorded Gemini and Nemotron assessments for source 18879, contrasting a humorous reading with an offensive-language label]({case_image})

This is a rendering of **actual cached responses**, not a live interface. Gemini treats the post as a joke about a band; Nemotron treats “trash” as an insult. The dataset label and both classic models say “Neither.” The case illustrates how interpretation can change a classification; it does not establish which model generalizes better.

{paired_n} examples currently have answers from both LLMs; {disagreement_n} of those pairs disagree. These are coverage counts, not a completed benchmark. Inspect the [paired records](results/cross_llm_disagreements.csv).

## Classic model results

The dataset is split into **17,180 training**, **3,682 validation**, and **3,682 test** examples. Logistic Regression was selected using validation macro F1 before test evaluation.

{classic_table}

![Macro F1 on the held-out test set](images/04_test_model_comparison.png)

The selected model's hate-speech precision, recall, and F1 are **0.338**, **0.585**, and **0.428**. Its confusion matrix shows 124 of 212 hate-speech examples correctly identified, with 243 false hate-speech predictions across the other classes.

[Classification report](results/logistic_regression_test_report.csv) · [Confusion matrix](images/05_logistic_regression_confusion_matrix.png) · [Row-level predictions](results/test_predictions.csv)

## What the errors reveal

Six recorded Logistic Regression errors cover **three hate-speech false positives and three false negatives**. Personal arguments containing identity-linked slurs can be overclassified as hate speech. Very short gendered insults can also be classified as offensive language even when the reference annotators chose hate speech. Several examples have ambiguous intent; the analysis keeps observed wording separate from possible explanations.

**Feature interpretation:** the 15 largest positive coefficients for each class reveal strong associations with identity-linked slurs for hate speech, profanity for offensive language, and dataset-specific topic words for neither. These associations explain class scores at a global level; they do not establish intent, causality, or whether a word is harmless in a different context.

[Six annotated cases](results/error_analysis.md) · [45-feature table](results/lr_top_features.csv) · [Feature-weight figure](images/07_lr_top_features.png)

The linked examples and feature names contain offensive language from the research dataset.

## Paired statistical comparison

Exact two-sided McNemar testing gives **p = {p_value:.6f}**: Logistic Regression alone is correct on **{lr_only}** test examples, and Random Forest alone on **{rf_only}**. The observed accuracy difference is statistically significant at the 5% threshold under the independence assumption. This test concerns **accuracy**, not the macro-F1 gap, and does not establish performance on a different dataset.

[Test counts and p-value](results/mcnemar_test.json)

## LLM benchmark status

Both LLMs receive the **same fixed instructions, JSON schema, and 30 prepared test tweets**, selected as ten per category without using prediction correctness. Responses provide a class, a short assessment, and a human-review flag. The LLM assessment is separate from an explanation of the classic model's internal decision.

| Source | Fixed model | Valid responses | Latest run status |
|---|---|---:|---|
| Gemini | `{gemini_model}` | {gemini_n}/30 | {gemini_status} |
| OpenRouter | `{router_model}` | {router_n}/30 | {router_status} |

{llm_results}

Cache identities include the model, prompt, schema, and ordered sample. Each successful response is saved immediately; completed IDs are skipped on resume. Requests have bounded attempts and stop on HTTP 429. OpenRouter begins at 4,096 output tokens and permits one retry at 16,384 only when the provider reports length truncation; actual allowances are recorded with each response. Missing or invalid answers never become guessed labels.

[Execution status](results/execution_status.json) · [All 30 comparison rows](results/llm_comparison_rows.csv) · [Exact prompt](results/llm_prompt.txt) · [Schema](results/llm_response_schema.json)

## Try the demo

1. Open [ContextCheck in Google Colab]({colab}) and click **Connect**.
2. Choose **Runtime → Run all**. The published notebook uses a CPU runtime and cached LLM responses by default; no API key is needed for the classic analysis or saved comparisons.
3. Scroll to **Section 31: Interactive demo**, enter a post, and click **Analyze**. The interface shows both classic predictions and whether they disagree. **Show cached LLM assessments** opens a saved example without a request.

Optional live LLM buttons require Colab Secrets and the corresponding provider flag. The interface runs inside Colab; it is not a separately hosted web application. [Exact setup and resume instructions](RUN_GUIDE.md).

## Data and reproducibility

**Source:** Davidson, Warmsley, Macy, and Weber (2017), *Automated Hate Speech Detection and the Problem of Offensive Language*. [Dataset repository](https://github.com/t-davidson/hate-speech-and-offensive-language).

- **24,783 → 24,770 → 24,544 rows:** initial cleaning, then prepared-text conflict and duplicate exclusion. EDA describes the intermediate table; modeling uses the final one.
- **TF-IDF:** 10,000 word-unigram/bigram features fitted on training data only; class-weighted Logistic Regression and Random Forest run on CPU.
- **Split checks:** stratified 70/15/15 partition with seed 42; identical prepared texts cannot cross sets. Source row IDs and split assignments are saved.
- **Four behavioral tests:** cache identity, resume and budgets, disabled requests, strict response validation, failure preservation, and bounded truncation recovery. CI runs on pushes and pull requests without API keys.

[Class balance](images/01_class_distribution.png) · [Tweet lengths](images/02_tweet_length_by_category.png) · [Text patterns](images/03_text_patterns_by_category.png) · [Data notes](data/README.md) · [Validation](VALIDATION.md)

| Path | Contents |
|---|---|
| `notebooks/` | Annotated Colab analysis, saved outputs, and demo |
| `src/` | Validated API/cache logic, report generation, and figure rendering |
| `results/` | Audits, predictions, reports, real response caches, and execution status |
| `images/` | EDA, evaluation figures, and portfolio visuals |
| `tests/` | Offline pytest suite |
| `docs/` | README template used by the report generator |
| `.github/workflows/` | Automated test workflow |

## Limits and next steps

This historical English Twitter sample was selected around offensive vocabulary. Annotation disagreement, dialect associations, missing conversation context, and sampling bias limit transfer to current platforms. Exact duplicates are controlled; shared authors and near duplicates are not. The small balanced LLM sample does not estimate platform prevalence, and pretrained models may have encountered this public dataset.

The immediate next step is completing the remaining cached LLM requests when provider capacity allows. Further work would use an independent contemporary dataset, annotation review, calibrated confidence, and a larger paired LLM evaluation. This is a research portfolio project, not an automated moderation policy.

## Author and license

**Fatima Shafqat** · [GitHub](https://github.com/fatimashafqat-data)

Original code and documentation use the [MIT license](LICENSE). Third-party data, excerpts, and dependencies retain their respective terms.
