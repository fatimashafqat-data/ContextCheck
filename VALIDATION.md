# Validation

The reference classic-model evaluation was executed in Colab on 3,682 test tweets. Feature extraction, model fitting and sample selection remain unchanged. The exact 30 sample IDs and Gemini cache signature are checked during execution.

The suite has four pytest functions covering key stability, completed-ID skipping, attempt budgets, disabled network access, schema validation, quota stopping and truncation recovery. Tests use controlled fixtures rather than live provider requests. Passing tests do not imply the hosted CI workflow has run.

OpenRouter initially permits 4,096 output tokens. Only a response whose finish reason is length triggers a bounded retry at 16,384 tokens. Prompt, schema and model are unchanged. The recovery policy and per-response output budgets are recorded. Earlier valid responses at 4,096 tokens remain reusable. This adaptive decoding protocol is a limitation of the exploratory comparison and should be reported with its scores.

The final verification requires all code sections in the current run to finish, a readable feature image, 45 finite coefficient rows, both validated response caches covering the same 30 IDs, passing tests and successful notebook capture. Unavailable quota, credentials or provider responses remain explicit incomplete states. A widget smoke check verifies prediction callbacks; visual appearance still requires human review.

Runtime certification and provider coverage are derived from execution_status.json. Automated checks use controlled fixtures and do not certify live API availability. Exact McNemar tests paired accuracy, not macro F1, and assumes independent tweet pairs.

## Published snapshot

All prior code sections completed in the supplied Colab run, and the saved notebook has no error outputs. The interpretability export contains 45 feature rows and a readable figure. All four behavioral tests passed in Colab. Saved coverage is Gemini 9/30 and OpenRouter 29/30, so the complete LLM benchmark and full completion gate remain false.

The publication revision changes repository links, documentation generation, and the default live-request flags. Model training, sampling, prompts, schema, and cached responses are unchanged. Saved notebook outputs reflect the supplied Colab run; publication-only edits are syntax-checked and exercised offline rather than represented as another Colab execution.

The two presentation figures render recorded metrics and the cached source-18879 comparison. They are reproducible with `python src/render_showcase.py` and are not screenshots of a running interface.
