# Running ContextCheck

## Explore the published notebook

1. Open [ContextCheck in Colab](https://colab.research.google.com/github/fatimashafqat-data/ContextCheck/blob/main/notebooks/contextcheck_toxicity_detector.ipynb).
2. Click **Copy to Drive** if you want an editable copy, then click **Connect**.
3. Choose **Runtime → Change runtime type**, select **CPU** or **None** for the hardware accelerator, and click **Save**. The project does not require a GPU.
4. Choose **Runtime → Run all**. The published notebook trains the classic models and restores the real LLM response snapshots. Live requests are disabled by default.
5. Scroll to **Section 31: Interactive demo**. Enter a post and click **Analyze** for Logistic Regression and Random Forest predictions. Choose a source ID and click **Show cached LLM assessments** to inspect an existing comparison.

Expected local outputs include three EDA figures, model metrics, a confusion matrix, six error cases, 45 top-feature rows, a feature plot, the exact McNemar result, and **4 passed** from pytest. The comparison table always contains the same 30 sample IDs. Missing LLM responses remain blank.

## Resume live LLM requests

The published snapshot contains **9 Gemini responses and 29 OpenRouter responses**. Completing the remaining 21 and 1 responses depends on provider availability and quota. Colab's free CPU runtime and an LLM provider's free API quota are separate services.

1. Click the **key icon** in Colab's left sidebar to open **Secrets**.
2. Add or enable `GEMINI_API_KEY` and `OPENROUTER_API_KEY`. Turn on **Notebook access** for each. Keep the values in Secrets.
3. In **Section 1: Runtime and experiment configuration**, change `RUN_GEMINI = False` to `RUN_GEMINI = True`, and `RUN_OPENROUTER = False` to `RUN_OPENROUTER = True`. Enable just one provider if that is the one you want to resume.
4. Keep each attempt budget at **35 or lower**. A budget includes retries and is not an assurance that the daily quota is available.
5. Choose **Runtime → Run all**. The provider cells skip successful cached IDs, save each new valid answer immediately, and print their completion counts.

HTTP 429 stops that provider's run. Temporary provider failures get bounded retries. Missing credentials, invalid responses, and provider errors are recorded without becoming predictions. When quota or availability recovers, run the notebook again. There is no automatic paid-model substitution.

The fixed sample, classification prompt, schema, and model identities remain unchanged. Optional **Ask Gemini** and **Ask OpenRouter** demo buttons make one separate request and do not add examples to the benchmark.

## Preserve progress and export

Colab runtime files can disappear after the runtime is deleted. After any successful run, use **Download notebook** and **Download project ZIP** in the final cell. The exported notebook includes small JSON cache snapshots. Opening that exported notebook in a fresh runtime restores those responses. The repository notebook restores the responses in the published version.

The final cell prints the actual completion state. `llm_benchmark_complete` and `expanded_notebook_colab_verified` remain false while either provider is below 30/30. Interpretation, classic analyses, and tests can be complete while the LLM benchmark remains partial. The recorded publication snapshot is intentionally marked incomplete for the remaining LLM work.

The exported ZIP contains the generated source, figures, documentation, saved results, and refreshed `SHA256SUMS.txt`. Presentation-only figures in the GitHub repository are rendered from saved results; their published versions remain linked when a fresh Colab export does not contain them locally.

## Run the automated tests

The suite uses controlled fixtures and makes no live API calls. From the repository root in a terminal:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

In Colab, after the project has been initialized:

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'],
               cwd='/content/contextcheck', check=True)
```

Expected result: **4 passed**. This checks software behavior, not LLM accuracy or provider availability.

## Reproduce the portfolio figures

The optional repository script reads saved results without model training or requests:

```bash
python src/render_showcase.py
```

The notebook remains the supported environment for the full analysis. `requirements.txt` records the versions used in the Colab run; `requirements-dev.txt` contains the smaller dependency set used by CI.
