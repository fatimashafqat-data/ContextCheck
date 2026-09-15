# Dataset documentation

## Source and labels

Davidson, T., Warmsley, D., Macy, M., and Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language*. ICWSM.

[Original repository](https://github.com/t-davidson/hate-speech-and-offensive-language)

The notebook reads `data/labeled_data.csv` directly from the source repository. Raw tweet files are not bundled in this package. The source repository provides the dataset, citation, license, and known limitations.

| Label | Category |
|---|---|
| 0 | Hate speech |
| 1 | Offensive language |
| 2 | Neither |

The labels are human annotations, not objective ground truth. The data contain offensive material and have documented bias limitations.

## Preparation audit

| Stage | Removed or excluded | Retained |
|---|---:|---:|
| Original CSV | — | 24,783 |
| Missing or blank records | 0 | 24,783 |
| Initial conflicting text groups | 2 rows | 24,781 |
| Additional normalized duplicates | 11 rows | 24,770 |
| Conflicting groups after text preparation | 49 rows across 11 groups | 24,721 |
| Additional prepared-text duplicates | 177 rows | 24,544 |

Initial comparison normalizes case and whitespace. Model preparation additionally decodes HTML entities and replaces usernames and links with placeholders. Conflicting examples are set aside, not relabeled. EDA figures describe the 24,770-row table; predictive models use the 24,544-row table.

The `source_id` field records the original CSV row position assigned before filtering. Split assignments and the LLM sample manifest retain that identifier. The upstream CSV is referenced by its branch URL rather than a pinned commit, so exact reproduction depends on source contents remaining unchanged. No source-file checksum is available in this recorded run.
