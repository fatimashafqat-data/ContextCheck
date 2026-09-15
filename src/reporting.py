"""Generate the project presentation from the recorded evaluation artifacts."""
import csv
import json
from pathlib import Path


def table(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return 'No rows available.'
    def clean(row):
        values = []
        for value in row:
            try:
                value = f'{float(value):.3f}'
            except ValueError:
                pass
            values.append(value.replace('|', '\\|').replace('\n', ' '))
        return '| ' + ' | '.join(values) + ' |'
    return '\n'.join([clean(rows[0]), '| ' + ' | '.join(['---']*len(rows[0])) + ' |'] + [clean(row) for row in rows[1:]])


def update_readme(root, repo='fatimashafqat-data/ContextCheck'):
    import re
    root = Path(root)
    repo = repo or 'fatimashafqat-data/ContextCheck'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Repository must have the form owner/name.')
    results = root / 'results'
    status = json.loads((results/'execution_status.json').read_text())
    stats = json.loads((results/'mcnemar_test.json').read_text())
    with (results/'test_model_comparison.csv').open() as stream:
        classic = {row['model']: row for row in csv.DictReader(stream)}
    with (results/'cross_llm_disagreements.csv').open() as stream:
        paired = list(csv.DictReader(stream))
    def provider_status(name):
        source = status['llm_sources'][name]
        if source['completed'] == source['planned']:
            return 'Complete'
        errors = source.get('request_errors', [])
        reason = source.get('stop_reason') or (errors[-1].get('kind') if errors else None)
        if reason == 'quota_or_rate_limit':
            return 'Paused after HTTP 429 (quota or rate limit)'
        if reason == 'upstream_failure':
            http = errors[-1].get('http_status')
            return f'Provider HTTP {http}; remaining response(s) pending'
        if reason:
            return str(reason).replace('_', ' ').capitalize()
        return 'Partial cached coverage; live requests optional'
    def figure(name):
        return 'images/'+name if (root/'images'/name).is_file() else f'https://raw.githubusercontent.com/{repo}/main/images/{name}'
    if status['llm_benchmark_complete']:
        llm_results = table(results/'all_models_30_comparison.csv') + '\n\n![Same-sample comparison](images/06_four_model_comparison.png)'
    else:
        llm_results = '**Full four-model benchmark scores are withheld until both sources reach 30/30.** Available responses remain inspectable, and failed requests remain missing.'
    sources = status['llm_sources']
    values = dict(repo=repo,
        colab=f'https://colab.research.google.com/github/{repo}/blob/main/notebooks/contextcheck_toxicity_detector.ipynb',
        cover=figure('contextcheck_cover.png'), case_image=figure('recorded_disagreement.png'),
        classic_table=table(results/'test_model_comparison.csv'),
        lr_f1=float(classic['Logistic Regression']['macro_f1']),
        rf_f1=float(classic['Random Forest']['macro_f1']),
        baseline_f1=float(classic['Majority baseline']['macro_f1']),
        paired_n=len(paired), disagreement_n=sum(row['llm_disagreement'].lower()=='true' for row in paired),
        gemini_n=sources['gemini']['completed'], router_n=sources['openrouter']['completed'],
        gemini_model=sources['gemini']['model'], router_model=sources['openrouter']['model'],
        gemini_status=provider_status('gemini'), router_status=provider_status('openrouter'),
        p_value=stats['p_value'], lr_only=stats['lr_only_correct'], rf_only=stats['rf_only_correct'],
        llm_results=llm_results)
    template=(root/'docs/README.template.md').read_text(encoding='utf-8')
    (root/'README.md').write_text(template.format(**values),encoding='utf-8')
