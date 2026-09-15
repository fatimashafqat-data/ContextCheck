import json, hashlib, re, time
from datetime import datetime, timezone
from pathlib import Path
FIELDS = ('label', 'explanation', 'needs_review')

class ProviderFailure(RuntimeError):
    """A controlled diagnostic containing no prompt, key or reasoning text."""
    def __init__(self, kind, status=None, retry_after=None, details=None):
        self.kind = kind
        self.status_code = status
        self.retry_after = retry_after
        self.details = details or {}
        super().__init__(kind)


def validate(result):
    if not isinstance(result, dict) or set(result) != set(FIELDS):
        raise ProviderFailure('schema_fields')
    if type(result['label']) is not int or result['label'] not in (0, 1, 2):
        raise ProviderFailure('schema_label')
    if not isinstance(result['explanation'], str) or not result['explanation'].strip():
        raise ProviderFailure('schema_explanation')
    if type(result['needs_review']) is not bool:
        raise ProviderFailure('schema_review_flag')
    return result


def parse_answer(answer):
    if not isinstance(answer, str) or not answer.strip():
        raise ProviderFailure('empty_answer')
    answer = answer.strip()
    # Remove only a complete outer Markdown fence, without repairing content.
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', answer, flags=re.S | re.I)
    if fence:
        answer = fence.group(1).strip()
    try:
        result = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        raise ProviderFailure('invalid_answer_json', details={'answer_characters': len(answer)}) from None
    return validate(result)


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def load_cache(path, config):
    path = Path(path)
    cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'model': config['model'], 'run_signature': signature(config), 'responses': {}}
    if cache.get('model') != config['model'] or cache.get('run_signature') != signature(config):
        raise ValueError('Cache identity does not match the experiment configuration.')
    sample_ids = [str(r['source_id']) for r in config['sample']]
    if len(sample_ids) != len(set(sample_ids)) or not set(cache['responses']).issubset(sample_ids):
        raise ValueError('Cache or sample IDs are inconsistent.')
    for key, row in cache['responses'].items():
        if str(row['source_id']) != key:
            raise ValueError('Cached source ID mismatch.')
        validate({field: row[field] for field in FIELDS})
    return cache


def send_classification(text, *, provider, model, prompt, schema, enabled,
                        get_key, max_tokens=4096, post=None):
    """Send one HTTP request. Token recovery and retries belong to the runner."""
    if not enabled:
        raise ProviderFailure('live_disabled')
    if not isinstance(text, str) or not text.strip():
        raise ProviderFailure('empty_input')
    try:
        key = get_key()
    except Exception:
        raise ProviderFailure('secret_unavailable') from None
    if not isinstance(key, str) or not key.strip():
        raise ProviderFailure('secret_unavailable')
    key = key.strip()
    import requests
    content = prompt + '\nTweet to classify (JSON string):\n' + json.dumps(text, ensure_ascii=False)
    if provider == 'gemini':
        url = 'https://generativelanguage.googleapis.com/v1beta/interactions'
        headers = {'x-goog-api-key': key}
        payload = {'model': model, 'input': content,
            'response_format': {'type': 'text', 'mime_type': 'application/json', 'schema': schema}}
    elif provider == 'openrouter':
        if not model.endswith(':free'):
            raise ProviderFailure('free_model_required')
        url = 'https://openrouter.ai/api/v1/chat/completions'
        headers = {'Authorization': 'Bearer ' + key}
        payload = {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'toxicity_classification', 'strict': True, 'schema': schema}},
            'provider': {'require_parameters': True}, 'max_tokens': max_tokens}
    else:
        raise ProviderFailure('unknown_provider')
    try:
        response = (post or requests.post)(url, headers=headers, json=payload, timeout=(15, 300 if max_tokens > 4096 else 180))
    except requests.RequestException:
        raise ProviderFailure('network_failure') from None
    retry_header = response.headers.get('Retry-After', '')
    retry_after = float(retry_header) if re.fullmatch(r'\d+(?:\.\d+)?', retry_header) else None
    try:
        body = response.json()
    except (ValueError, TypeError):
        raise ProviderFailure('non_json_http_response', response.status_code,
            retry_after, {'response_characters': len(response.content)}) from None
    if not isinstance(body, dict):
        raise ProviderFailure('unexpected_http_body', response.status_code)
    if not response.ok or body.get('error'):
        error = body.get('error') or {}
        if not isinstance(error, dict):
            error = {}
        try:
            status = int(error.get('code', response.status_code))
        except (ValueError, TypeError):
            status = response.status_code
        # Map raw provider information to a small, controlled diagnostic vocabulary.
        lower_message = str(error.get('message', '')).lower()
        if status == 429:
            kind = 'quota_or_rate_limit'
        elif status in (401, 402, 403, 404):
            kind = {401: 'authentication', 402: 'credits_unavailable',
                    403: 'provider_declined', 404: 'model_unavailable'}[status]
        else:
            kind = 'upstream_failure'
        raise ProviderFailure(kind, status, retry_after,
            {'daily_quota_indicated': any(s in lower_message for s in ['per day', 'daily', 'per_day'])})
    diagnostics = {'http_status': response.status_code}
    if provider == 'openrouter':
        choices = body.get('choices') or []
        if not choices:
            raise ProviderFailure('missing_choices', response.status_code)
        choice = choices[0]
        message = choice.get('message') or {}
        finish = choice.get('finish_reason')
        usage = body.get('usage') or {}
        diagnostics.update({
            'finish_reason': finish if finish in ['stop', 'length', 'error', 'content_filter', 'tool_calls'] else 'other',
            'completion_tokens': usage.get('completion_tokens'),
            'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
            'max_tokens': max_tokens,
        })
        if finish == 'length':
            raise ProviderFailure('output_truncated', response.status_code, details=diagnostics)
        if finish == 'content_filter' or message.get('refusal'):
            raise ProviderFailure('provider_declined', response.status_code, details=diagnostics)
        if finish != 'stop':
            raise ProviderFailure('incomplete_generation', response.status_code, details=diagnostics)
        answer = message.get('content')
        diagnostics['generation_id'] = str(body.get('id', ''))[:100]
        diagnostics['provider'] = str(body.get('provider', ''))[:100]
    else:
        answer = ''.join(item.get('text', '') for step in body.get('steps', [])
            if step.get('type') == 'model_output' for item in step.get('content', [])
            if item.get('type') == 'text')
    try:
        result = parse_answer(answer)
    except ProviderFailure as error:
        error.details.update(diagnostics)
        raise
    return result, diagnostics


def run_cached(config, path, classify, *, enabled, budget=35, gap=15,
               seconds_limit=1800, sleep=time.sleep):
    """Resume missing IDs with finite retries and retain every validated success."""
    cache = load_cache(path, config)
    save_json(path, cache)
    state = {'enabled': bool(enabled), 'attempts': 0, 'errors': [], 'stop_reason': None}
    if not enabled or len(cache['responses']) == len(config['sample']):
        return cache, state
    if type(budget) is not int or not 1 <= budget <= 35:
        raise ValueError('Attempt budget must be between 1 and 35.')
    started_run = time.monotonic()
    consecutive_failures = 0
    # Record recovery settings alongside records without invalidating successful base requests.
    cache['recovery_policy'] = {'version': 1, 'max_attempts_per_id': 2,
        'openrouter_base_max_tokens': 4096, 'openrouter_truncation_max_tokens': 16384,
        'stop_on_429': True, 'max_consecutive_failed_ids': 3}
    for row in config['sample']:
        key = str(row['source_id'])
        if key in cache['responses']:
            continue
        token_budget = int(cache.get('resume_options', {}).get(key, {}).get('max_tokens', 4096))
        next_gap = gap
        success = False
        for attempt_for_id in range(2):
            if state['attempts'] >= budget or time.monotonic() - started_run >= seconds_limit:
                state['stop_reason'] = 'run_budget_reached'
                break
            if state['attempts']:
                sleep(max(15, next_gap))
            state['attempts'] += 1
            print(f"{config['model']}: attempt {state['attempts']}, source ID {key}", flush=True)
            before = time.perf_counter()
            try:
                result, diagnostics = classify(row['model_text'], token_budget)
                validate(result)
                cache['responses'][key] = {**result, 'source_id': int(key),
                    'elapsed_seconds': round(time.perf_counter()-before, 2),
                    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'diagnostics': diagnostics}
                save_json(path, cache)
                print(f"Saved {len(cache['responses'])}/{len(config['sample'])}", flush=True)
                success = True
                consecutive_failures = 0
                break
            except ProviderFailure as error:
                diagnostic = {'source_id': int(key), 'kind': error.kind,
                    'http_status': error.status_code, 'retry_after_seconds': error.retry_after,
                    **error.details, 'at_utc': datetime.now(timezone.utc).isoformat()}
                state['errors'].append(diagnostic)
                with Path(path).with_suffix('.errors.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(diagnostic) + '\n')
                print('Request diagnostic:', json.dumps(diagnostic), flush=True)
                if error.kind in ['quota_or_rate_limit', 'secret_unavailable', 'authentication',
                                  'credits_unavailable', 'model_unavailable', 'free_model_required']:
                    state['stop_reason'] = error.kind
                    break
                if error.kind == 'output_truncated' and config.get('provider') == 'openrouter' and token_budget == 4096:
                    token_budget = 16384
                    cache.setdefault('resume_options', {})[key] = {'max_tokens': token_budget}
                    save_json(path, cache)
                    next_gap = 15
                elif error.kind == 'provider_declined':
                    break
                elif error.retry_after and error.retry_after > 60:
                    state['stop_reason'] = 'provider_requested_long_wait'
                    break
                else:
                    next_gap = max(20, error.retry_after or 0)
            except Exception as error:
                state['errors'].append({'source_id': int(key), 'kind': 'local_error',
                                        'error_type': type(error).__name__})
                state['stop_reason'] = 'local_error'
                print('Local error:', type(error).__name__, flush=True)
                break
        if state['stop_reason']:
            break
        if not success:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                state['stop_reason'] = 'repeated_provider_failures'
                break
    save_json(path, cache)
    state['completed'] = len(cache['responses'])
    return cache, state
