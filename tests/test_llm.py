import json
import pytest
from src.llm import signature, validate, parse_answer, ProviderFailure, run_cached, send_classification

def config():
    return {'model':'fixed:free','provider':'openrouter','prompt':'fixed instruction',
            'schema':{},'sample':[{'source_id':1,'model_text':'first'},
                                  {'source_id':2,'model_text':'second'}]}


def valid_answer():
    return {'label':2,'explanation':'An ordinary sentence.','needs_review':False}


def test_keys_and_resume(tmp_path):
    cfg=config()
    assert signature(cfg)==signature(dict(reversed(list(cfg.items()))))
    for name,value in [('provider','gemini'),('model','other:free'),('prompt','different'),
                       ('schema',{'type':'object'}),('sample',[])]:
        assert signature(cfg)!=signature(dict(cfg,**{name:value}))
    calls=[]
    def request(text,tokens):
        calls.append(text)
        return valid_answer(), {'max_tokens':tokens}
    path=tmp_path/'cache.json'
    first,state=run_cached(cfg,path,request,enabled=True,budget=1,sleep=lambda _:None)
    assert len(first['responses'])==1 and state['attempts']==1
    second,state=run_cached(cfg,path,request,enabled=True,budget=2,sleep=lambda _:None)
    assert calls==['first','second'] and len(second['responses'])==2


def test_disabled_never_reads_credentials_or_calls_network(tmp_path):
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected external access')
    cache,state=run_cached(config(),tmp_path/'cache.json',forbidden,enabled=False)
    assert state['attempts']==0 and not cache['responses']
    with pytest.raises(ProviderFailure,match='live_disabled'):
        send_classification('hello',provider='openrouter',model='fixed:free',prompt='',
            schema={},enabled=False,get_key=forbidden,post=forbidden)


def test_quota_and_strict_schema_preserve_progress(tmp_path):
    calls=[]
    def request(text,tokens):
        calls.append(text)
        if text=='second':raise ProviderFailure('quota_or_rate_limit',429)
        return valid_answer(),{}
    path=tmp_path/'cache.json'
    cache,state=run_cached(config(),path,request,enabled=True,budget=10,sleep=lambda _:None)
    assert calls==['first','second'] and set(cache['responses'])=={'1'}
    assert state['stop_reason']=='quota_or_rate_limit'
    assert set(json.loads(path.read_text())['responses'])=={'1'}
    for invalid in [dict(valid_answer(),label=True),dict(valid_answer(),explanation=''),
                    dict(valid_answer(),needs_review='false'),dict(valid_answer(),extra=1)]:
        with pytest.raises(ProviderFailure):validate(invalid)
    assert parse_answer('```json\n'+json.dumps(valid_answer())+'\n```')==valid_answer()
    with pytest.raises(ProviderFailure):parse_answer('not JSON')


def test_truncation_recovery_and_http_diagnostics(tmp_path):
    cfg=config();cfg['sample']=cfg['sample'][:1]
    token_budgets=[]
    def request(text,tokens):
        token_budgets.append(tokens)
        if tokens==4096:raise ProviderFailure('output_truncated',200)
        return valid_answer(),{'max_tokens':tokens}
    cache,state=run_cached(cfg,tmp_path/'cache.json',request,enabled=True,budget=2,sleep=lambda _:None)
    assert token_budgets==[4096,16384] and len(cache['responses'])==1
    class Reply:
        ok=True;status_code=200;headers={};content=b'{}'
        def __init__(self,body):self.body=body
        def json(self):return self.body
    result,details=send_classification('hello',provider='openrouter',model='fixed:free',
        prompt='fixed',schema={},enabled=True,get_key=lambda:'test-only-key',
        post=lambda *a,**k:Reply({'choices':[{'finish_reason':'stop',
             'message':{'content':json.dumps(valid_answer())}}]}))
    assert result==valid_answer()
    with pytest.raises(ProviderFailure,match='output_truncated'):
        send_classification('hello',provider='openrouter',model='fixed:free',prompt='fixed',schema={},
            enabled=True,get_key=lambda:'test-only-key',post=lambda *a,**k:Reply({
            'choices':[{'finish_reason':'length','message':{'content':''}}],
            'usage':{'completion_tokens':4096,'completion_tokens_details':{'reasoning_tokens':4096}}}))
