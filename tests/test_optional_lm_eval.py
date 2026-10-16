import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from reproduction.configuration import ROOT
from reproduction import lm_evaluation as optional

@pytest.mark.parametrize('mode',['skipped','missing'])
def test_actual_train_import_with_lm_eval_blocked(mode):
    result=subprocess.run([sys.executable,str(ROOT/'tests/optional_lm_runner.py'),mode],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout.splitlines()[-1]);assert report['imported']
    if mode=='skipped':assert report['skipped'] and report['score']==0
    else:assert '0.4.10' in report['error']

def test_real_installed_full_factory_and_original_wrapper_behavior(monkeypatch):
    optional.load_lm_eval.cache_clear()
    from lm_eval.models.huggingface import HFLM
    wrapper,evaluate=optional.load_lm_eval()
    from lm_eval import simple_evaluate
    assert issubclass(wrapper,HFLM) and evaluate is simple_evaluate
    def init(self,**kwargs):self.pretrained=kwargs['pretrained']
    def generate(self,context,max_length,stop,**kwargs):return dict(context=context,max_length=max_length,stop=stop,kwargs=kwargs)
    monkeypatch.setattr(HFLM,'__init__',init);monkeypatch.setattr(HFLM,'_model_generate',generate)
    model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2,_attn_implementation='eager'))
    current=wrapper(pretrained=model)
    tree=ast.parse(subprocess.check_output(['git','show','baseline-lsl-original:train_continual.py'],cwd=ROOT,text=True,encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HFWithExternalCache')
    ns=dict(HFLM=HFLM);exec(compile(ast.Module(body=[cls],type_ignores=[]),'original-wrapper','exec'),ns)
    original=ns['HFWithExternalCache'](pretrained=model)
    assert current.num_layers==original.num_layers==2 and current.max_gen_toks==original.max_gen_toks==1024
    actual=current._model_generate('prompt',10,['stop'],temperature=0)
    expected=original._model_generate('prompt',10,['stop'],temperature=0)
    assert actual['context']==expected['context'] and actual['max_length']==expected['max_length']
    assert actual['kwargs']['temperature']==expected['kwargs']['temperature']
    assert type(actual['kwargs']['past_key_values']) is type(expected['kwargs']['past_key_values'])
    optional.load_lm_eval.cache_clear()

def test_wrong_full_version_fails_early(monkeypatch):
    monkeypatch.setattr(optional,'version',lambda name:'0.5.0')
    with pytest.raises(RuntimeError,match='0.4.10'):optional.require_lm_eval()

def test_actual_formal_harness_protocol_and_metrics_match_original(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'tests/optional_lm_runner.py'),'protocol',str(tmp_path)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout.splitlines()[-1]);assert report['protocol_equal']
    assert [c['batch_size'] for c in report['calls']]==[256,256,541]
    assert [c['num_fewshot'] for c in report['calls']]==[0,None,None]

def test_nltk_not_required_until_explicit_full_preflight(monkeypatch):
    import nltk
    def absent(*args,**kwargs):raise LookupError('controlled absent tokenizer data')
    monkeypatch.setattr(nltk.data,'find',absent)
    with pytest.raises(RuntimeError,match='punkt'):optional.require_nltk_resources()

def test_harness_protocol_only_adds_lazy_dependency_load():
    # Runtime wrapper test above accompanies this exhaustive protocol AST check.
    old=ast.parse(subprocess.check_output(['git','show','32930e6:train_continual.py'],cwd=ROOT,text=True,encoding='utf-8'))
    new=ast.parse((ROOT/'train_continual.py').read_text(encoding='utf-8'))
    before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='lm_eval_harness')
    after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='lm_eval_harness')
    assignments=[n for n in after.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='load_lm_eval']
    assert len(assignments)==1
    after.body.remove(assignments[0]);assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)

def test_real_full_environment_hflm_cpu_call_without_download():
    from scripts.check_environment import tiny_full_lm_check
    assert tiny_full_lm_check().startswith('PASS real HFLM call')
