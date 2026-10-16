"""Minimal Smoke retains LSL and task eval while excluding general benchmarks."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from reproduction.configuration import ROOT,read_plan,resolve
from reproduction.resource_validation import required_datasets

IGBO='Tommy0201/igbo_to_english_split'
NEG='assafbk/lsl-negatives-fineweb-edu'
BENCHMARKS={'openai/gsm8k','openai/openai_humaneval','google/IFEval'}


def test_minimal_smoke_config_only_changes_skip_flag():
    name='reproduction/configs/smoke_1p5b.json'
    old=json.loads(subprocess.check_output(['git','show',f'a7f1d91:{name}'],cwd=ROOT))
    current=read_plan('smoke_1p5b')
    expected=copy.deepcopy(old);expected['overrides']['skip_lm_eval']=True
    assert current==expected
    cfg=resolve(current)
    assert cfg.run_cl_eval and cfg.activate_cl_method and cfg.train_neg_gmm
    assert cfg.diagnostics_enabled and required_datasets(current)=={IGBO,NEG}


@pytest.mark.parametrize('name',['single_igbo','single_chemistry','single_cybersecurity','continual_7b_lsl','continual_7b_lora','continual_7b_microbatch2'])
def test_formal_configs_unchanged_and_require_benchmarks(name):
    path=f'reproduction/configs/{name}.json'
    assert (ROOT/path).read_text(encoding='utf-8')==subprocess.check_output(['git','show',f'a7f1d91:{path}'],cwd=ROOT,text=True,encoding='utf-8')
    plan=read_plan(name)
    assert plan['overrides']['skip_lm_eval'] is False
    assert BENCHMARKS<=required_datasets(plan)


@pytest.mark.parametrize('task,repo',[('gsm8k','openai/gsm8k'),('humaneval','openai/openai_humaneval'),('ifeval','google/IFEval')])
def test_resources_follow_enabled_benchmark_tasks(task,repo):
    plan=read_plan('smoke_1p5b');plan['overrides']['skip_lm_eval']=False
    plan['overrides']['lm_eval_tasks']=[task]
    assert required_datasets(plan)=={IGBO,NEG,repo}


@pytest.mark.parametrize('name,expected_error',[('smoke_1p5b','--parity-receipt'),('single_igbo','verify_evaluators.py')])
def test_actual_launch_preflight_only_requires_enabled_evaluator_audit(name,expected_error,tmp_path,monkeypatch,capsys):
    # Mock hardware boundary only; execute actual launcher resource selection and
    # audit policy. Stop before Parity validation, torchrun or model computation.
    from scripts import run_reproduction as launch
    from reproduction import lm_evaluation
    monkeypatch.setattr(lm_evaluation,'load_lm_eval',lambda:(None,None))
    monkeypatch.setattr(lm_evaluation,'require_nltk_resources',lambda *args:None)
    selected=[]
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,is_bf16_supported=lambda:True)))
    monkeypatch.setitem(sys.modules,'flash_attn',SimpleNamespace())
    monkeypatch.setattr(launch.platform,'system',lambda:'Linux')
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(launch,'validate_resource',lambda entry,*args:selected.append(entry['repo_id']))
    monkeypatch.setattr(sys,'argv',['run','--config',name,'--output-root',str(tmp_path/'runs'),
        '--evaluator-audit',str(tmp_path/'missing-audit.json'),'--acknowledge-issues'])
    with pytest.raises(SystemExit) as error:launch.main()
    assert error.value.code==2 and expected_error in capsys.readouterr().err
    if name=='smoke_1p5b':assert set(selected)=={'Qwen/Qwen2.5-1.5B-Instruct',IGBO,NEG}
    else:assert BENCHMARKS<=set(selected)
    assert not (tmp_path/'runs').exists()


def test_original_skipped_harness_returns_before_benchmark_load(tmp_path):
    tree=ast.parse((ROOT/'train_continual.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='lm_eval_harness')
    node.decorator_list=[]
    def forbidden(*args,**kwargs):raise AssertionError('benchmark/model loading must not occur')
    messages=[]
    namespace=dict(os=os,json=json,config=resolve(read_plan('smoke_1p5b')),local_rank=0,
        base_model_performance_cache_path=lambda config:str(tmp_path/'absent-cache.json'),
        logger=SimpleNamespace(warning=messages.append),simple_evaluate=forbidden,HFLM=forbidden,
        load_dataset=forbidden)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original-skipped-harness','exec'),namespace)
    total,scores,samples=namespace['lm_eval_harness'](None,None,['gsm8k','humaneval','ifeval'],return_dummy=True,use_base_model_performance_cache=True)
    assert total==0 and samples==[] and scores['total']==0
    assert messages and 'placeholders, not measured scores' in messages[0]
