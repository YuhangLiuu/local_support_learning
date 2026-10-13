"""CPU preparation tests: no model/data downloads and no training launcher calls."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import copy

import pytest

from reproduction.configuration import ROOT,COMMON,read_plan,resolve,serialized
from reproduction.memory import estimate
from scripts.collect_results import rows_for_run,retention


@pytest.mark.parametrize('name',['single_igbo','single_chemistry','single_cybersecurity','continual_7b_lsl',
                              'continual_7b_lora','smoke_1p5b','smoke_7b','continual_7b_microbatch2'])
def test_configs_complete_and_no_environment_leak(name,monkeypatch):
    monkeypatch.setenv('LORA_R','999');monkeypatch.setenv('FT_LEARNING_RATE','9');monkeypatch.setenv('FT_GLOBAL_BATCH_SIZE','999')
    plan=read_plan(name);c=resolve(plan)
    assert c.ft_learning_rate==1e-4 and c.lora_alpha==2*c.lora_r and c.ft_global_batch_size!=999
    assert c.wandb_logger is False and c.diagnostics_enabled and c.diagnostics_save_checkpoints
    assert c.seed==123 and c.seed in plan['seeds']
    if name=='single_cybersecurity':assert (c.lora_r,c.ft_global_batch_size,c.gmm_n_pos_comps,c.smoothing_alpha)==(512,16,32,.3)
    if name=='single_chemistry':assert (c.lora_r,c.gmm_n_pos_comps,c.smoothing_alpha)==(128,32,.1)
    if name=='continual_7b_lsl':
        assert (c.lora_r,c.gmm_n_pos_comps,c.gmm_n_neg_comps,c.smoothing_alpha,c.lm_eval_limit)==(128,16,32,.1,300)
        assert plan['task_order']==['igbo_translation','chemdata','cybersecurity_ift']


def test_global_microbatch_validation_and_declared_seed():
    plan=read_plan('continual_7b_lsl')
    assert resolve(plan,microbatch=2,world_size=4).ft_global_batch_size==32
    with pytest.raises(ValueError):resolve(plan,microbatch=3)
    with pytest.raises(ValueError):resolve(plan,seed=999)


@pytest.mark.parametrize('script,args',[
 ('run_reproduction.py',['--config','continual_7b_lsl','--microbatch','2']),
 ('gpu_parity_test.py',['--config','smoke_7b','--output','tmp/not-created-parity']),
 ('prefetch_resources.py',[]),('verify_datasets.py',[]),('verify_evaluators.py',[]),
 ('collect_results.py',['--runs','missing-run','--output','tmp/not-created-collection']),('check_gpu_env.py',[])])
def test_dry_run_without_gpu_resources_or_side_effects(script,args):
    proc=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args,'--dry-run'],cwd=ROOT,
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    assert isinstance(json.loads(proc.stdout),dict)
    assert not (ROOT/'tmp/not-created-parity').exists()


def test_launch_conflict_and_gpu_refusal(tmp_path):
    (tmp_path/'existing').mkdir()
    command=[sys.executable,str(ROOT/'scripts/run_reproduction.py'),'--config','smoke_7b','--output-root',str(tmp_path)]
    p=subprocess.run(command+['--experiment-id','existing','--dry-run'],capture_output=True,text=True)
    assert p.returncode!=0 and 'already exists' in p.stderr
    p=subprocess.run(command+['--experiment-id','must-not-start'],capture_output=True,text=True)
    assert p.returncode!=0 and not (tmp_path/'must-not-start').exists()


def test_qwen_memory_parameters_match_real_meta_model():
    torch=pytest.importorskip('torch');pytest.importorskip('pomegranate')
    # Entire real architecture on meta: zero model weights allocated/downloaded.
    from configuration_qwen2 import Qwen2Config
    from modeling_qwen2 import Qwen2ForCausalLM
    cfg=dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,vocab_size=256,
             num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=False)
    with torch.device('meta'):model=Qwen2ForCausalLM(Qwen2Config(**cfg,attn_implementation='eager'))
    output=estimate(cfg,serialized(resolve(read_plan('smoke_7b'))),2)
    assert output['base_parameters']==sum(p.numel() for p in model.parameters())
    assert output['base_fp32_gib']==output['base_parameters']*4/2**30


def test_original_helpers_and_synthetic_evaluator_cases():
    pytest.importorskip('datasets')
    from reproduction.data_access import original_data_helpers
    ns=original_data_helpers(dict(resources=[]))
    assert ns['_seceval_f1']('AB','AB')==1.
    assert ns['_seceval_f1']('A','AB')==pytest.approx(2/3)
    assert ns['_seceval_f1']('answer B','B')==0.
    assert ns['_chemdata_to_prepped'](dict(instruction=' x ',input=' y ',output='z'))==dict(prompt='x\n\ny',answer='z')
    assert ns['_igbo_translation_to_prepped'](dict(igbo='Igbo',english='English'))['answer']=='Igbo'


def make_run(root,complete=True):
    (root/'experiment/reproduction').mkdir(parents=True)
    plan=read_plan('single_igbo');plan['overrides']['ft_epochs_per_dataset']={'igbo_translation':1}
    c=resolve(plan)
    (root/'launch.json').write_text(json.dumps(dict(experiment_id='unit',plan=plan,tier=plan['tier'],effective_config=serialized(c))))
    (root/'completion.json').write_text(json.dumps(dict(complete=complete)))
    record=dict(stage=0,dataset='igbo_translation',task_scores=[.1,.2],cross_task_scores={},ppls=[5,4],cross_ppls={},
                lm_eval={k:[.5,.4] for k in ['gsm8k','humaneval','ifeval','total']},git_commit='test')
    (root/'experiment/reproduction/original_metrics.jsonl').write_text(json.dumps(record)+'\n')


def test_complete_results_only_and_metric_epoch_identity(tmp_path):
    make_run(tmp_path/'good')
    rows=rows_for_run(tmp_path/'good')
    assert len(rows)==12 and {r['epoch'] for r in rows}=={-1,0}
    assert all(r['seed']==123 and r['configuration_signature'] for r in rows)
    make_run(tmp_path/'bad',False)
    with pytest.raises(ValueError,match='incomplete'):rows_for_run(tmp_path/'bad')
    assert retention(dict(gsm8k=.5,humaneval=.25,ifeval=.75),dict(gsm8k=.4,humaneval=.2,ifeval=.6))==pytest.approx(.8)


def test_all_shell_scripts_parse():
    candidates=[Path('C:/Program Files/Git/bin/bash.exe'),Path('/bin/bash')]
    bash=next((p for p in candidates if p.exists()),None)
    if bash is None:pytest.skip('bash unavailable')
    for script in (ROOT/'scripts').glob('*.sh'):
        assert subprocess.run([str(bash),'-n',str(script)],capture_output=True).returncode==0


@pytest.mark.parametrize('full,prompt,limit,expected',[(12,8,6,4),(5,3,6,2),(20,8,6,6)])
def test_optional_truncation_patch_known_answer(full,prompt,limit,expected):
    kept=min(full,limit)
    corrected_prompt=max(0,prompt-max(0,full-kept))
    assert kept-corrected_prompt==expected
    patch=(ROOT/'reproduction/patches/left_truncation_mask.patch').read_text()
    assert patch.count('kept_prompt = max(0, plen - removed)')==2
    assert 'kept_prompt = max(0, plen - removed)' not in (ROOT/'train_continual.py').read_text(encoding='utf-8')


@pytest.mark.parametrize('script',['prefetch_resources.py','verify_datasets.py','verify_evaluators.py'])
def test_server_only_guards_prevent_local_downloads(script):
    if sys.platform=='linux':pytest.skip('Windows safety guard test')
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'server-only' in result.stderr


def test_deleted_resource_manifest_is_metadata_only():
    manifest=json.loads((ROOT/'reproduction/resources.lock.json').read_text())
    for entry in manifest['resources']:
        assert entry['revision'] and entry['status'] in {'not_cached','downloaded','tokenizer_only','download_failed','inaccessible','inventoried'}
        if entry['status']=='not_cached':assert not entry['verified_files']
