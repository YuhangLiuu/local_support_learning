import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from reproduction.configuration import ROOT,read_plan,resolve,resources
from reproduction.live_run import stream_process,validate_artifacts,check_idle_nvidia
from reproduction.training_progress import TrainingProgress,bounded_lm_loader


@pytest.mark.parametrize('name,budget',[('continual_1p5b_lsl',1000000),('continual_1p5b_lsl_a10',65536)])
def test_dedicated_config(name,budget):
    p=read_plan(name);c=resolve(p)
    assert p['seeds']==[123] and c.seed==123
    assert p['task_order']==['igbo_translation','chemdata','cybersecurity_ift']
    assert (c.lora_r,c.lora_alpha,c.ft_learning_rate,c.seq_len_train)==(128,256,1e-4,1024)
    assert c.ft_global_batch_size//c.local_batch_size==32 and c.local_batch_size==1
    assert (c.gmm_n_pos_comps,c.gmm_n_neg_comps,c.jl_k,c.smoothing_alpha)==(16,32,256,.1)
    assert c.gmm_fit_num_tokens==budget and c.save_cl_state and c.diagnostics_save_checkpoints
    assert c.ft_epochs_per_dataset==dict(igbo_translation=3,chemdata=3,cybersecurity_ift=1)
    assert not c.skip_lm_eval and c.run_cl_eval and p['tier']=='hardware_adapted'
    _,entries=resources(ROOT/'reproduction/resources.lock.json')
    assert p['model_revision']==entries[c.model_name]['revision']


@pytest.mark.parametrize('extra',[[],['--seed','456'],['--microbatch','2'],['--diagnostics','off']])
def test_dry_run_seed_and_batch_constraints(tmp_path,extra):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_reproduction.py'),'--config','continual_1p5b_lsl_a10',
                           '--output-root',str(tmp_path),'--dry-run',*extra],capture_output=True,text=True)
    if extra:assert result.returncode!=0
    else:
        assert result.returncode==0,result.stderr
        report=json.loads(result.stdout)
        assert report['effective_batch']['accumulation']==32
        assert report['effective_config']['seed']==123
        assert '-m' in report['worker_command'] and 'reproduction.worker' in report['worker_command']
    assert not list(tmp_path.iterdir())


def test_live_tee_preserves_traceback_exit_and_progress(tmp_path,capsys):
    script='import sys; print("Step 1/10",flush=True); print("Traceback: test error",file=sys.stderr); sys.exit(7)'
    path=tmp_path/'process.log'
    assert stream_process([sys.executable,'-u','-c',script],path,ROOT,os.environ.copy())==7
    text=capsys.readouterr().out
    assert 'Step 1/10' in text and 'Traceback: test error' in text
    assert path.read_text()==text


def test_idle_gpu_guard():
    check_idle_nvidia(lambda *args,**kwargs:f'{os.getpid()}, python, 200\n')
    with pytest.raises(ValueError,match='already running'):
        check_idle_nvidia(lambda *args,**kwargs:'999999, training.py, 10000\n')


def test_hardware_lm_batch_only_changes_batch():
    calls=[]
    def record(*args,**kwargs):calls.append((args,kwargs));return 'original-result'
    cls,eval_fn=bounded_lm_loader(lambda:(record,record),1)()
    assert cls(pretrained='same-model',batch_size=256,max_length=8192,dtype='bf16')=='original-result'
    assert eval_fn(model='wrapper',batch_size=541,tasks=['ifeval'],limit=300,apply_chat_template=True)=='original-result'
    assert calls==[((),dict(pretrained='same-model',batch_size=1,max_length=8192,dtype='bf16')),
                   ((),dict(model='wrapper',batch_size=1,tasks=['ifeval'],limit=300,apply_chat_template=True))]


def test_progress_wrappers_preserve_real_cpu_math_rng_and_restore(tmp_path):
    import torch
    def trial(observe):
        torch.manual_seed(123)
        model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        losses=[]
        train=SimpleNamespace(world_size=1,stage_context=lambda *a,**k:None,wandb_log=lambda *a,**k:None,
                              fit_cl_memory=lambda *a,**k:1,eval_dataset=lambda *a,**k:(.5,2),
                              lm_eval_harness=lambda *a,**k:({},{}),save_diagnostic_checkpoint=lambda *a:None)
        def main(config,model,tokenizer,dataloaders,tasks):
            for step in range(2):
                train.stage_context(model,tasks['current'],0,0)
                optimizer.zero_grad();loss=model(torch.ones(1,3)).square().mean();loss.backward()
                train.wandb_log({'Train/train_loss':loss.detach().item(),'Train/lr':1e-4})
                optimizer.step();losses.append(loss.detach().clone())
            train.fit_cl_memory(model,None,None,config,which='pos')
            train.eval_dataset(model,dataset_name='igbo_translation')
            train.save_diagnostic_checkpoint(model,config,'igbo_translation',0)
            return 'unchanged'
        train.main=main
        original={key:getattr(train,key) for key in vars(train) if callable(getattr(train,key))}
        progress=TrainingProgress(train,tmp_path,SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0))).install() if observe else None
        try:
            result=train.main(SimpleNamespace(task_idx=0,ft_max_num_epochs=1,ft_global_batch_size=32),model,None,
                              {'igbo_translation':(None,None,64)},{'current':'igbo_translation'})
        finally:
            if progress:progress.close()
        assert result=='unchanged'
        assert all(getattr(train,key) is value for key,value in original.items())
        return [p.detach().clone() for p in model.parameters()],[p.grad.clone() for p in model.parameters()],losses,torch.get_rng_state()
    left,right=trial(False),trial(True)
    assert all(torch.equal(a,b) for part1,part2 in zip(left[:3],right[:3]) for a,b in zip(part1,part2))
    assert torch.equal(left[3],right[3])
    records=[json.loads(line) for line in (tmp_path/'progress.jsonl').read_text().splitlines()]
    assert [r['progress'] for r in records if r['event']=='training']==['1/2','2/2']


def artifact_fixture(root):
    p=read_plan('continual_1p5b_lsl_a10');experiment=root/'experiment'
    (experiment/'reproduction').mkdir(parents=True);(experiment/'checkpoints').mkdir();(experiment/'diagnostics').mkdir()
    records=[];fits=[dict(event='gmm_complete',which='neg',active_owner_ratio=1)]
    for stage,task in enumerate(p['task_order']):
        epochs=p['overrides']['ft_epochs_per_dataset'][task];values=[.5]*(epochs+1)
        records.append(dict(stage=stage,dataset=task,task_scores=values,ppls=values,
                            cross_task_scores={t:values for t in p['task_order'] if t!=task},
                            cross_ppls={t:values for t in p['task_order'] if t!=task},
                            lm_eval={t:values for t in p['overrides']['lm_eval_tasks']}))
        for epoch in range(epochs):
            path=experiment/'checkpoints'/f'cl_state_stage{stage}_{task}_epoch{epoch}.pt'
            path.write_bytes(b'synthetic; no real checkpoint claim')
            path.with_suffix('.json').write_text(json.dumps(dict(stage=stage,epoch=epoch,configuration={
                **p['overrides'],'seed':123,'source_model_revision':p['model_revision']})))
            fits.append(dict(event='gmm_complete',which='pos',active_owner_ratio=1))
    (experiment/'reproduction/original_metrics.jsonl').write_text('\n'.join(map(json.dumps,records)))
    (experiment/'reproduction/task_results.csv').write_text('synthetic,task,matrix\n')
    for family in ('gate_metrics','gmm_metrics','adapter_metrics'):
        (experiment/'diagnostics'/f'{family}.rank0.jsonl').write_text('\n'.join(json.dumps({'stage':s}) for s in range(3)))
    (root/'progress.jsonl').write_text('\n'.join(map(json.dumps,fits)))
    return p


def test_complete_artifacts(tmp_path):
    p=artifact_fixture(tmp_path);result=validate_artifacts(tmp_path,p)
    assert result['stages']==3 and len(result['checkpoints'])==7 and result['gmm_fits']==8


@pytest.mark.parametrize('broken',['checkpoint','stage','diagnostics','gmm','revision','nan'])
def test_incomplete_artifacts_rejected(tmp_path,broken):
    p=artifact_fixture(tmp_path);exp=tmp_path/'experiment'
    if broken=='checkpoint':next((exp/'checkpoints').glob('*.pt')).unlink()
    elif broken=='diagnostics':(exp/'diagnostics/adapter_metrics.rank0.jsonl').write_text('{}')
    elif broken=='gmm':(tmp_path/'progress.jsonl').write_text('{}')
    elif broken=='revision':
        side=next((exp/'checkpoints').glob('*.json'));data=json.loads(side.read_text());data['configuration']['source_model_revision']='wrong';side.write_text(json.dumps(data))
    else:
        path=exp/'reproduction/original_metrics.jsonl';rows=path.read_text().splitlines()
        if broken=='stage':rows.pop()
        else:
            row=json.loads(rows[0]);row['ppls'][0]=float('nan');rows[0]=json.dumps(row)
        path.write_text('\n'.join(rows))
    with pytest.raises(ValueError):validate_artifacts(tmp_path,p)


@pytest.mark.parametrize('mode',['pass','million_tokens','stale_receipt'])
def test_actual_launcher_strict_preflight_no_training(tmp_path,monkeypatch,capsys,mode):
    from scripts import run_reproduction as launch
    from reproduction import live_run,environment_runtime,lm_evaluation
    import importlib.metadata
    config_name='continual_1p5b_lsl' if mode=='million_tokens' else 'continual_1p5b_lsl_a10'
    plan=read_plan(config_name);config=resolve(plan)
    props=SimpleNamespace(name='NVIDIA A10',total_memory=24*2**30)
    cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,is_bf16_supported=lambda:True,
                         current_device=lambda:0,get_device_capability=lambda:(8,6),get_device_properties=lambda i:props)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda,__version__='test-torch',version=SimpleNamespace(cuda='12.8',hip=None)))
    monkeypatch.setitem(sys.modules,'flash_attn',SimpleNamespace())
    monkeypatch.setattr(environment_runtime,'check_training_compatibility',lambda *a:None)
    monkeypatch.setattr(launch.platform,'system',lambda:'Linux')
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda *a:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(live_run,'check_idle_nvidia',lambda:None)
    monkeypatch.setattr(live_run,'runtime_imports',lambda:{'transformers':{'version':'4.56.2','path':'synthetic/site-packages'}})
    monkeypatch.setattr(lm_evaluation,'load_lm_eval',lambda:(None,None))
    monkeypatch.setattr(lm_evaluation,'require_nltk_resources',lambda *a:None)
    original_version=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'2.8.3' if name=='flash_attn' else original_version(name))
    model_dir=tmp_path/'model';model_dir.mkdir()
    (model_dir/'config.json').write_text(json.dumps(dict(hidden_size=1536,intermediate_size=8960,num_hidden_layers=28,
        vocab_size=151936,num_attention_heads=12,num_key_value_heads=2,tie_word_embeddings=True)))
    selected=[]
    monkeypatch.setattr(launch,'validate_resource',lambda entry,*args:selected.append(entry['repo_id']))
    monkeypatch.setattr(launch,'local_resource',lambda *args:model_dir)
    receipt=tmp_path/'receipt.json'
    from scripts.gpu_parity_test import geometry
    receipt.write_text(json.dumps(dict(passed=True,core_hash='stale' if mode=='stale_receipt' else launch.code_hash(),
        model_revision=plan['model_revision'],geometry=geometry(config),
        gmm_skip_on_fail=config.gmm_skip_on_fail,gmm_active_ratios=[[0,'neg',.75],[0,'pos',.75],[1,'pos',.75]],
        environment=dict(torch='test-torch',transformers='4.56.2',flash_attn='2.8.3',capability=[8,6],backend='cuda'))))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(versions={'lm_eval':'0.4.10'},humaneval_canonical_solution={'pass@1':1})))
    output=tmp_path/'runs'
    monkeypatch.setattr(sys,'argv',['run','--config',config_name,'--seed','123','--preflight-only',
        '--parity-receipt',str(receipt),'--evaluator-audit',str(audit),'--output-root',str(output),'--acknowledge-issues'])
    never=SimpleNamespace(called=False)
    def forbidden(*args,**kwargs):never.called=True;raise AssertionError('must not train during preflight')
    monkeypatch.setattr(live_run,'stream_process',forbidden)
    if mode=='pass':
        launch.main()
        assert 'PREFLIGHT PASS' in capsys.readouterr().out
        assert len(selected)==10  # actual model + nine required datasets selected
    else:
        with pytest.raises(SystemExit) as error:launch.main()
        assert error.value.code==2
        message=capsys.readouterr().err
        assert ('million-token' if mode=='million_tokens' else 'stale') in message
    assert not never.called and not output.exists()
