import ast
import copy
import json
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
from datasets import Dataset,DatasetDict

from reproduction.configuration import ROOT,read_plan,resolve,resources
from reproduction.pilot_sampling import PilotSampling,SOURCES,select_indices,digest,validate_plan,check_saved_manifest
from reproduction.resource_validation import required_datasets
from scripts.gpu_parity_test import geometry


def test_pilot_plan_geometry_resources_and_formal_unchanged():
    plan=read_plan('continual_1p5b_pilot_seed123');c=resolve(plan);validate_plan(plan,c)
    assert geometry(c)==geometry(resolve(read_plan('continual_1p5b_lsl_a10')))
    assert c.gmm_skip_on_fail and c.train_neg_gmm and c.use_prepared_negatives_set
    assert c.diagnostics_level=='extended' and c.diagnostics_every_n_forwards==100
    assert c.save_cl_state and c.diagnostics_save_checkpoints and c.run_cl_eval
    assert len(required_datasets(plan))==6 and 'openai/gsm8k' not in required_datasets(plan)
    for name in ('continual_1p5b_lsl_a10','continual_7b_lsl'):
        path=f'reproduction/configs/{name}.json'
        assert (ROOT/path).read_text(encoding='utf-8')==subprocess.check_output(['git','show',f'b8e3504:{path}'],cwd=ROOT,text=True,encoding='utf-8')


def test_sampling_independent_reproducible_rng_unchanged():
    state=random.getstate();a=select_indices(5000,2000,123,'chemdata','train')
    select_indices(5000,2000,123,'cybersecurity_ift','train')
    assert a==select_indices(5000,2000,123,'chemdata','train')
    assert a!=select_indices(5000,2000,456,'chemdata','train')
    assert a!=select_indices(5000,2000,123,'igbo_translation','train')
    assert state==random.getstate() and len(a)==2000
    assert select_indices(1200,100,123,'igbo_translation','test')==select_indices(1200,100,123,'igbo_translation_benchmark_version','test')


def fixture(tmp_path,replay=None):
    plan=read_plan('continual_1p5b_pilot_seed123')
    manifest,_=resources(ROOT/'reproduction/resources.lock.json')
    pools={name:DatasetDict({split:Dataset.from_dict({'prompt':[f'{name}/{split}/{i}' for i in range(n)],'answer':['answer']*n})
                           for split,n in [('train',2200),('test',160)]}) for name in SOURCES}
    train=SimpleNamespace(load_dataset_splits=lambda name,seed:DatasetDict(pools[name]))
    sampler=PilotSampling(train,plan,manifest,tmp_path/'samples.json',replay).install()
    return sampler,train,pools


def test_real_dataset_selection_manifest_and_baseline_replay(tmp_path):
    s,train,pools=fixture(tmp_path/'lsl')
    for name in SOURCES:
        selected=train.load_dataset_splits(name,123)
        assert len(selected['test'])==100
        assert len(selected['train'])==(2000 if name in s.plan['task_order'] else 0)
    document=s.verify();check_saved_manifest(s.path,s.plan)
    other,baseline,_=fixture(tmp_path/'lora',s.path)
    other.plan=copy.deepcopy(other.plan);other.plan['overrides']['activate_cl_method']=False
    for name in reversed(SOURCES):baseline.load_dataset_splits(name,123)
    assert other.verify()==document
    for r in document['datasets'].values():
        assert r['indices_sha256']==digest(r['indices']) and all(x['revision'] for x in r['sources'])
    changed,_,_=fixture(tmp_path/'bad',s.path)
    changed.entries['AI4Chem/ChemData700K']['revision']='wrong'
    with pytest.raises(ValueError,match='mismatch'):changed.load('chemdata',123)


def test_recursion_preserves_full_benchmark_train_before_blending(tmp_path):
    s,train,pools=fixture(tmp_path)
    original=train.load_dataset_splits
    def upstream(name,seed):
        if name=='chemdata':
            benchmark=train.load_dataset_splits('chembench',seed)
            assert len(benchmark['train'])==2200
        return DatasetDict(pools[name])
    s.original=upstream
    assert len(train.load_dataset_splits('chemdata',123)['train'])==2000
    assert len(train.load_dataset_splits('chembench',123)['train'])==0


def test_manifest_content_overlap_and_corruption_refused(tmp_path):
    s,train,_=fixture(tmp_path)
    for name in SOURCES:train.load_dataset_splits(name,123)
    first=s.document['datasets']['igbo_translation/train']['row_sha256'][0]
    s.document['datasets']['chembench/test']['row_sha256'][0]=first
    with pytest.raises(ValueError,match='overlap'):s.verify()
    assert not s.path.exists()


def test_actual_build_dataloaders_uses_sampled_rows_for_training_and_gmm(tmp_path):
    s,train,pools=fixture(tmp_path)
    tree=ast.parse((ROOT/'train_continual.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_dataloaders')
    # Execute the actual construction logic; only worker/process and collate boundaries are bounded.
    negatives=DatasetDict({k:Dataset.from_dict({'text':['reference']*500}) for k in ('train','test')})
    real_load=train.load_dataset_splits
    def load(name,seed):return negatives if name=='lsl_negatives' else real_load(name,seed)
    ns=dict(load_dataset_splits=load,get_collate_fn_per_dataset=lambda *a:None,
            make_dl=lambda dataset,*a,**kw:dataset,multi_epoch_loader=lambda ds:ds,
            use_ddp=False,DATASET_BENCHMARK={'igbo_translation':'igbo_translation_benchmark_version','chemdata':'chembench','cybersecurity_ift':'seceval'},
            SFTCollate=lambda *a,**kw:None)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'real-build-dataloaders','exec'),ns)
    cfg=resolve(s.plan);loaders=ns['build_dataloaders'](cfg,None)
    for task in s.plan['task_order']:
        assert loaders[task][2]==2000 and len(loaders[task][1])==100
        assert loaders[task][0].to_dict()==loaders['gmm_pos_'+task][0].to_dict()
        record=s.document['datasets'][task+'/train']
        assert loaders[task][0]['prompt']==pools[task]['train'].select(record['indices'])['prompt']
    assert loaders['gmm_neg'][2]==4*16384//1024
    s.verify()


@pytest.mark.parametrize('change',['seed','rank','budget','epochs'])
def test_invalid_pilot_controls_refused(change):
    p=read_plan('continual_1p5b_pilot_seed123');c=resolve(p)
    if change=='seed':c.seed=456
    if change=='rank':c.lora_r=4
    if change=='budget':c.gmm_fit_num_tokens=1000000
    if change=='epochs':c.ft_epochs_per_dataset['chemdata']=3
    with pytest.raises(ValueError):validate_plan(p,c)


def test_pilot_dry_run_no_download_or_training(tmp_path):
    run=subprocess.run([sys.executable,str(ROOT/'scripts/run_reproduction.py'),'--config','continual_1p5b_pilot_seed123',
                        '--output-root',str(tmp_path),'--experiment-id','dry','--dry-run'],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    report=json.loads(run.stdout)
    assert report['effective_config']['gmm_fit_num_tokens']==16384 and report['effective_batch']['accumulation']==32
    assert not (tmp_path/'dry').exists()


def test_actual_upstream_chemistry_split_and_mixed_pool_preserved(tmp_path):
    from reproduction.data_access import original_data_helpers
    # Execute unchanged upstream transforms/splitting/blending on synthetic raw rows.
    # No Hub access is permitted in this controlled fixture.
    manifest,_=resources(ROOT/'reproduction/resources.lock.json')
    ns=original_data_helpers(manifest)
    raw=Dataset.from_dict({'instruction':[f'chemical training item {i}' for i in range(3500)],'input':['']*3500,'output':['answer']*3500})
    bench=Dataset.from_dict({'question':[f'benchmark item {i}' for i in range(800)],
                            'A':['a']*800,'B':['b']*800,'C':['c']*800,'D':['d']*800,'answer':['A']*800})
    def load(repo,**kw):
        if repo=='AI4Chem/ChemData700K':return raw
        if repo=='AI4Chem/ChemBench4K':return bench
        raise AssertionError('unexpected network/dataset')
    ns['load_dataset']=load
    module=SimpleNamespace(load_dataset_splits=ns['load_dataset_splits'])
    plan=read_plan('continual_1p5b_pilot_seed123')
    sampler=PilotSampling(module,plan,manifest,tmp_path/'samples.json').install()
    # Recursion is through the original function namespace, as in a real imported module.
    ns['load_dataset_splits']=module.load_dataset_splits
    selected=module.load_dataset_splits('chemdata',123)
    heldout=module.load_dataset_splits('chembench',123)['test']
    assert len(selected['train'])==2000 and len(selected['test'])==100
    assert any('benchmark item' in r['prompt'] for r in selected['train'])
    assert set(selected['train']['prompt']).isdisjoint(heldout['prompt'])
    assert set(selected['train']['prompt']).isdisjoint(selected['test']['prompt'])
    assert sampler.document['datasets']['chemdata/train']['original_count']==3000


@pytest.mark.parametrize('fails',[False,True])
def test_pilot_gmm_observation_counts_tokens_owners_and_removes_hook(tmp_path,fails):
    import torch
    from reproduction.training_progress import TrainingProgress
    class Backbone(torch.nn.Module):
        def forward(self,full_dataset_input):return None
    model=torch.nn.Module();model.model=Backbone();owner=torch.nn.Module()
    owner.is_cl_memory_owner=True;owner.is_disabled=[False,True];model.add_module('owner',owner)
    def fit(model,loader,tokenizer,config,which='pos'):
        model.model(full_dataset_input=[dict(attention_mask=torch.tensor([[0,1,1],[1,1,1]]))])
        if fails:raise RuntimeError('EM failed')
        return .5
    noop=lambda *a,**kw:None
    train=SimpleNamespace(fit_cl_memory=fit,main=noop,stage_context=noop,wandb_log=noop,eval_dataset=noop,
                          lm_eval_harness=noop,save_diagnostic_checkpoint=noop)
    mock_torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0))
    observer=TrainingProgress(train,tmp_path,mock_torch,gmm_details=True).install()
    rng=torch.get_rng_state().clone()
    try:
        if fails:
            with pytest.raises(RuntimeError,match='EM failed'):train.fit_cl_memory(model,None,None,None)
        else:
            assert train.fit_cl_memory(model,None,None,None)==.5
            records=[json.loads(line) for line in (tmp_path/'progress.jsonl').read_text().splitlines()]
            assert records[-1]['fit_tokens']==5
            assert records[-1]['disabled_owners']==[dict(owner='owner',adapter=1)]
        assert not model.model._forward_pre_hooks and torch.equal(rng,torch.get_rng_state())
    finally:observer.close()
    assert train.fit_cl_memory is fit


def test_pilot_artifact_acceptance_partial_fit_and_no_fake_scores(tmp_path):
    from reproduction.live_run import validate_artifacts
    s,train,_=fixture(tmp_path/'sampling')
    for name in SOURCES:train.load_dataset_splits(name,123)
    s.verify();plan=s.plan;exp=tmp_path/'run/experiment'
    (exp/'reproduction').mkdir(parents=True);(exp/'checkpoints').mkdir();(exp/'diagnostics').mkdir()
    (exp/'sampling_manifest.json').write_text(s.path.read_text())
    records=[]
    for stage,task in enumerate(plan['task_order']):
        records.append(dict(stage=stage,dataset=task,task_scores=[.5,.6],ppls=[2.,1.],lm_eval={},
            cross_task_scores={t:[.5,.6] for t in plan['task_order'] if t!=task},
            cross_ppls={t:[2.,1.] for t in plan['task_order'] if t!=task}))
        path=exp/'checkpoints'/f'cl_state_stage{stage}_{task}_epoch0.pt';path.write_bytes(b'test fixture')
        path.with_suffix('.json').write_text(json.dumps(dict(stage=stage,epoch=0,configuration={**plan['overrides'],'seed':123,'source_model_revision':plan['model_revision']})))
    metrics=exp/'reproduction/original_metrics.jsonl';metrics.write_text('\n'.join(map(json.dumps,records)))
    (exp/'reproduction/task_results.csv').write_text('synthetic test fixture')
    for family in ['gate_metrics','gmm_metrics','adapter_metrics']:
        (exp/'diagnostics'/f'{family}.rank0.jsonl').write_text('\n'.join(json.dumps({'stage':s}) for s in range(3)))
    fits=[dict(event='gmm_complete',which=w,active_owner_ratio=.75,fit_tokens=16384,disabled_owners=[{'owner':'a','adapter':0}]) for w in ['neg','pos','pos','pos']]
    (tmp_path/'run/progress.jsonl').write_text('\n'.join(map(json.dumps,fits)))
    assert validate_artifacts(tmp_path/'run',plan)['gmm_fits']==4
    records[0]['lm_eval']={'gsm8k':[0.,0.]};metrics.write_text('\n'.join(map(json.dumps,records)))
    with pytest.raises(ValueError,match='must not report'):validate_artifacts(tmp_path/'run',plan)


@pytest.mark.parametrize('mode',['pass','rank4','resource','memory'])
def test_pilot_real_preflight_keeps_resource_memory_and_parity_guards(tmp_path,monkeypatch,capsys,mode):
    import importlib.metadata
    from scripts import run_reproduction as launch
    from reproduction import live_run,environment_runtime,memory
    plan=read_plan('continual_1p5b_pilot_seed123');config=resolve(plan)
    cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,is_bf16_supported=lambda:True,
                         current_device=lambda:0,get_device_capability=lambda:(8,6),
                         get_device_properties=lambda i:SimpleNamespace(name='NVIDIA A10',total_memory=24*2**30))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda,__version__='test-torch',version=SimpleNamespace(cuda='12.8',hip=None)))
    monkeypatch.setitem(sys.modules,'flash_attn',SimpleNamespace())
    monkeypatch.setattr(launch.platform,'system',lambda:'Linux')
    monkeypatch.setattr(environment_runtime,'check_training_compatibility',lambda *a:None)
    monkeypatch.setattr(live_run,'check_idle_nvidia',lambda:None)
    monkeypatch.setattr(live_run,'runtime_imports',lambda **kw: {} if kw=={'include_lm_eval':False} else pytest.fail('Pilot imported lm-eval'))
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda *a:SimpleNamespace(free=100*2**30))
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}')
    monkeypatch.setattr(launch,'local_resource',lambda *a:model)
    monkeypatch.setattr(memory,'estimate',lambda *a:{'planning_range_gib':[10,30 if mode=='memory' else 12]})
    selected=[]
    def resource(entry,*a):
        if mode=='resource':raise ValueError('invalid resource')
        selected.append(entry['repo_id'])
    monkeypatch.setattr(launch,'validate_resource',resource)
    geo=geometry(config)
    if mode=='rank4':geo.update(lora_r=4,lora_alpha=8)
    receipt=tmp_path/'parity.json';receipt.write_text(json.dumps(dict(passed=True,core_hash=launch.code_hash(),
        model_revision=plan['model_revision'],geometry=geo,gmm_skip_on_fail=True,
        gmm_active_ratios=[[0,'neg',.75],[0,'pos',.75],[1,'pos',.75]],
        environment={'torch':'test-torch','transformers':'4.56.2','flash_attn':'2.8.3','capability':[8,6],'backend':'cuda'})))
    version=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'2.8.3' if name=='flash_attn' else version(name))
    monkeypatch.setattr(live_run,'stream_process',lambda *a:pytest.fail('preflight started training'))
    monkeypatch.setattr(sys,'argv',['run','--config','continual_1p5b_pilot_seed123','--preflight-only',
                                  '--parity-receipt',str(receipt),'--output-root',str(tmp_path/'runs'),'--acknowledge-issues'])
    if mode=='pass':
        launch.main();assert len(selected)==7
        assert 'PREFLIGHT PASS' in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit):launch.main()
        assert {'rank4':'geometry','resource':'invalid resource','memory':'planning range'}[mode] in capsys.readouterr().err
    assert not (tmp_path/'runs').exists()
