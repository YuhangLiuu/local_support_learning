"""Real filesystem/child-process checks; no Hub downloads or GPU execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from reproduction import configuration as config
from reproduction.configuration import ROOT,local_resource,resolve_cache_root,safe_resource_file
from reproduction.resource_validation import validate_resource,MARKER,required_datasets
from reproduction.gpu_environment import inventory,platform_commands
from scripts.gpu_parity_test import run_worker


def entry():
    blobs={'config.json':b'{}','tokenizer.json':b'{}','model.safetensors':b'fake weights'}
    return dict(kind='model',repo_id='Qwen/tiny',revision='a'*40,status='not_cached',
                local_path='.cache\\lsl/model\\Qwen--tiny',
                files=[dict(name=n,size=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in blobs.items()]),blobs


def cached(root,marker=True):
    item,blobs=entry();directory=local_resource(item,root)
    directory.mkdir(parents=True)
    for name,blob in blobs.items():(directory/name).write_bytes(blob)
    if marker:(directory/MARKER).write_text(json.dumps(dict(repo_id=item['repo_id'],revision=item['revision'])))
    return item,directory


@pytest.mark.parametrize('legacy',['.cache/lsl/model/Qwen--tiny','.cache\\lsl\\model\\Qwen--tiny','.cache/lsl\\model/Qwen--tiny'])
def test_portable_legacy_relative_paths(legacy,monkeypatch):
    monkeypatch.delenv('LSL_CACHE_ROOT',raising=False)
    item,_=entry();item['local_path']=legacy
    assert local_resource(item)==(ROOT/'.cache/lsl/model/Qwen--tiny').resolve()


def test_cache_root_precedence(tmp_path,monkeypatch):
    monkeypatch.setenv('LSL_CACHE_ROOT',str(tmp_path/'env'))
    manifest=dict(cache_root='relative/cache')
    assert resolve_cache_root(tmp_path/'cli',manifest)==(tmp_path/'cli').resolve()
    assert resolve_cache_root(None,manifest)==(tmp_path/'env').resolve()
    monkeypatch.delenv('LSL_CACHE_ROOT')
    assert resolve_cache_root(None,manifest)==ROOT/'relative/cache'
    assert resolve_cache_root()==ROOT/'.cache/lsl'


@pytest.mark.parametrize('value',['reproduction/configs/smoke_1p5b.json','reproduction\\configs/smoke_1p5b.json'])
def test_config_paths_portable_and_relative_to_repository(value,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.read_plan(value)['name']=='smoke_1p5b'


def test_legacy_absolute_migration_requires_override(tmp_path,monkeypatch):
    monkeypatch.delenv('LSL_CACHE_ROOT',raising=False)
    item,_=entry();item['local_path']='C:\\Users\\old\\cache\\model\\Qwen--tiny'
    with pytest.raises(ValueError,match='legacy absolute'):local_resource(item)
    assert local_resource(item,tmp_path)==tmp_path/'model/Qwen--tiny'
    assert item['revision']=='a'*40


@pytest.mark.parametrize('value',['C:\\old\\cache','C:relative','\\\\old-server\\cache'])
def test_foreign_windows_root_rejected_on_linux(value,monkeypatch):
    monkeypatch.setattr(config,'os',SimpleNamespace(name='posix',environ={}))
    with pytest.raises(ValueError,match='Windows absolute'):config.portable_path(value)


@pytest.mark.parametrize('name',['../escape','a/../../escape','/absolute','C:\\escape','a\\..\\escape','x:stream','.', 'bad\x00path','bad?name'])
def test_unsafe_file_paths_rejected(tmp_path,name):
    with pytest.raises(ValueError):safe_resource_file(tmp_path,name)


@pytest.mark.parametrize('field,value',[('repo_id','../escape'),('repo_id','Qwen/../../escape'),('kind','../model'),('local_path','.cache/../escape/model/Qwen--tiny'),('local_path','.cache/lsl/model/WRONG')])
def test_resource_escape_rejected(tmp_path,field,value):
    item,_=entry();item[field]=value
    with pytest.raises(ValueError):local_resource(item,tmp_path)


def test_symlink_escape_rejected(tmp_path):
    outside=tmp_path/'outside';outside.mkdir()
    root=tmp_path/'cache';root.mkdir()
    try:(root/'link').symlink_to(outside,target_is_directory=True)
    except OSError:
        if os.name!='nt':raise
        link=str(root/'link').replace("'","''");target=str(outside).replace("'","''")
        result=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',f"$null = New-Item -ItemType Junction -Path '{link}' -Target '{target}' -ErrorAction Stop"],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    with pytest.raises(ValueError,match='escapes'):safe_resource_file(root,'link/weights')


def test_missing_model_reports_identity_revision_and_root(tmp_path):
    item,_=entry()
    with pytest.raises(ValueError) as error:validate_resource(item,tmp_path)
    assert all(text in str(error.value) for text in ['Qwen/tiny','a'*40,str(tmp_path),'prefetch_resources.py','config.json'])


def test_actual_cache_overrides_stale_status_and_is_read_only(tmp_path):
    item,directory=cached(tmp_path)
    before={p.name:p.read_bytes() for p in directory.iterdir()}
    assert validate_resource(item,tmp_path)==directory
    assert item['status']=='not_cached'
    assert before=={p.name:p.read_bytes() for p in directory.iterdir()}


def test_incomplete_wrong_revision_and_corrupt_hash(tmp_path):
    item,directory=cached(tmp_path)
    (directory/MARKER).write_text(json.dumps(dict(repo_id=item['repo_id'],revision='b'*40)))
    with pytest.raises(ValueError,match='revision'):validate_resource(item,tmp_path)
    (directory/MARKER).write_text(json.dumps(dict(repo_id=item['repo_id'],revision=item['revision'])))
    (directory/'model.safetensors').write_bytes(b'bad! weights')
    with pytest.raises(ValueError,match='sha256'):validate_resource(item,tmp_path,verify_hash=True)
    (directory/'config.json').unlink()
    with pytest.raises(ValueError,match='config.json'):validate_resource(item,tmp_path)


def test_legacy_cache_without_marker_needs_revision_evidence(tmp_path):
    item,directory=cached(tmp_path,marker=False)
    assert validate_resource(item,tmp_path)==directory  # pinned hashes of every required file
    for f in item['files']:f['sha256']=None
    with pytest.raises(ValueError,match='provenance'):validate_resource(item,tmp_path)
    for f in item['files']:
        meta=directory/'.cache/huggingface/download'/(f['name']+'.metadata')
        meta.parent.mkdir(parents=True,exist_ok=True);meta.write_text(item['revision']+'\netag\n0\n')
    assert validate_resource(item,tmp_path)==directory


def test_download_retry_reuse_and_lock_immutable(tmp_path,monkeypatch):
    import huggingface_hub
    from scripts.prefetch_resources import download
    item,blobs=entry();manifest=dict(schema_version=1,cache_root='.cache/lsl',resources=[item])
    lock=tmp_path/'resources.json';lock.write_text(json.dumps(manifest));before=lock.read_bytes();calls=[]
    def snapshot(repo,**kwargs):
        calls.append(kwargs)
        assert kwargs['revision']=='a'*40 and repo=='Qwen/tiny'
        if len(calls)==1:raise ConnectionError('synthetic interruption')
        directory=Path(kwargs['local_dir']);directory.mkdir(parents=True,exist_ok=True)
        for n,b in blobs.items():(directory/n).write_bytes(b)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',snapshot)
    cache=tmp_path/'cache'
    with pytest.raises(RuntimeError,match='safe to rerun'):download(manifest,lock,models=True,cache_root=cache)
    download(manifest,lock,models=True,cache_root=cache,verify_hash=True)
    download(manifest,lock,models=True,cache_root=cache,verify_hash=True)
    assert len(calls)==2 and lock.read_bytes()==before
    assert validate_resource(item,cache,verify_hash=True).is_dir()


@pytest.mark.parametrize('flags,expected_count',[( ['--models'],1),(['--models','--datasets'],3)])
def test_plan_selects_only_requested_resources_and_dry_run(flags,expected_count):
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/prefetch_resources.py'),'--config','smoke_1p5b',*flags,'--dry-run'],capture_output=True,text=True,timeout=15)
    assert proc.returncode==0,proc.stderr
    report=json.loads(proc.stdout)
    assert len(report['resources'])==expected_count
    assert not any(r['repo_id']=='Qwen/Qwen2.5-7B-Instruct' for r in report['resources'])
    assert not (ROOT/'.cache/lsl/model/Qwen--Qwen2.5-1.5B-Instruct').exists()


def fake_torch(kind,available=True):
    props=SimpleNamespace(name=kind+' test GPU',total_memory=192*2**30,gcnArchName='gfx942' if kind=='rocm' else None)
    cuda=SimpleNamespace(is_available=lambda:available,device_count=lambda:1 if available else 0,
        current_device=lambda:0,get_device_properties=lambda n:props,
        get_device_capability=lambda *args:(0,0) if kind=='rocm' else (8,0),
        device=lambda n:nullcontext(),is_bf16_supported=lambda **kw:True)
    return SimpleNamespace(__version__='test',version=SimpleNamespace(hip='7.2' if kind=='rocm' else None,cuda='12.8' if kind=='cuda' else None),cuda=cuda)


@pytest.mark.parametrize('kind',['cuda','rocm','cpu'])
def test_gpu_inventory_simulation(kind):
    result=inventory(fake_torch(kind,available=kind!='cpu'))
    assert result['backend']==kind
    assert len(result['gpus'])==int(kind!='cpu')
    commands=platform_commands(kind)
    if kind=='rocm':assert commands and all('nvidia' not in str(c) and 'nvcc' not in str(c) for c in commands)
    if kind=='cpu':assert commands==[]


@pytest.mark.parametrize('kind',['rocm','cuda','cpu'])
def test_environment_cli_accepts_rocm_without_cuda_runtime(kind,monkeypatch,capsys):
    from scripts import check_gpu_env
    monkeypatch.setitem(sys.modules,'torch',fake_torch(kind,available=kind!='cpu'))
    monkeypatch.setitem(sys.modules,'flash_attn',SimpleNamespace())
    monkeypatch.setattr(check_gpu_env.metadata,'version',lambda name:'2.8.3')
    monkeypatch.setattr(check_gpu_env.subprocess,'check_output',lambda *a,**kw:'simulated tool info')
    monkeypatch.setattr(sys,'argv',['check_gpu_env','--config','smoke_1p5b','--require-gpu'])
    if kind=='cpu':
        with pytest.raises(SystemExit) as error:check_gpu_env.main()
        assert error.value.code==2
    else:check_gpu_env.main()
    report=json.loads(capsys.readouterr().out)
    assert report['backend']==kind
    if kind=='rocm':assert report['cuda_runtime'] is None and report['hip_runtime']=='7.2'


def test_worker_failure_reports_actual_root_exception_and_log(tmp_path):
    log=tmp_path/'off.log'
    with pytest.raises(RuntimeError) as error:
        run_worker([sys.executable,'-c','raise RuntimeError("synthetic root cause")'],log,dict(os.environ))
    assert str(log) in str(error.value) and 'synthetic root cause' in str(error.value)
    assert log.is_file()


def test_vllm_conflict_is_reported_without_changing_versions(monkeypatch):
    from scripts import check_gpu_env
    versions=dict(vllm='0.unknown',transformers='4.56.2',torch='2.12.0')
    monkeypatch.setattr(check_gpu_env.metadata,'version',lambda name:versions[name])
    monkeypatch.setattr(check_gpu_env.metadata,'requires',lambda name:['transformers>=5','torch>=2'])
    report=check_gpu_env.vllm_dependency_status()
    assert report['required_by_lsl'] is False
    assert report['conflicts']==[dict(package='transformers',required='>=5',installed='4.56.2')]
    assert versions['transformers']=='4.56.2'


def test_json_receipt_parity_and_smoke_dry_runs(tmp_path):
    parity=subprocess.run([sys.executable,str(ROOT/'scripts/gpu_parity_test.py'),'--config','reproduction/configs/smoke_1p5b.json','--output',str(tmp_path/'parity.json'),'--dry-run'],cwd=ROOT,capture_output=True,text=True)
    assert parity.returncode==0,parity.stderr
    data=json.loads(parity.stdout)
    assert data['receipt'].endswith('parity.json') and data['output'].endswith('parity.artifacts')
    assert data['modes']==['off','basic','extended']
    assert not list(tmp_path.iterdir())
    bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path('/bin/bash')
    env={**os.environ,'PYTHON':sys.executable.replace('\\','/')}
    result=subprocess.run([str(bash),str(ROOT/'scripts/smoke_test.sh'),'1p5b','--dry-run'],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['effective_config']['model_name']=='Qwen/Qwen2.5-1.5B-Instruct'


def test_rocm_setup_cannot_install_or_replace_packages():
    bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path('/bin/bash')
    script=str(ROOT/'scripts/setup_gpu_env.sh')
    result=subprocess.run([str(bash),script,'--backend','rocm'],capture_output=True,text=True)
    assert result.returncode==2 and 'installation is intentionally disabled' in result.stderr
    result=subprocess.run([str(bash),script,'--backend','rocm','--check-existing','--dry-run'],capture_output=True,text=True)
    assert result.returncode==0 and 'no installation' in result.stdout


def test_pinned_revisions_and_algorithm_files_unchanged():
    old=json.loads(subprocess.check_output(['git','show','44b2119:reproduction/resources.lock.json'],cwd=ROOT,text=True,encoding='utf-8'))
    now=json.loads((ROOT/'reproduction/resources.lock.json').read_text(encoding='utf-8'))
    assert [(r['repo_id'],r['revision'],r['files']) for r in old['resources']]==[(r['repo_id'],r['revision'],r['files']) for r in now['resources']]
    assert all('\\' not in r['local_path'] for r in now['resources'])
    result=subprocess.check_output(['git','diff','44b2119','--','modeling_qwen2.py','cl_memory.py','diagnostics','reproduction/configs',':(exclude)reproduction/configs/smoke_1p5b.json'],cwd=ROOT)
    assert not result
