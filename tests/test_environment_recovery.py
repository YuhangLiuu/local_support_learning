"""Synthetic wheels/filesystem/subprocess boundaries; never installs on Windows."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

from reproduction import environment_runtime as runtime
from reproduction import environment_bootstrap as bootstrap


@pytest.fixture
def house(tmp_path,monkeypatch):
    catalog=tmp_path/'profiles.json'
    pins={'transformers':'4.56.2','tokenizers':'0.22.2','huggingface-hub':'0.36.2'}
    catalog.write_text(json.dumps(dict(profiles={p:dict(pins=pins,roots=list(pins)) for p in ('minimal','full')})))
    monkeypatch.setattr(runtime,'CATALOG',catalog);monkeypatch.setattr(bootstrap,'CATALOG',catalog)
    identity=dict(os='Linux',python='3.11',implementation='cpython',soabi='cpython-311',machine='x86_64',libc=['glibc','2.35'],image_id='test-image',
        gpu=dict(backend='cuda',gpu='synthetic A10',cuda='12.8'),gpu_versions={'torch':'2.9.0','flash-attn':'2.8.3'})
    state=dict(identity=identity,in_venv=False,executable=sys.executable,executable_sha256=runtime.sha(sys.executable),instance_id='boot-1',
        gpu_packages={'torch':dict(version='2.9.0'),'flash-attn':dict(version='2.8.3')},gpu_dependencies={})
    directory=tmp_path/'wheelhouse';(directory/'wheels').mkdir(parents=True);records=[]
    for name,version in pins.items():
        filename=f'{name.replace("-","_")}-{version}-py3-none-any.whl';path=directory/'wheels'/filename
        requires='Requires-Dist: huggingface-hub<1.0,>=0.34\n' if name=='transformers' else ''
        with zipfile.ZipFile(path,'w') as archive:archive.writestr(f'{name.replace("-","_")}-{version}.dist-info/METADATA',f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{requires}')
        records.append(dict(file='wheels/'+filename,bytes=path.stat().st_size,sha256=runtime.sha(path),name=name,version=version))
    artifact_catalog=tmp_path/'artifacts.json'
    artifact_catalog.write_text(json.dumps(dict(packages={r['name']:dict(version=r['version'],artifacts=[dict(filename=Path(r['file']).name,sha256=r['sha256'],size=r['bytes'],packagetype='bdist_wheel')]) for r in records})))
    monkeypatch.setattr(runtime,'ARTIFACTS',artifact_catalog);monkeypatch.setattr(bootstrap,'ARTIFACTS',artifact_catalog)
    manifest=dict(schema_version=1,profile='minimal',identity=identity,catalog_sha256=runtime.catalog_digest(),wheels=records,nltk_files=[])
    runtime.atomic_json(directory/'manifest.json',manifest)
    return directory,state,manifest


def test_actual_wheel_zip_hash_and_dependency_closure(house):
    directory,state,manifest=house
    assert runtime.verify_wheelhouse(directory,state,'minimal')==manifest


@pytest.mark.parametrize('field,value',[('python','3.12'),('machine','aarch64'),('os','Windows'),('image_id','other-image'),('gpu',dict(backend='rocm',hip='7.2',gpu='MI300'))])
def test_wheelhouse_incompatible_stack_rejected(house,field,value):
    directory,state,_=house;state=copy.deepcopy(state);state['identity'][field]=value
    with pytest.raises(ValueError,match='mismatch'):runtime.verify_wheelhouse(directory,state,'minimal')


def test_corrupt_wheel_and_unapproved_profile_rejected(house):
    directory,state,manifest=house
    with pytest.raises(ValueError,match='mismatch'):runtime.verify_wheelhouse(directory,state,'full')
    wheel=directory/manifest['wheels'][0]['file'];wheel.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='integrity'):runtime.verify_wheelhouse(directory,state,'minimal')


def test_changing_manifest_hash_cannot_forge_official_wheel(house):
    directory,state,manifest=house;item=manifest['wheels'][0];wheel=directory/item['file']
    with wheel.open('ab') as stream:stream.write(b'altered trailing payload')
    item.update(bytes=wheel.stat().st_size,sha256=runtime.sha(wheel))
    runtime.atomic_json(directory/'manifest.json',manifest)
    with pytest.raises(ValueError,match='provenance'):runtime.verify_wheelhouse(directory,state,'minimal')


def test_prepare_rejects_manually_tampered_download_plan(house,monkeypatch):
    directory,state,_=house
    expected=dict(identity=state['identity'],catalog_sha256=runtime.catalog_digest(),profile='minimal',total_download_bytes=0)
    monkeypatch.setattr(bootstrap,'create_plan',lambda *a,**kw:expected)
    modified=expected|dict(total_download_bytes=1)
    with pytest.raises(ValueError,match='official artifact'):bootstrap.prepare(directory,state,modified,8)


@pytest.mark.parametrize('name',['minimal','full'])
def test_real_prepare_pipeline_with_synthetic_download_bytes(house,monkeypatch,name):
    import io
    directory,state,manifest=house;payloads={};artifacts=[]
    for item in manifest['wheels']:
        path=directory/item['file'];url='https://files.pythonhosted.org/'+path.name;payloads[url]=path.read_bytes();path.unlink()
        artifacts.append(dict(package=item['name'],version=item['version'],file=path.name,url=url,bytes=item['bytes'],sha256=item['sha256'],source_build=False))
    (directory/'manifest.json').unlink();nltk=[]
    if name=='full':
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as archive:archive.writestr('punkt_tab/english/collocations.tab',b'synthetic NLTK resource')
        raw=buffer.getvalue();url='https://raw.githubusercontent.com/nltk/nltk_data/synthetic/packages/tokenizers/punkt_tab.zip'
        payloads[url]=raw
        import hashlib
        nltk=[dict(file='punkt_tab.zip',url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())]
    plan=dict(identity=state['identity'],catalog_sha256=runtime.catalog_digest(),profile=name,artifacts=artifacts,nltk=nltk,total_download_bytes=sum(a['bytes'] for a in artifacts+nltk))
    monkeypatch.setattr(bootstrap,'create_plan',lambda *a,**kw:plan)
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',lambda url,**kwargs:io.BytesIO(payloads[url]))
    monkeypatch.setattr(bootstrap.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*2**30))
    prepared=bootstrap.prepare(directory,state,plan,8)
    assert prepared==runtime.verify_wheelhouse(directory,state,name)
    assert len(prepared['wheels'])==3 and bool(prepared['nltk_files'])==(name=='full')


def test_wrong_hub_and_tokenizer_versions_rejected():
    with pytest.raises(ValueError,match='incompatible'):
        runtime.check_dependency_closure({'transformers':dict(version='4.56.2',requires=['huggingface-hub<1']), 'huggingface-hub':dict(version='2.2.0')},{})
    with pytest.raises(ValueError,match='incompatible'):
        runtime.check_dependency_closure({'transformers':dict(version='4.56.2',requires=['tokenizers>=0.22,<0.23']), 'tokenizers':dict(version='0.21.0')},{})


def test_unlocked_transitive_dependency_rejected():
    with pytest.raises(ValueError,match='unlocked'):runtime.check_dependency_closure({'application':dict(version='1',requires=['unlocked>=1'])},{})


@pytest.mark.parametrize('backend',['cuda','rocm'])
def test_cuda_rocm_base_policy_without_nvidia_tools(house,backend):
    _,state,_=house;state=copy.deepcopy(state);state['identity']['gpu']['backend']=backend
    runtime.check_base(state)


def test_broken_interpreter_old_venv_and_no_gpu_rejected(house,tmp_path):
    _,state,_=house;state=copy.deepcopy(state);state['executable']=str(tmp_path/'broken-link')
    with pytest.raises(ValueError,match='missing/broken'):runtime.check_base(state)
    state['executable']=sys.executable;state['in_venv']=True
    with pytest.raises(ValueError,match='old/active'):runtime.check_base(state)
    state['in_venv']=False;state['identity']['gpu']=dict(backend='cpu')
    with pytest.raises(ValueError,match='unavailable'):runtime.check_base(state)


def test_real_abi_tag_mismatch_rejected(house):
    directory,state,manifest=house
    path=directory/manifest['wheels'][0]['file'];wrong=path.with_name('transformers-4.56.2-cp310-cp310-manylinux_2_17_x86_64.whl');path.rename(wrong)
    manifest['wheels'][0]['file']='wheels/'+wrong.name;runtime.atomic_json(directory/'manifest.json',manifest)
    with pytest.raises(ValueError,match='ABI'):runtime.verify_wheelhouse(directory,state,'minimal')


def test_system_transformers_path_leak_rejected(house,monkeypatch,tmp_path):
    monkeypatch.setattr(runtime.md,'distribution',lambda name:SimpleNamespace(version='4.56.2',locate_file=lambda _:tmp_path/'system'))
    with pytest.raises(ValueError,match='leaked'):runtime.check_application_paths('minimal',tmp_path/'venv')
    monkeypatch.setattr(runtime.md,'distribution',lambda name:SimpleNamespace(version='5.14.0',locate_file=lambda _:tmp_path/'venv'))
    with pytest.raises(ValueError,match='4.56.2'):runtime.check_application_paths('minimal',tmp_path/'venv')


def simulated_runner(events,fail_install=False,fail_verify=False):
    def run(command,env=None):
        command=list(map(str,command));events.append(command)
        if 'venv' in command:
            target=Path(command[-1]);(target/'bin').mkdir(parents=True);(target/'bin/python').write_text('synthetic executable')
        elif 'pip' in command:
            assert '--no-index' in command and '--no-deps' in command and '--require-hashes' in command and '--ignore-installed' in command
            assert env['PIP_NO_INDEX']=='1' and 'PIP_INDEX_URL' not in env
            if fail_install:raise RuntimeError('synthetic partial install failure')
        else:
            if fail_verify:raise RuntimeError('synthetic real-kernel check failure')
            output=Path(command[command.index('--output')+1]);runtime.atomic_json(output,dict(passed=True,profile=command[command.index('--profile')+1]))
        return 'simulated verification subprocess PASS'
    return run


@pytest.mark.parametrize('backend',['cuda','rocm'])
def test_offline_install_reuse_and_old_instance_isolation(house,tmp_path,monkeypatch,backend):
    directory,state,manifest=house;state['identity']['gpu']['backend']=backend
    runtime.atomic_json(directory/'manifest.json',manifest)
    events=[];monkeypatch.setattr(bootstrap,'run',simulated_runner(events))
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',lambda *a,**kw:pytest.fail('offline install attempted network'))
    root=tmp_path/'persistent'
    first,report=bootstrap.install(root,directory,state,'minimal',tmp_path/'cache')
    assert (root/'current-minimal.json').is_file() and report.is_file()
    assert not (first/'installation_failed.json').exists()
    count=len(events);second,_=bootstrap.install(root,directory,state,'minimal',tmp_path/'cache')
    assert first==second and len(events)==count+1  # verify, no reinstall
    newer=copy.deepcopy(state);newer['instance_id']='boot-2'
    third,third_report=bootstrap.install(root,directory,newer,'minimal',tmp_path/'cache')
    assert third!=first and first.is_dir()
    monkeypatch.setattr(bootstrap,'run',simulated_runner(events,fail_verify=True))
    with pytest.raises(RuntimeError):bootstrap.install(root,directory,newer,'minimal',tmp_path/'cache')
    assert json.loads(third_report.read_text())['passed'] is False


@pytest.mark.parametrize('failure',['install','verify'])
def test_partial_install_or_failed_acceptance_never_publishes_ready(house,tmp_path,monkeypatch,failure):
    directory,state,_=house;root=tmp_path/'persistent';events=[]
    monkeypatch.setattr(bootstrap,'run',simulated_runner(events,fail_install=failure=='install',fail_verify=failure=='verify'))
    with pytest.raises(RuntimeError):bootstrap.install(root,directory,state,'minimal',tmp_path/'cache')
    assert not (root/'current-minimal.json').exists()
    assert list((root/'envs').glob('*/installation_failed.json'))


def test_broken_old_environment_is_preserved_and_rebuilt(house,tmp_path,monkeypatch):
    directory,state,_=house;events=[];root=tmp_path/'persistent'
    monkeypatch.setattr(bootstrap,'run',simulated_runner(events))
    first,_=bootstrap.install(root,directory,state,'minimal',tmp_path/'cache')
    (first/'bin/python').unlink()
    second,_=bootstrap.install(root,directory,state,'minimal',tmp_path/'cache')
    assert second!=first and first.exists() and '-rebuild-' in second.name
    count=len(events)
    third,_=bootstrap.install(root,directory,state,'minimal',tmp_path/'cache')
    assert third==second and len(events)==count+1


def test_no_public_network_fallback_and_bad_download_hash(tmp_path,monkeypatch):
    import io
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(b'wrong'))
    item=dict(file='x.whl',url='https://files.pythonhosted.org/x.whl',bytes=5,sha256='0'*64)
    with pytest.raises(ValueError,match='hash'):bootstrap.download_artifact(item,tmp_path)
    assert not (tmp_path/'x.whl').exists() and not (tmp_path/'x.whl.partial').exists()
    item['url']='http://untrusted.example/x.whl'
    with pytest.raises(ValueError,match='source'):bootstrap.download_artifact(item,tmp_path)
    def offline(*args,**kwargs):raise OSError('network unavailable')
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',offline)
    with pytest.raises(OSError,match='unavailable'):bootstrap.read_url('https://pypi.org/pypi/x/json')


def test_inherited_pip_targets_cannot_redirect_install_into_system(monkeypatch):
    for name in ('PIP_TARGET','PIP_PREFIX','PIP_USER','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_PYTHON'):
        monkeypatch.setenv(name,'untrusted-system-location')
    env=bootstrap.offline_env()
    assert env['PIP_NO_INDEX']=='1'
    assert not any(name in env for name in ('PIP_TARGET','PIP_PREFIX','PIP_USER','PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_PYTHON'))


def test_profile_locks_are_explicit_and_gpu_packages_excluded():
    for name in ('minimal','full'):
        pins=runtime.profile(name)['pins']
        assert pins['transformers']=='4.56.2' and pins['tokenizers']=='0.22.2' and pins['huggingface-hub']=='0.36.2' and pins['pomegranate']=='1.1.2'
        assert not any(runtime.protected(p) for p in pins)
    assert 'lm-eval' not in runtime.profile('minimal')['pins']
    assert runtime.profile('full')['pins']['lm-eval']=='0.4.10'


@pytest.mark.parametrize('name',['minimal','full'])
def test_committed_artifact_metadata_closes_all_python_targets(name):
    catalog=json.loads(runtime.ARTIFACTS.read_text());pins=runtime.profile(name)['pins']
    packages={p:dict(version=v,requires=catalog['packages'][p]['requires_dist']) for p,v in pins.items()}
    for p,v in pins.items():
        assert catalog['packages'][p]['version']==v
        assert catalog['packages'][p]['artifacts'] and all(len(a['sha256'])==64 for a in catalog['packages'][p]['artifacts'])
    for minor in ['3.11','3.12']:
        runtime.check_dependency_closure(packages,{'torch':'2.9.0'},dict(python_version=minor,python_full_version=minor+'.0',os_name='posix',sys_platform='linux',platform_system='Linux',platform_machine='x86_64'))
    if name=='full':assert all(len(a['sha256'])==64 and a['revision'] in a['url'] for a in catalog['nltk'])


@pytest.mark.parametrize('name',['minimal','full'])
@pytest.mark.parametrize('minor',[11,12])
def test_actual_linux_download_plan_from_metadata_only(name,minor,monkeypatch):
    tags,_,_,_,_=runtime.packaging_api()
    platforms=[f'manylinux_2_{n}_x86_64' for n in range(40,4,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']
    abi=f'cp3{minor}'
    supported=list(tags.cpython_tags((3,minor),abis=[abi],platforms=platforms))+list(tags.compatible_tags((3,minor),interpreter=abi,platforms=platforms))
    monkeypatch.setattr(tags,'sys_tags',lambda:iter(supported))
    monkeypatch.setattr(bootstrap.urllib.request,'urlopen',lambda *args,**kw:pytest.fail('planning attempted network'))
    monkeypatch.setattr(bootstrap,'sys',SimpleNamespace(version_info=(3,minor,0)))
    state=dict(identity=dict(python=f'3.{minor}',machine='x86_64',gpu_versions={'torch':'2.9.0'}),gpu_dependencies={})
    plan=bootstrap.create_plan(state,name,allow_build=True)
    assert len(plan['artifacts'])==len(runtime.profile(name)['pins'])
    assert all(not runtime.protected(a['package']) for a in plan['artifacts'])
    assert all(a['sha256'] and a['bytes']>0 for a in plan['artifacts'])
    assert bool(plan['nltk'])==(name=='full')


def test_lightweight_launcher_versions_reject_pollution_and_allow_missing_optional_lm(monkeypatch):
    versions={'transformers':'4.56.2','pomegranate':'1.1.2','tokenizers':'0.22.2','huggingface-hub':'0.36.2'}
    def get(name):
        if name not in versions:raise runtime.md.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(runtime.md,'version',get)
    runtime.check_training_compatibility(True)
    with pytest.raises(ValueError,match='lm-eval==0.4.10'):runtime.check_training_compatibility(False)
    versions['transformers']='5.14.0'
    with pytest.raises(ValueError,match='4.56.2'):runtime.check_training_compatibility(True)


def test_bootstrap_dry_run_never_installs(tmp_path):
    script=runtime.ROOT/'scripts/bootstrap_environment.py'
    result=subprocess.run([sys.executable,str(script),'--install','--profile','full','--root',str(tmp_path/'absent'),'--dry-run'],capture_output=True,text=True)
    assert result.returncode==0 and json.loads(result.stdout)['install']
    assert not (tmp_path/'absent').exists()


def test_strict_report_blocks_failed_or_wrong_profile(tmp_path):
    from scripts.check_environment import verify_report,acceptance_hash
    path=tmp_path/'report.json';path.write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError,match='failed/stale'):verify_report(path)
    path.write_text(json.dumps(dict(passed=True,acceptance_hash=acceptance_hash(),profile='minimal')))
    with pytest.raises(ValueError,match='Full'):verify_report(path,requires_full=True)


def test_managed_minimal_rejects_formal_even_if_system_lm_eval_exists(tmp_path,monkeypatch):
    (tmp_path/'bootstrap.json').write_text(json.dumps(dict(profile='minimal')))
    monkeypatch.setattr(runtime,'sys',SimpleNamespace(prefix=str(tmp_path)))
    with pytest.raises(ValueError,match='Minimal'):runtime.check_training_compatibility(False)
    from reproduction import lm_evaluation
    monkeypatch.setattr(lm_evaluation,'sys',SimpleNamespace(prefix=str(tmp_path)))
    with pytest.raises(RuntimeError,match='Full'):lm_evaluation.require_lm_eval()


def test_real_tiny_lsl_environment_self_test_cpu(tmp_path):
    from scripts.check_environment import tiny_lsl_check
    assert tiny_lsl_check(tmp_path/'diagnostics').startswith('PASS')
    assert list((tmp_path/'diagnostics').rglob('*.jsonl'))


def test_instance_identity_does_not_depend_on_host_boot_id_alone(monkeypatch):
    first=runtime.probe('synthetic-image')
    monkeypatch.setattr(runtime.platform,'node',lambda:'different-cloud-container')
    second=runtime.probe('synthetic-image')
    assert first['identity']==second['identity']  # wheels remain compatible
    assert first['instance_id']!=second['instance_id']  # venv must not be reused
