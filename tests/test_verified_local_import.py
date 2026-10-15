"""Offline local import: actual file hashes, atomic writes and shared validation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reproduction import resource_validation as rv
from reproduction.configuration import ROOT,local_resource,resources


@pytest.fixture
def local_model(tmp_path,monkeypatch):
    names=['config.json','generation_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors']
    blobs={name:('synthetic '+name).encode() for name in names}
    references=[dict(name=n,size=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in blobs.items()]
    entry=dict(kind='model',repo_id='Qwen/synthetic',revision='a'*40,local_path='.cache\\lsl/model\\Qwen--synthetic',
               status='not_cached',files=[dict(name=f['name'],size=f['size'],sha256=f['sha256'] if f['name'].endswith('.safetensors') else None) for f in references])
    evidence=tmp_path/'trusted_evidence.json'
    evidence.write_text(json.dumps(dict(schema_version=1,resources=[dict(repo_id=entry['repo_id'],revision=entry['revision'],files=references)])))
    monkeypatch.setattr(rv,'EVIDENCE_PATH',evidence)
    cache=tmp_path/'cache';location=local_resource(entry,cache);location.mkdir(parents=True)
    for name,blob in blobs.items():(location/name).write_bytes(blob)
    return entry,cache,location,evidence


def import_model(fixture,**kwargs):
    entry,cache,_,_=fixture
    return rv.import_local_resource(entry,cache,source_provider='modelscope',**kwargs)


def test_import_all_files_and_real_provenance(local_model):
    entry,cache,location,_=local_model
    state=import_model(local_model)
    assert rv.validate_resource(entry,cache)==location
    assert state['source_provider']=='modelscope' and state['source_revision'] is None
    assert state['source_revision_status']=='unknown'
    assert state['verified_against']==dict(provider='huggingface',repo_id=entry['repo_id'],revision='a'*40)
    assert state['verified_at'] and len(state['verified_files'])==7
    assert all(f['sha256']==hashlib.sha256((location/f['name']).read_bytes()).hexdigest() for f in state['verified_files'])


@pytest.mark.parametrize('name',['model.safetensors','tokenizer.json','config.json'])
def test_correct_size_wrong_hash_rejected(local_model,name):
    _,_,location,_=local_model
    target=location/name;blob=target.read_bytes();target.write_bytes(b'X'+blob[1:])
    with pytest.raises(ValueError,match='SHA256 mismatch'):import_model(local_model)
    assert not (location/rv.MARKER).exists()


@pytest.mark.parametrize('name',['model.safetensors','tokenizer_config.json','merges.txt'])
def test_missing_required_file_rejected(local_model,name):
    location=local_model[2];(location/name).unlink()
    with pytest.raises(ValueError,match='missing'):import_model(local_model)
    assert not (location/rv.MARKER).exists()


def test_source_revision_never_becomes_hf_revision(local_model):
    state=import_model(local_model,source_revision='master')
    assert state['source_revision']=='master' and state['source_revision_status']=='declared_unconfirmed'
    assert state['revision']=='a'*40 and state['verified_against']['revision']=='a'*40


def test_target_revision_change_invalidates_sidecar(local_model):
    entry,cache,_,_=local_model;import_model(local_model)
    entry=copy.deepcopy(entry);entry['revision']='b'*40
    with pytest.raises(ValueError,match='revision'):rv.validate_resource(entry,cache)


def test_atomic_write_interruption_creates_no_attestation(local_model,monkeypatch):
    location=local_model[2]
    def fail(*args):raise OSError('synthetic write interruption')
    monkeypatch.setattr(rv.os,'replace',fail)
    with pytest.raises(OSError,match='interruption'):import_model(local_model)
    assert not (location/rv.MARKER).exists() and not list(location.glob('.lsl-resource.*.tmp'))
    with pytest.raises(ValueError):rv.validate_resource(local_model[0],local_model[1])


def test_atomic_failure_preserves_previous_attestation(local_model,monkeypatch):
    location=local_model[2];import_model(local_model);before=(location/rv.MARKER).read_bytes()
    def fail(*args):raise OSError('interruption')
    monkeypatch.setattr(rv.os,'replace',fail)
    with pytest.raises(OSError):import_model(local_model,source_revision='declared-other')
    assert (location/rv.MARKER).read_bytes()==before
    rv.validate_resource(local_model[0],local_model[1])


@pytest.mark.parametrize('legacy',['.cache/lsl/model/Qwen--synthetic','.cache\\lsl\\model/Qwen--synthetic'])
def test_import_path_compatibility(local_model,legacy):
    local_model[0]['local_path']=legacy
    import_model(local_model)
    assert rv.validate_resource(local_model[0],local_model[1])==local_model[2]


def test_offline_import_never_uses_network(local_model,monkeypatch):
    import urllib.request
    import requests
    import huggingface_hub
    def forbidden(*args,**kwargs):raise AssertionError('network forbidden')
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    monkeypatch.setattr(requests.sessions.Session,'request',forbidden)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',forbidden)
    import_model(local_model)
    rv.validate_resource(local_model[0],local_model[1])


def test_repeated_import_reuses_verified_attestation(local_model):
    state=import_model(local_model);marker=local_model[2]/rv.MARKER;before=marker.read_bytes()
    assert import_model(local_model)==state and marker.read_bytes()==before


def test_model_cache_isolation(local_model):
    entry,cache,_,_=local_model;import_model(local_model)
    other=copy.deepcopy(entry);other['repo_id']='Qwen/other';other['local_path']='.cache/lsl/model/Qwen--other'
    with pytest.raises(ValueError):rv.validate_resource(other,cache)
    with pytest.raises(ValueError):rv.validate_resource(entry,cache/'different-experiment-cache')


def test_imported_content_is_rechecked_by_parity_validator(local_model):
    from scripts import gpu_parity_test
    entry,cache,location,_=local_model;import_model(local_model)
    assert gpu_parity_test.validate_resource(entry,cache)==location
    target=location/'tokenizer.json';blob=target.read_bytes();target.write_bytes(b'!'+blob[1:])
    with pytest.raises(ValueError,match='SHA256 mismatch'):gpu_parity_test.validate_resource(entry,cache)


@pytest.mark.parametrize('failure',['missing_hash','wrong_size','wrong_pinned_hash','missing_file_evidence'])
def test_insufficient_official_evidence_rejected(local_model,failure):
    entry,_,location,path=local_model
    doc=json.loads(path.read_text());files=doc['resources'][0]['files']
    if failure=='missing_hash':files[0]['sha256']=None
    if failure=='wrong_size':files[0]['size']+=1
    if failure=='wrong_pinned_hash':entry['files'][-1]['sha256']='0'*64
    if failure=='missing_file_evidence':files.pop(0)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='evidence'):import_model(local_model)
    assert not (location/rv.MARKER).exists()


def test_failed_reimport_does_not_replace_previous_sidecar(local_model):
    entry,cache,location,_=local_model;import_model(local_model)
    before=(location/rv.MARKER).read_bytes();(location/'model.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError):import_model(local_model)
    assert (location/rv.MARKER).read_bytes()==before
    with pytest.raises(ValueError):rv.validate_resource(entry,cache)


def test_extra_tokenizer_override_is_rejected(local_model):
    (local_model[2]/'special_tokens_map.json').write_text('{}')
    with pytest.raises(ValueError,match='runtime-affecting'):import_model(local_model)
    assert not (local_model[2]/rv.MARKER).exists()


def test_modelscope_bookkeeping_files_are_not_forged_hf_evidence(local_model):
    (local_model[2]/'configuration.json').write_text('{"source":"ModelScope"}')
    state=import_model(local_model)
    assert 'configuration.json' not in {f['name'] for f in state['verified_files']}


def test_imported_model_supports_tokenizer_only_preflight(local_model):
    import_model(local_model)
    assert rv.validate_resource(local_model[0],local_model[1],weights=False)==local_model[2]


def test_committed_evidence_covers_real_pinned_model():
    _,entries=resources(ROOT/'reproduction/resources.lock.json')
    entry=entries['Qwen/Qwen2.5-1.5B-Instruct']
    assert len(rv.import_evidence(entry))==7
    data=json.loads((ROOT/'reproduction/local_import_evidence.json').read_text())['resources'][0]
    assert data['repo_id']==entry['repo_id'] and data['revision']==entry['revision']
    assert {f['name'] for f in data['files']}=={f['name'] for f in rv.selected_files(entry)}
    weight=next(f for f in data['files'] if f['name']=='model.safetensors')
    assert weight['size']==3087467144 and weight['sha256']=='dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
    assert all(entry['revision'] in f['source_url'] and len(f['sha256'])==64 for f in data['files'])


def test_cli_import_works_without_linux_or_network(local_model):
    entry,cache,location,evidence=local_model
    # Execute actual CLI and validator using a synthetic trusted evidence fixture.
    manifest=cache.parent/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,resources=[entry],cache_root='.cache/lsl')))
    plan=json.loads((ROOT/'reproduction/configs/smoke_1p5b.json').read_text());plan['overrides']['model_name']=entry['repo_id']
    plan_path=cache.parent/'plan.json';plan_path.write_text(json.dumps(plan))
    code='import sys; from pathlib import Path; import reproduction.resource_validation as r; r.EVIDENCE_PATH=Path(sys.argv.pop(1)); from scripts.prefetch_resources import main; main()'
    command=[sys.executable,'-c',code,str(evidence),'--manifest',str(manifest),'--config',str(plan_path),'--models','--cache-root',str(cache),'--import-local','--source','modelscope','--verify-sha256']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['source_provider']=='modelscope'
    assert rv.validate_resource(entry,cache)==location
    check=[sys.executable,'-c',code,str(evidence),'--manifest',str(manifest),'--config',str(plan_path),'--models','--cache-root',str(cache),'--check-only','--verify-sha256']
    result=subprocess.run(check,cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
