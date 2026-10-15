"""Read-only, machine-local validation of pinned Hub snapshots; never downloads."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime,timezone
from fnmatch import fnmatch
from .configuration import local_resource, safe_resource_file, resolve_cache_root

TOKENIZER_PATTERNS=['config.json','generation_config.json','tokenizer*','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json','chat_template*']
MODEL_PATTERNS=TOKENIZER_PATTERNS+['*.safetensors','*.safetensors.index.json']
MARKER='.lsl-resource.json'
EVIDENCE_PATH=Path(__file__).resolve().parent/'local_import_evidence.json'

def import_evidence(entry,weights=True):
    """Only repository-reviewed, revision-bound content evidence is authoritative."""
    if not EVIDENCE_PATH.is_file():raise ValueError('missing trusted local import evidence file')
    document=json.loads(EVIDENCE_PATH.read_text(encoding='utf-8'))
    if document.get('schema_version')!=1:raise ValueError('unsupported local import evidence schema')
    records=[r for r in document['resources'] if r['repo_id']==entry['repo_id'] and r['revision']==entry['revision']]
    if len(records)!=1:raise ValueError(f"missing trusted evidence for {entry['repo_id']} at {entry['revision']}")
    evidence={f['name']:f for f in records[0]['files']}
    required=selected_files(entry,weights)
    missing=[]
    for item in required:
        reference=evidence.get(item['name'],{})
        digest=reference.get('sha256','')
        if (not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)
                or reference.get('size')!=item.get('size')
                or item.get('sha256') and item['sha256']!=digest):missing.append(item['name'])
    if missing:raise ValueError('missing/mismatched trusted SHA256 evidence: '+', '.join(missing))
    return evidence

def verify_import_files(entry,location,weights=True):
    evidence=import_evidence(entry,weights)
    if not location.is_dir():raise ValueError(f'local model directory not found: {location}')
    # Extra recognized files can override tokenizer/config selection despite valid
    # required files (ModelScope's unrelated bookkeeping files remain harmless).
    allowed={f['name'] for f in selected_files(entry)}
    unexpected=[p.name for p in location.iterdir() if p.is_file() and any(fnmatch(p.name,pattern) for pattern in MODEL_PATTERNS) and p.name not in allowed]
    if unexpected:raise ValueError('unverified runtime-affecting files: '+', '.join(sorted(unexpected)))
    verified=[];failures=[]
    for item in selected_files(entry,weights):
        target=safe_resource_file(location,item['name']);expected=evidence[item['name']]
        if not target.is_file():failures.append(item['name']+' (missing)');continue
        if target.stat().st_size!=expected['size']:failures.append(item['name']+' (size mismatch)');continue
        with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=expected['sha256']:failures.append(item['name']+' (SHA256 mismatch)');continue
        verified.append(dict(name=item['name'],bytes=target.stat().st_size,sha256=digest))
    if failures:raise ValueError('local content verification failed: '+', '.join(failures))
    return verified

def import_local_resource(entry,cache_root=None,manifest=None,source_provider=None,source_revision=None):
    """Verify in place, then atomically attest content, without claiming Hub origin."""
    if entry['kind']!='model':raise ValueError('local import currently supports model resources only')
    if not source_provider or not source_provider.strip():raise ValueError('actual source provider must be declared')
    location=local_resource(entry,cache_root,manifest)
    verified=verify_import_files(entry,location)
    state=dict(schema_version=2,provenance_type='verified_local_import',repo_id=entry['repo_id'],revision=entry['revision'],
               source_provider=source_provider.strip(),source_revision=source_revision,
               source_revision_status='declared_unconfirmed' if source_revision else 'unknown',
               verified_against=dict(provider='huggingface',repo_id=entry['repo_id'],revision=entry['revision']),
               verification_method='per_file_sha256_against_official_revision_evidence',verified_files=verified,
               verified_at=datetime.now(timezone.utc).isoformat())
    marker=safe_resource_file(location,MARKER)
    # Reuse a previous identical attestation only after hashing actual files again.
    if marker.is_file():
        try:previous=json.loads(marker.read_text(encoding='utf-8'))
        except ValueError:previous={}
        if all(previous.get(k)==v for k,v in state.items() if k!='verified_at'):
            validate_resource(entry,cache_root,manifest);return previous
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=location,prefix='.lsl-resource.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(state,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,marker)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    validate_resource(entry,cache_root,manifest)
    return state

def required_datasets(plan):
    tasks={'igbo_translation':['Tommy0201/igbo_to_english_split'],
           'chemdata':['AI4Chem/ChemData700K','AI4Chem/ChemBench4K'],
           'cybersecurity_ift':['Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset','XuanwuAI/SecEval']}
    result={repo for task in plan['task_order'] for repo in tasks[task]}
    if plan['overrides']['activate_cl_method']:result.add('assafbk/lsl-negatives-fineweb-edu')
    if not plan['overrides']['skip_lm_eval']:
        benchmarks={'gsm8k':'openai/gsm8k','humaneval':'openai/openai_humaneval','ifeval':'google/IFEval'}
        result.update(benchmarks[task] for task in plan['overrides']['lm_eval_tasks'])
    return result

def selected_files(entry,weights=True):
    patterns=None if entry['kind']=='dataset' else MODEL_PATTERNS if weights else TOKENIZER_PATTERNS
    files=[f for f in entry['files'] if patterns is None or any(fnmatch(f['name'],p) for p in patterns)]
    if not files:raise ValueError(f"empty file inventory: {entry['repo_id']}")
    if entry['kind']=='model':
        names={f['name'] for f in files}
        if 'config.json' not in names or not any(n.startswith('tokenizer') for n in names):raise ValueError('model inventory lacks config/tokenizer')
        if weights and not any(n.endswith('.safetensors') for n in names):raise ValueError('model inventory lacks weights')
    return files

def validate_resource(entry,cache_root=None,manifest=None,weights=True,verify_hash=False):
    location=local_resource(entry,cache_root,manifest)
    files=selected_files(entry,weights)
    missing=[]
    for item in files:
        target=safe_resource_file(location,item['name'])
        if not target.is_file() or item.get('size') is not None and target.stat().st_size!=item['size']:
            missing.append(item['name']);continue
        if verify_hash and item.get('sha256'):
            with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=item['sha256']:missing.append(item['name']+' (sha256 mismatch)')
    marker=safe_resource_file(location,MARKER)
    state=json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else {}
    if state:
        revision_ok=state.get('repo_id')==entry['repo_id'] and state.get('revision')==entry['revision']
        if state.get('provenance_type')=='verified_local_import':
            if state.get('schema_version')!=2:raise ValueError('unsupported local import sidecar schema')
            expected=dict(provider='huggingface',repo_id=entry['repo_id'],revision=entry['revision'])
            if state.get('verified_against')!=expected:raise ValueError('local import target revision mismatch')
            if not state.get('source_provider') or state.get('verification_method')!='per_file_sha256_against_official_revision_evidence':
                raise ValueError('incomplete verified local import provenance')
            # Recheck all imported content even in ordinary Parity preflight.
            actual=verify_import_files(entry,location,weights)
            recorded={f['name']:f for f in state.get('verified_files',[])}
            if any(recorded.get(f['name'])!=f for f in actual):raise ValueError('local import sidecar file evidence mismatch')
    else:
        # Migrate old caches only when each required file has Hub revision metadata
        # or a matching pinned content hash; directory/status alone proves nothing.
        revision_ok=True
        for item in files:
            metadata=safe_resource_file(location,'.cache/huggingface/download/'+item['name']+'.metadata')
            lines=metadata.read_text(encoding='utf-8').splitlines() if metadata.is_file() else []
            if lines and lines[0]==entry['revision']:continue
            target=safe_resource_file(location,item['name'])
            if item.get('sha256') and target.is_file():
                with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if digest==item['sha256']:continue
            revision_ok=False;break
    if missing or not revision_ok:
        detail=', '.join(missing[:5]) if missing else 'revision provenance missing/mismatched'
        raise ValueError(f"{entry['kind'].capitalize()} not cached or incomplete: {entry['repo_id']}\nExpected revision: {entry['revision']}\nCache root: {resolve_cache_root(cache_root,manifest)}\nExpected path: {location}\nReason: {detail}\nRun scripts/prefetch_resources.py with --config <plan> --models --cache-root <root> first (add --datasets for training).")
    return location
