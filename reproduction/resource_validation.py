"""Read-only, machine-local validation of pinned Hub snapshots; never downloads."""
import hashlib
import json
from fnmatch import fnmatch
from .configuration import local_resource, safe_resource_file, resolve_cache_root

TOKENIZER_PATTERNS=['config.json','generation_config.json','tokenizer*','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json','chat_template*']
MODEL_PATTERNS=TOKENIZER_PATTERNS+['*.safetensors','*.safetensors.index.json']
MARKER='.lsl-resource.json'

def required_datasets(plan):
    tasks={'igbo_translation':['Tommy0201/igbo_to_english_split'],
           'chemdata':['AI4Chem/ChemData700K','AI4Chem/ChemBench4K'],
           'cybersecurity_ift':['Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset','XuanwuAI/SecEval']}
    result={repo for task in plan['task_order'] for repo in tasks[task]}
    if plan['overrides']['activate_cl_method']:result.add('assafbk/lsl-negatives-fineweb-edu')
    if not plan['overrides']['skip_lm_eval']:result.update(['openai/gsm8k','openai/openai_humaneval','google/IFEval'])
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
