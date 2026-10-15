"""Inventory/pin official Hub resources, then download within a checked disk budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import platform

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproduction.configuration import resolve_cache_root,local_resource,read_plan,resolve,safe_resource_file
from reproduction.resource_validation import TOKENIZER_PATTERNS,MODEL_PATTERNS,MARKER,selected_files,validate_resource,required_datasets,import_local_resource
SOURCES=[
    ("model","Qwen/Qwen2.5-7B-Instruct"),("model","Qwen/Qwen2.5-1.5B-Instruct"),
    ("dataset","Tommy0201/igbo_to_english_split"),("dataset","AI4Chem/ChemData700K"),
    ("dataset","AI4Chem/ChemBench4K"),("dataset","Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"),
    ("dataset","XuanwuAI/SecEval"),("dataset","assafbk/lsl-negatives-fineweb-edu"),
    ("dataset","openai/gsm8k"),("dataset","openai/openai_humaneval"),("dataset","google/IFEval")]

def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save(path,manifest):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

def inventory(path,cache):
    from huggingface_hub import HfApi
    api=HfApi();resources=[]
    for kind,repo in SOURCES:
        entry=dict(kind=kind,repo_id=repo,status="pending",local_path=(Path('.cache/lsl')/kind/repo.replace('/','--')).as_posix())
        try:
            info=api.repo_info(repo_id=repo,repo_type=kind,files_metadata=True,timeout=30)
            entry.update(revision=info.sha,status="inventoried",files=[dict(name=f.rfilename,size=f.size,
                sha256=getattr(f.lfs,"sha256",None) if f.lfs else None) for f in info.siblings],
                gated=info.gated,total_bytes=sum(f.size or 0 for f in info.siblings))
            print(f"{kind} {repo}: {info.sha} {entry['total_bytes']/2**20:.1f} MiB",flush=True)
        except Exception as error:
            entry.update(status="inaccessible",error_type=type(error).__name__)
            print(f"{repo}: inaccessible ({type(error).__name__})",flush=True)
        resources.append(entry)
        save(path,dict(schema_version=1,observed_at=datetime.now(timezone.utc).isoformat(),cache_root='.cache/lsl',resources=resources))
    return load_manifest(path)

def download(manifest,path,models=False,tokenizers=False,datasets=False,max_gib=40,verify_hash=False,cache_root=None,repo_ids=None):
    from fnmatch import fnmatch
    from huggingface_hub import snapshot_download
    planned=[]
    for entry in manifest['resources']:
        if repo_ids is not None and entry['repo_id'] not in repo_ids:continue
        patterns=None
        if entry['kind']=='dataset' and not datasets:continue
        if entry['kind']=='model':
            if models:patterns=TOKENIZER_PATTERNS+["*.safetensors","*.safetensors.index.json"]
            elif tokenizers:patterns=TOKENIZER_PATTERNS
            else:continue
        files=selected_files(entry,weights=models)
        location=local_resource(entry,cache_root,manifest)
        missing=sum(f['size'] or 0 for f in files if not safe_resource_file(location,f['name']).is_file() or safe_resource_file(location,f['name']).stat().st_size!=f['size'])
        planned.append((entry,patterns,files,missing,location))
    needed=sum(item[3] for item in planned)
    cache=resolve_cache_root(cache_root,manifest);cache.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(cache).free
    if needed>max_gib*2**30 or needed+20*2**30>free:
        raise RuntimeError(f"download budget refused: estimated {needed/2**30:.2f} GiB; free {free/2**30:.2f} GiB; reserve 20 GiB")
    print(f"download estimate {needed/2**30:.2f} GiB; free {free/2**30:.2f} GiB",flush=True)
    failed=False
    for entry,patterns,files,_,location in planned:
        try:
            try:
                validate_resource(entry,cache_root,manifest,weights=models,verify_hash=verify_hash)
                print(f"reuse validated snapshot: {entry['repo_id']}",flush=True)
                continue
            except ValueError:pass
            safe_resource_file(location,'.cache/huggingface/download')
            marker=safe_resource_file(location,MARKER)
            snapshot_download(entry['repo_id'],repo_type=entry['kind'],revision=entry['revision'],
                local_dir=str(location),allow_patterns=patterns,max_workers=2,force_download=verify_hash)
            checked=[]
            for f in files:
                target=safe_resource_file(location,f['name'])
                if not target.is_file() or (f['size'] is not None and target.stat().st_size!=f['size']):
                    raise ValueError(f"size mismatch: {f['name']}")
                digest=None
                if verify_hash and f['sha256']:
                    with target.open('rb') as stream:
                        digest=hashlib.file_digest(stream,'sha256').hexdigest()
                    if digest!=f['sha256']:raise ValueError(f"checksum mismatch: {f['name']}")
                checked.append(dict(name=f['name'],bytes=target.stat().st_size,sha256=digest))
            marker.write_text(json.dumps(dict(repo_id=entry['repo_id'],revision=entry['revision'],verified_files=checked),indent=2),encoding='utf-8')
            validate_resource(entry,cache_root,manifest,weights=models,verify_hash=verify_hash)
        except Exception as error:
            failed=True
            print(f"{entry['repo_id']}: download failed ({type(error).__name__}): {error}",flush=True)
    if failed:raise RuntimeError('resource download/verification failed; safe to rerun the same command')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'reproduction/resources.lock.json')
    p.add_argument('--cache-root');p.add_argument('--config',help='select only this plan model (datasets require --datasets)')
    p.add_argument('--check-only',action='store_true',help='validate selected resources without downloading')
    p.add_argument('--import-local',action='store_true',help='verify existing model content offline and atomically record its actual provenance')
    p.add_argument('--source',help='actual provider of existing files, e.g. modelscope')
    p.add_argument('--source-revision',help='optional user-declared source version (not treated as the HF revision)')
    p.add_argument('--inventory',action='store_true');p.add_argument('--models',action='store_true')
    p.add_argument('--tokenizers',action='store_true');p.add_argument('--datasets',action='store_true')
    p.add_argument('--max-gib',type=float,default=40);p.add_argument('--verify-sha256',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--relocate',action='store_true',help='change cache paths without changing pinned revisions')
    args=p.parse_args()
    manifest=load_manifest(args.manifest) if args.manifest.exists() else None
    if manifest is None:p.error('pinned manifest missing; refusing automatic repinning')
    plan=read_plan(args.config) if args.config else None
    selected_model=resolve(plan).model_name if plan else None
    dataset_ids=required_datasets(plan) if plan else set()
    chosen=[r for r in manifest['resources'] if (r['kind']=='model' and (args.models or args.tokenizers) and (selected_model is None or r['repo_id']==selected_model)) or (r['kind']=='dataset' and args.datasets and r['repo_id'] in dataset_ids)]
    if not args.config and not args.inventory and (args.models or args.tokenizers or args.datasets):p.error('--config is required to explicitly select experiment resources')
    cache=resolve_cache_root(args.cache_root,manifest)
    if args.import_local and (not args.config or not args.models or args.datasets or args.tokenizers or args.check_only or args.inventory or not args.source or not args.verify_sha256):
        p.error('--import-local requires --config --models --source --verify-sha256; cannot combine with datasets/tokenizers/check-only/inventory')
    if args.dry_run:
        print(json.dumps(dict(resources=[dict(repo_id=r['repo_id'],revision=r['revision'],local_path=str(local_resource(r,args.cache_root,manifest))) for r in chosen],cache_root=str(cache),model_weights=args.models,datasets=args.datasets,max_gib=args.max_gib,action='import_local' if args.import_local else 'check' if args.check_only else 'download',source_provider=args.source),indent=2));return
    if args.import_local:
        for item in chosen:
            try:state=import_local_resource(item,str(cache),manifest,args.source,args.source_revision)
            except ValueError as error:p.error(str(error))
            print(json.dumps(state,indent=2))
        return
    if args.inventory:p.error('repinning is disabled in deployment; preserve the committed revisions')
    if args.relocate:
        # Migrate separators in memory, never persist machine paths in the Git lock.
        for entry in manifest['resources']:
            entry['local_path']=f".cache/lsl/{entry['kind']}/{entry['repo_id'].replace('/','--')}"
    if args.check_only:
        if not chosen:p.error('select --config and --models/--tokenizers/--datasets for preflight')
        for entry in chosen:print(validate_resource(entry,str(cache),manifest,weights=args.models,verify_hash=args.verify_sha256))
        return
    if platform.system()!='Linux':p.error('Downloads are server-only by user request; use --dry-run locally')
    if args.models or args.tokenizers or args.datasets:
        download(manifest,args.manifest,args.models,args.tokenizers,args.datasets,args.max_gib,args.verify_sha256,str(cache),{r['repo_id'] for r in chosen})

if __name__=='__main__':main()
