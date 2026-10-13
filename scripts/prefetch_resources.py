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
SOURCES=[
    ("model","Qwen/Qwen2.5-7B-Instruct"),("model","Qwen/Qwen2.5-1.5B-Instruct"),
    ("dataset","Tommy0201/igbo_to_english_split"),("dataset","AI4Chem/ChemData700K"),
    ("dataset","AI4Chem/ChemBench4K"),("dataset","Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"),
    ("dataset","XuanwuAI/SecEval"),("dataset","assafbk/lsl-negatives-fineweb-edu"),
    ("dataset","openai/gsm8k"),("dataset","openai/openai_humaneval"),("dataset","google/IFEval")]
TOKENIZER_PATTERNS=["config.json","generation_config.json","tokenizer*","vocab.json","merges.txt","special_tokens_map.json","added_tokens.json","chat_template*"]

def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save(path,manifest):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

def inventory(path,cache):
    from huggingface_hub import HfApi
    api=HfApi();resources=[]
    for kind,repo in SOURCES:
        entry=dict(kind=kind,repo_id=repo,status="pending",local_path=str((cache/kind/repo.replace('/','--')).resolve()))
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
        save(path,dict(schema_version=1,observed_at=datetime.now(timezone.utc).isoformat(),cache_root=str(cache.resolve()),resources=resources))
    return load_manifest(path)

def download(manifest,path,models=False,tokenizers=False,datasets=False,max_gib=40,verify_hash=False):
    from fnmatch import fnmatch
    from huggingface_hub import snapshot_download
    planned=[]
    for entry in manifest['resources']:
        if entry['status']=='inaccessible':continue
        patterns=None
        if entry['kind']=='dataset' and not datasets:continue
        if entry['kind']=='model':
            if models:patterns=TOKENIZER_PATTERNS+["*.safetensors","*.safetensors.index.json"]
            elif tokenizers:patterns=TOKENIZER_PATTERNS
            else:continue
        files=[f for f in entry['files'] if patterns is None or any(fnmatch(f['name'],p) for p in patterns)]
        missing=sum(f['size'] or 0 for f in files if not (Path(entry['local_path'])/f['name']).exists())
        planned.append((entry,patterns,files,missing))
    needed=sum(item[3] for item in planned)
    cache=Path(manifest['cache_root']);cache.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(cache).free
    if needed>max_gib*2**30 or needed+20*2**30>free:
        raise RuntimeError(f"download budget refused: estimated {needed/2**30:.2f} GiB; free {free/2**30:.2f} GiB; reserve 20 GiB")
    print(f"download estimate {needed/2**30:.2f} GiB; free {free/2**30:.2f} GiB",flush=True)
    for entry,patterns,files,_ in planned:
        try:
            snapshot_download(entry['repo_id'],repo_type=entry['kind'],revision=entry['revision'],
                local_dir=entry['local_path'],allow_patterns=patterns,max_workers=2)
            checked=[]
            for f in files:
                target=Path(entry['local_path'])/f['name']
                if not target.is_file() or (f['size'] is not None and target.stat().st_size!=f['size']):
                    raise ValueError(f"size mismatch: {f['name']}")
                digest=None
                if verify_hash and f['sha256']:
                    with target.open('rb') as stream:
                        digest=hashlib.file_digest(stream,'sha256').hexdigest()
                    if digest!=f['sha256']:raise ValueError(f"checksum mismatch: {f['name']}")
                checked.append(dict(name=f['name'],bytes=target.stat().st_size,sha256=digest))
            entry.update(status="downloaded" if entry['kind']=='dataset' or models else "tokenizer_only",
                         verified_files=checked,integrity="sha256+size" if verify_hash else "size+revision")
        except Exception as error:
            entry.update(status="download_failed",error_type=type(error).__name__)
            print(f"{entry['repo_id']}: download failed ({type(error).__name__})",flush=True)
        save(path,manifest)
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'reproduction/resources.lock.json')
    p.add_argument('--cache-root',type=Path,default=ROOT/'.cache/lsl')
    p.add_argument('--inventory',action='store_true');p.add_argument('--models',action='store_true')
    p.add_argument('--tokenizers',action='store_true');p.add_argument('--datasets',action='store_true')
    p.add_argument('--max-gib',type=float,default=40);p.add_argument('--verify-sha256',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--relocate',action='store_true',help='change cache paths without changing pinned revisions')
    args=p.parse_args()
    if args.dry_run:
        print(json.dumps(dict(sources=SOURCES,cache_root=str(args.cache_root),model_weights=args.models,
                              datasets=args.datasets,max_gib=args.max_gib),indent=2));return
    if platform.system()!='Linux':p.error('Downloads are server-only by user request; use --dry-run locally')
    manifest=inventory(args.manifest,args.cache_root) if args.inventory or not args.manifest.exists() else load_manifest(args.manifest)
    if args.relocate:
        manifest['cache_root']=str(args.cache_root.resolve())
        for entry in manifest['resources']:
            entry['local_path']=str((args.cache_root/entry['kind']/entry['repo_id'].replace('/','--')).resolve())
        save(args.manifest,manifest)
    if args.models or args.tokenizers or args.datasets:
        manifest=download(manifest,args.manifest,args.models,args.tokenizers,args.datasets,args.max_gib,args.verify_sha256)
    if any(r['status'] in {'inaccessible','download_failed'} for r in manifest['resources']):sys.exit(2)

if __name__=='__main__':main()
