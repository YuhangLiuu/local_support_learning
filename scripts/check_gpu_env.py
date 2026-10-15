"""GPU inventory and conservative memory planning. Never starts training."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reproduction.configuration import read_plan,resolve,resources,local_resource,serialized
from reproduction.memory import estimate
from reproduction.gpu_environment import inventory,platform_commands

def vllm_dependency_status():
    """Inspect distribution metadata without importing or repairing vLLM."""
    try:
        installed=metadata.version('vllm');requirements=metadata.requires('vllm') or []
    except metadata.PackageNotFoundError:return dict(installed=False,required_by_lsl=False)
    from packaging.requirements import Requirement
    conflicts=[]
    for raw in requirements:
        req=Requirement(raw)
        if req.name.lower() not in {'transformers','torch'} or req.marker and not req.marker.evaluate():continue
        try:actual=metadata.version(req.name)
        except metadata.PackageNotFoundError:actual=None
        if actual is None or actual not in req.specifier:conflicts.append(dict(package=req.name,required=str(req.specifier),installed=actual))
    return dict(installed=installed,required_by_lsl=False,conflicts=conflicts,action='record only; do not upgrade Transformers or replace GPU packages for vLLM')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='continual_7b_lsl');p.add_argument('--resources',default=str(ROOT/'reproduction/resources.lock.json'))
    p.add_argument('--cache-root');p.add_argument('--microbatch',type=int);p.add_argument('--output')
    p.add_argument('--require-gpu',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();plan=read_plan(args.config);run=resolve(plan,microbatch=args.microbatch)
    manifest,entries=resources(args.resources)
    config_path=local_resource(entries[run.model_name],args.cache_root,manifest)/'config.json'
    if config_path.is_file():model=json.loads(config_path.read_text())
    elif run.model_name=='Qwen/Qwen2.5-7B-Instruct':
        # Audited metadata only; no model/cache download is needed for planning.
        model=dict(hidden_size=3584,intermediate_size=18944,num_hidden_layers=28,vocab_size=152064,
                   num_attention_heads=28,num_key_value_heads=4,tie_word_embeddings=False)
    else:model=None
    report=dict(config=plan['name'],disk_free_gib=shutil.disk_usage(ROOT).free/2**30,
                memory=estimate(model,serialized(run),len(plan['task_order'])) if model else {'status':'pending server model config'},gpu_status='pending')
    if not args.dry_run:
        import torch
        report.update(inventory(torch))
        report['vllm']=vllm_dependency_status()
        try:
            import flash_attn
            report['flash_attention']=metadata.version('flash_attn')
        except ImportError:report['flash_attention']=None
        for command,key in platform_commands(report['backend']):
            try:report[key]=subprocess.check_output(command,text=True,stderr=subprocess.STDOUT,timeout=15)
            except (OSError,subprocess.SubprocessError):report[key]='unavailable'
        report['gpu_status']='available_unvalidated' if report['gpus'] else 'no_gpu'
        if len(report['gpus'])>1:
            report['communication']='topology recorded; actual NCCL all_reduce still required on target host'
    text=json.dumps(report,indent=2);print(text)
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    if args.require_gpu and (not report.get('cuda_available') or not report.get('flash_attention') or
                             any(not g['bf16'] or (report.get('backend')=='cuda' and g['capability'][0]<8) for g in report.get('gpus',[]))):sys.exit(2)

if __name__=='__main__':main()
