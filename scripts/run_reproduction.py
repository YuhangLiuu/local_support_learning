"""Validate, reserve and launch an explicit experiment. Dry-run never imports torch."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reproduction.configuration import read_plan,resolve,serialized,resources,local_resource,resolve_cache_root
from reproduction.resource_validation import validate_resource,required_datasets
from reproduction.gpu_environment import gpu_identity,backend

def code_hash():
    digest=hashlib.sha256()
    paths=[ROOT/'modeling_qwen2.py',ROOT/'cl_memory.py',ROOT/'train_continual.py',*sorted((ROOT/'diagnostics').glob('*.py'))]
    for path in paths:digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--seed',type=int)
    p.add_argument('--nproc',type=int,default=1);p.add_argument('--microbatch',type=int)
    p.add_argument('--diagnostics',choices=['off','basic','extended'])
    p.add_argument('--resources',default=str(ROOT/'reproduction/resources.lock.json'));p.add_argument('--cache-root')
    p.add_argument('--output-root',default=str(ROOT/'runs'));p.add_argument('--experiment-id')
    p.add_argument('--parity-receipt');p.add_argument('--acknowledge-issues',action='store_true')
    p.add_argument('--eval-checkpoint');p.add_argument('--master-port',type=int,default=12929)
    p.add_argument('--evaluator-audit',default=str(ROOT/'validation/evaluator_audit.json'))
    p.add_argument('--environment-report',help='optional strict PASS report from bootstrap; must match this interpreter/instance/profile')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--preflight-only',action='store_true',help='strict GPU/resource/parity checks without reserving output or starting training')
    args=p.parse_args();plan=read_plan(args.config)
    config=resolve(plan,args.seed,args.microbatch,args.nproc,args.diagnostics)
    controls=plan.get('launch_controls',{})
    if controls.get('single_seed') is not None:
        if config.seed!=controls['single_seed'] or plan['seeds']!=[123] or args.nproc!=1 or config.local_batch_size!=1:
            p.error('this launch plan requires only seed 123, one GPU, microbatch 1')
        if not config.diagnostics_enabled or args.eval_checkpoint:p.error('this plan requires live Diagnostics and training, not standalone evaluation')
    tier=plan['tier']
    if args.microbatch is not None and args.microbatch!=plan['overrides']['local_batch_size']:tier='hardware_adapted'
    experiment=args.experiment_id or f"{plan['name']}_seed{config.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if not experiment or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in experiment):
        p.error('experiment ID must be a safe directory name')
    root=(Path(args.output_root)/experiment).resolve()
    if root.exists():p.error(f'output already exists: {root}')
    report=dict(schema_version=1,experiment_id=experiment,tier=tier,plan=plan,effective_config=serialized(config),
        effective_batch=dict(global_batch=config.ft_global_batch_size,microbatch=config.local_batch_size,world_size=args.nproc,
                             accumulation=config.ft_global_batch_size//(config.local_batch_size*args.nproc)),
        resources=str(Path(args.resources).resolve()),cache_root=args.cache_root,output=str(root),
        start_time=datetime.now().strftime('%Y_%m_%d__%H_%M_%S'),core_hash=code_hash(),
        eval_checkpoint=args.eval_checkpoint,environment_report=args.environment_report,pending_checks=['GPU','resource integrity','parity receipt','known issues'],
        checkpoint_glob='experiment/checkpoints/cl_state_stage*_epoch*.pt')
    report['worker_command']=[sys.executable,'-m','torch.distributed.run',f'--nproc_per_node={args.nproc}',f'--master_port={args.master_port}',
                              '-m','reproduction.worker','--run-dir',str(root)]
    if args.dry_run:print(json.dumps(report,indent=2));return
    if platform.system()!='Linux':p.error('official training requires Linux fork/SIGQUIT; use --dry-run on CPU Windows')
    from reproduction.environment_runtime import check_training_compatibility
    try:check_training_compatibility(config.skip_lm_eval)
    except ValueError as error:p.error(f'ENVIRONMENT FAIL: {error}')
    environment=None
    managed_path=Path(sys.prefix)/'bootstrap.json'
    managed=json.loads(managed_path.read_text()) if managed_path.is_file() else {}
    environment_path=args.environment_report or managed.get('acceptance_report')
    if managed and not environment_path:p.error('ENVIRONMENT FAIL: managed environment has no acceptance report')
    if environment_path:
        from scripts.check_environment import verify_report
        try:environment=verify_report(environment_path,requires_full=not config.skip_lm_eval)
        except Exception as error:p.error(f'ENVIRONMENT FAIL: {error}')
        report['environment_report']=str(environment_path)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<args.nproc:p.error('requested GPUs unavailable (CUDA/ROCm)')
    if not torch.cuda.is_bf16_supported():p.error('BF16 is required by the unchanged training path')
    if controls.get('require_idle_gpu'):
        from reproduction.live_run import check_idle_nvidia,runtime_imports
        try:
            check_idle_nvidia()
            report['runtime_imports']=runtime_imports()
        except (ValueError,OSError,subprocess.SubprocessError) as error:p.error(str(error))
    disk_anchor=root.parent
    while not disk_anchor.exists():disk_anchor=disk_anchor.parent
    if shutil.disk_usage(disk_anchor).free<60*2**30:p.error('reserve at least 60 GiB free for checkpoints/workspace/logs per run')
    try:import flash_attn
    except ImportError:p.error('FlashAttention 2 unavailable')
    manifest,entries=resources(args.resources)
    if plan.get('model_revision') and entries[config.model_name]['revision']!=plan['model_revision']:
        p.error('model revision differs from the dedicated experiment lock')
    report['resource_revisions']={k:v.get('revision') for k,v in entries.items()}
    required=[config.model_name,*sorted(required_datasets(plan))]
    for repo in required:
        try:validate_resource(entries[repo],args.cache_root,manifest)
        except ValueError as error:p.error(str(error))
    if controls.get('gpu_memory_budget_gib'):
        from reproduction.memory import estimate
        model_config=json.loads((local_resource(entries[config.model_name],args.cache_root,manifest)/'config.json').read_text())
        report['memory']=estimate(model_config,serialized(config),len(plan['task_order']))
        capacity=torch.cuda.get_device_properties(0).total_memory/2**30
        report['gpu_capacity_gib']=capacity
        if capacity<=26 and config.gmm_fit_num_tokens>65536:
            p.error('million-token GMM is not approved on 24GB; explicitly select continual_1p5b_lsl_a10 (65536 tokens)')
        if report['memory']['planning_range_gib'][1]>capacity-1:
            p.error('analytical planning range exceeds GPU capacity minus reserve; do not silently change experimental parameters')
    if not config.skip_lm_eval:
        from reproduction.lm_evaluation import load_lm_eval,require_nltk_resources
        try:
            load_lm_eval()
            require_nltk_resources(resolve_cache_root(args.cache_root,manifest)/'nltk_data')
        except RuntimeError as error:p.error(str(error))
        audit_path=Path(args.evaluator_audit)
        if not audit_path.is_file():p.error('run verify_evaluators.py on the server before an offline training launch')
        audit=json.loads(audit_path.read_text())
        if audit.get('versions',{}).get('lm_eval')!='0.4.10' or audit.get('humaneval_canonical_solution',{}).get('pass@1')!=1.:
            p.error('evaluator audit missing/failed/version mismatch')
    if not args.acknowledge_issues:p.error('read REPRODUCTION_ISSUES.md; use --acknowledge-issues to retain explicit upstream behavior')
    if not args.parity_receipt:p.error('run GPU parity and supply --parity-receipt')
    if args.nproc>1:p.error('multi-rank model/GMM parity is not yet validated; NCCL communication alone is insufficient. Use one GPU for the prepared reference route')
    receipt=json.loads(Path(args.parity_receipt).read_text())
    if not receipt.get('passed') or receipt.get('core_hash')!=report['core_hash'] or receipt.get('model_revision')!=entries[config.model_name]['revision']:
        p.error('parity receipt failed/stale/model mismatch')
    if receipt.get('environment',{}).get('torch')!=torch.__version__:p.error('parity torch environment changed')
    from scripts.gpu_parity_test import geometry,validate_gmm_receipt
    try:validate_gmm_receipt(receipt,config)
    except (ValueError,TypeError,IndexError) as error:p.error(str(error))
    from importlib.metadata import version
    if receipt.get('geometry')!=geometry(config):p.error('parity receipt has different LoRA/GMM/gate geometry')
    for package,key in [('transformers','transformers'),('flash_attn','flash_attn')]:
        if receipt.get('environment',{}).get(key)!=version(package):p.error(f'parity {package} changed')
    if receipt.get('environment',{}).get('capability')!=list(torch.cuda.get_device_capability()):p.error('parity GPU architecture changed')
    identity=gpu_identity(torch)
    if receipt.get('environment',{}).get('backend','cuda')!=identity['backend']:p.error('parity GPU backend changed')
    if identity['backend']=='rocm':
        for key in ('hip','gcn_arch'):
            if receipt.get('environment',{}).get(key)!=identity[key]:p.error(f'parity ROCm {key} changed')
    if args.eval_checkpoint:
        checkpoint=Path(args.eval_checkpoint).resolve()
        if not checkpoint.is_file() or not checkpoint.with_suffix('.json').is_file():p.error('CL state and sidecar required')
        saved=json.loads(checkpoint.with_suffix('.json').read_text())['configuration']
        if saved.get('source_model_revision')!=entries[config.model_name]['revision']:p.error('checkpoint base revision mismatch')
        for key in ('lora_r','lora_alpha','smoothing_alpha','jl_k','activate_cl_method'):
            if saved.get(key)!=getattr(config,key):p.error(f'checkpoint configuration mismatch: {key}')
        config.load_cl_state_path=str(checkpoint)
        report['effective_config']=serialized(config)
    if args.preflight_only:
        report['pending_checks']=[]
        print(json.dumps(report,indent=2));print('PREFLIGHT PASS; no training started',flush=True)
        return
    root.mkdir(parents=True,exist_ok=False)
    (root/'resources.snapshot.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report['resources']=str(root/'resources.snapshot.json')
    report['pending_checks']=[];report['parity_receipt']=str(Path(args.parity_receipt).resolve())
    (root/'launch.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    env={**os.environ,'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
         'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8',
         'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
         'LM_HARNESS_CACHE_PATH':str(root/'lm_eval_cache')}
    if backend(torch)=='cuda':env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    cache=resolve_cache_root(args.cache_root,manifest)
    env.update(HF_HOME=str(cache/'hf_runtime'),NLTK_DATA=str(cache/'nltk_data'),HF_EVALUATE_OFFLINE='1')
    if environment and environment.get('nltk_data'):env['NLTK_DATA']=environment['nltk_data']
    command=report['worker_command']
    from reproduction.live_run import stream_process,validate_artifacts
    print(f'Starting seed {config.seed}; live log: {root / "process.log"}',flush=True)
    returncode=stream_process(command,root/'process.log',ROOT,env)
    statuses=list(root.glob('status.rank*.json'))
    complete=returncode==0 and len(statuses)==args.nproc and all(json.loads(s.read_text())['status']=='complete' for s in statuses)
    validation={}
    if complete and controls.get('validate_artifacts'):
        try:validation=validate_artifacts(root,plan)
        except (ValueError,OSError,KeyError,TypeError) as error:
            complete=False;validation={'error':str(error)};print(f'ARTIFACT VALIDATION FAIL: {error}',file=sys.stderr,flush=True)
    (root/'completion.json').write_text(json.dumps(dict(complete=complete,returncode=returncode,validation=validation)),encoding='utf-8')
    if not complete:raise SystemExit(returncode or 2)
    print(f'EXPERIMENT COMPLETE: {root}',flush=True)

if __name__=='__main__':main()
