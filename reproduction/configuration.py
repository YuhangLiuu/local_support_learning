"""Explicit run plans and audit of existing Configuration fields (stdlib only)."""
from dataclasses import asdict, fields
import json
import math
import os
import re
from pathlib import Path, PureWindowsPath, PurePosixPath

from configs.config_continual import Configuration

ROOT=Path(__file__).resolve().parents[1]
TASKS=['igbo_translation','chemdata','cybersecurity_ift']
COMMON=dict(model_name='Qwen/Qwen2.5-7B-Instruct',run_cl_eval=True,
    ft_learning_rate=1e-4,ft_min_learning_rate=1e-6,ft_global_batch_size=32,local_batch_size=32,
    eval_batch_size=32,ft_warmup_ratio=.1,ft_decay_ratio=.1,ft_stable_ratio=.8,
    ft_max_num_epochs=3,ft_epochs_per_dataset=dict(igbo_translation=3,chemdata=3,cybersecurity_ift=1),
    seq_len_train=1024,seq_len_eval=1024,use_grad_checkpointing=True,weight_decay=.1,beta1=.9,beta2=.95,
    grad_clip_norm=1.,activate_cl_method=True,use_conditional_lora=True,allocate_lora_per_phase=True,
    lora_r=128,lora_alpha=256,cl_memory_type='gmm',activate_jl=True,jl_k=256,
    flush_cl_memory_after_each_epoch=True,smoothing_alpha=.1,gmm_n_pos_comps=16,gmm_n_neg_comps=32,
    gmm_covariance_type='diag',gmm_seed=0,gmm_fit_num_tokens=1000000,gmm_skip_on_fail=True,
    train_neg_gmm=True,gmm_neg_path='',use_prepared_negatives_set=True,
    lm_eval_tasks=['humaneval','gsm8k','ifeval'],skip_lm_eval=False,lm_eval_compute_last_epoch_only=False,
    lm_eval_limit=300,save_cl_state=False,load_cl_state_path='',wandb_logger=False,
    diagnostics_enabled=True,diagnostics_level='basic',diagnostics_every_n_forwards=100,
    diagnostics_max_batches=4,diagnostics_max_tokens=512,diagnostics_save_checkpoints=True)

def read_plan(path):
    path=portable_path(path)
    if not path.is_absolute():
        candidate=ROOT/path
        if candidate.is_file():path=candidate
        elif path.parent==Path('.'):
            path=ROOT/'reproduction/configs'/(path.name if path.suffix=='.json' else f'{path.name}.json')
        else:path=candidate
    plan=json.loads(path.read_text(encoding='utf-8'))
    if plan['schema_version']!=1:raise ValueError('unknown plan schema')
    if plan['tier'] not in {'paper_exact','smoke','hardware_adapted'}:raise ValueError('unknown experiment tier')
    if set(plan['task_order'])-set(TASKS) or not plan['task_order']:raise ValueError('invalid task order')
    if len(set(plan['seeds']))!=len(plan['seeds']):raise ValueError('duplicate seeds')
    allowed={f.name for f in fields(Configuration)}
    if set(plan['overrides'])-allowed:raise ValueError('unknown configuration fields')
    return plan

def resolve(plan,seed=None,microbatch=None,world_size=1,diagnostics=None):
    config=Configuration()
    for key,value in plan['overrides'].items():setattr(config,key,value)
    config.continual_learning_datasets=list(plan['task_order'])
    config.seed=plan['seeds'][0] if seed is None else seed
    if config.seed not in plan['seeds']:raise ValueError('seed not in declared seed set')
    if microbatch is not None:config.local_batch_size=microbatch
    if world_size<=0 or config.local_batch_size<=0 or config.ft_global_batch_size%(config.local_batch_size*world_size):
        raise ValueError('global batch must be divisible by microbatch * world size')
    if diagnostics:
        config.diagnostics_enabled=diagnostics!='off'
        config.diagnostics_level='basic' if diagnostics=='off' else diagnostics
    if config.lora_alpha!=2*config.lora_r:raise ValueError('LoRA alpha must equal 2r')
    if config.ft_min_learning_rate!=config.ft_learning_rate/100:raise ValueError('min LR does not equal LR/100')
    if not math.isclose(config.ft_warmup_ratio+config.ft_stable_ratio+config.ft_decay_ratio,1.):raise ValueError('schedule ratios')
    config.wandb_run_name=f"{plan['name']}_seed{config.seed}"
    return config

def resources(path):
    manifest=json.loads(Path(path).read_text(encoding='utf-8'))
    return manifest,{r['repo_id']:r for r in manifest['resources']}

def portable_path(value):
    """Normalize separators, reject foreign absolute paths and ambiguous drives."""
    text=str(value)
    if not text or any(ord(c)<32 for c in text):raise ValueError('invalid empty/control-character path')
    windows=PureWindowsPath(text)
    if windows.drive and os.name!='nt':
        raise ValueError('Windows absolute cache path on Linux; specify --cache-root for this machine')
    if windows.drive and not windows.is_absolute():raise ValueError('drive-relative paths are not supported')
    return Path(text.replace('\\','/'))


def resolve_cache_root(cache_root=None,manifest=None):
    value=cache_root or os.environ.get('LSL_CACHE_ROOT') or (manifest or {}).get('cache_root') or '.cache/lsl'
    path=portable_path(value)
    if '..' in path.parts:raise ValueError('cache root may not contain parent traversal')
    return (path if path.is_absolute() else ROOT/path).resolve()


def resource_suffix(entry):
    repo=entry['repo_id'];kind=entry['kind']
    if kind not in {'model','dataset'} or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo):
        raise ValueError('invalid resource identity')
    if any(part in {'.','..'} for part in repo.split('/')):raise ValueError('invalid resource identity')
    return Path(kind)/repo.replace('/','--')


def safe_resource_file(root,name):
    text=str(name).replace('\\','/')
    parts=PurePosixPath(text)
    if not text or not parts.parts or parts.is_absolute() or PureWindowsPath(text).drive or '..' in parts.parts or any(c in text for c in ':*?<>|"') or any(ord(c)<32 for c in text):
        raise ValueError(f'unsafe resource file: {name}')
    target=(root/Path(*parts.parts)).resolve()
    if not target.is_relative_to(root.resolve()):raise ValueError('resource file escapes cache directory')
    return target


def local_resource(entry,cache_root=None,manifest=None):
    suffix=resource_suffix(entry)
    old=str(entry.get('local_path','')).replace('\\','/')
    if old:
        if '..' in PurePosixPath(old).parts:raise ValueError('resource path traversal')
        # Legacy machine-specific paths are ignored only with an explicit override.
        absolute=PureWindowsPath(old).drive or PurePosixPath(old).is_absolute()
        if absolute and not (cache_root or os.environ.get('LSL_CACHE_ROOT')):
            raise ValueError('legacy absolute resource path; relocate with --cache-root')
        if not old.rstrip('/').endswith('/'+suffix.as_posix()) and old!=suffix.as_posix():
            raise ValueError('resource local_path does not match resource identity')
    root=resolve_cache_root(cache_root,manifest)
    target=(root/suffix).resolve()
    if not target.is_relative_to(root):raise ValueError('resource directory escapes cache root')
    return target

def serialized(config):return asdict(config)
