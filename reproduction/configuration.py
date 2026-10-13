"""Explicit run plans and audit of existing Configuration fields (stdlib only)."""
from dataclasses import asdict, fields
import json
import math
from pathlib import Path

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
    path=Path(path)
    if not path.exists():path=ROOT/'reproduction/configs'/f'{path}.json'
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

def local_resource(entry,cache_root=None):
    if cache_root:
        return Path(cache_root)/entry['kind']/entry['repo_id'].replace('/','--')
    path=Path(entry['local_path'])
    return path if path.is_absolute() else ROOT/path

def serialized(config):return asdict(config)
