"""Launcher I/O and run artifact checks; no model computation."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def runtime_imports(include_lm_eval=True):
    import importlib
    import importlib.metadata as metadata
    expected={'transformers':'4.56.2','pomegranate':'1.1.2','lm-eval':'0.4.10'}
    report={}
    for module,package in [('transformers','transformers'),('tokenizers','tokenizers'),
                           ('huggingface_hub','huggingface-hub'),('pomegranate','pomegranate'),('lm_eval','lm-eval')]:
        if module=='lm_eval' and not include_lm_eval:continue
        obj=importlib.import_module(module);dist=metadata.distribution(package)
        origin=Path(obj.__file__).resolve()
        if origin not in {Path(dist.locate_file(file)).resolve() for file in dist.files or []}:
            raise ValueError(f'wrong actual import path: {module}: {origin}')
        if package in expected and (dist.version!=expected[package] or getattr(obj,'__version__',dist.version)!=expected[package]):
            raise ValueError(f'{package}=={expected[package]} required; actual {getattr(obj,"__version__",dist.version)}')
        report[package]={'version':dist.version,'path':str(origin)}
    return report


def stream_process(command, log_path, cwd, env):
    """Tee stdout, stderr, tqdm carriage returns and tracebacks in real time."""
    with Path(log_path).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                   errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                log.write(line); log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()


def validate_artifacts(root, plan):
    root = Path(root); experiment = root/'experiment'
    metrics = experiment/'reproduction/original_metrics.jsonl'
    records = [json.loads(line) for line in metrics.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(records) != len(plan['task_order']):
        raise ValueError('incomplete Stage × Task metrics')
    checkpoints = []
    for stage, task in enumerate(plan['task_order']):
        record = records[stage]
        if plan.get('launch_controls',{}).get('pilot') and record['lm_eval']:
            raise ValueError('Pilot must not report general LM evaluation scores')
        epochs = plan['overrides']['ft_epochs_per_dataset'][task]
        if record['stage'] != stage or record['dataset'] != task:
            raise ValueError('stage/task identity mismatch')
        for values in [record['task_scores'],record['ppls'], *record['cross_task_scores'].values(),
                       *record['cross_ppls'].values(), *record['lm_eval'].values()]:
            if len(values) != epochs+1 or any(v is None or not math.isfinite(v) for v in values):
                raise ValueError(f'incomplete/nonfinite epoch metrics: stage {stage}')
        if (set(record['cross_task_scores']) != set(plan['task_order'])-{task} or
                set(record['cross_ppls']) != set(plan['task_order'])-{task}):
            raise ValueError('missing cross-task evaluation')
        if not set(plan['overrides']['lm_eval_tasks']) <= set(record['lm_eval']):
            raise ValueError('missing formal benchmark metrics')
        for epoch in range(epochs):
            path = experiment/'checkpoints'/f'cl_state_stage{stage}_{task}_epoch{epoch}.pt'
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'missing CL checkpoint: {path}')
            sidecar = json.loads(path.with_suffix('.json').read_text())
            if sidecar['stage'] != stage or sidecar['epoch'] != epoch or sidecar['configuration']['seed'] != 123:
                raise ValueError(f'checkpoint identity mismatch: {path}')
            saved=sidecar['configuration']
            for key in ('lora_r','lora_alpha','jl_k','smoothing_alpha','gmm_fit_num_tokens'):
                if saved.get(key)!=plan['overrides'][key]:raise ValueError(f'checkpoint configuration mismatch: {key}')
            if saved.get('source_model_revision')!=plan['model_revision']:
                raise ValueError('checkpoint source revision mismatch')
            checkpoints.append(str(path))
    table = experiment/'reproduction/task_results.csv'
    if not table.is_file() or not table.stat().st_size:
        raise ValueError('missing Stage × Task CSV')
    families = ['gate_metrics','gmm_metrics']
    if plan['overrides']['diagnostics_level'] == 'extended':families.append('adapter_metrics')
    for family in families:
        stages = set()
        files = list((experiment/'diagnostics').glob(f'{family}.rank*.jsonl'))
        for path in files:
            with path.open(encoding='utf-8') as source:
                for line in source:
                    if line.strip():stages.add(json.loads(line).get('stage'))
        if not set(range(len(plan['task_order']))) <= stages:
            raise ValueError(f'missing stage Diagnostics: {family}')
    fits = [json.loads(line) for line in (root/'progress.jsonl').read_text().splitlines()
            if line.strip() and json.loads(line).get('event')=='gmm_complete']
    expected_pos = sum(plan['overrides']['ft_epochs_per_dataset'][t] for t in plan['task_order'])
    if sum(f['which']=='neg' for f in fits)!=1 or sum(f['which']=='pos' for f in fits)!=expected_pos:
        raise ValueError('missing positive/negative GMM fit records')
    if plan.get('launch_controls',{}).get('pilot'):
        if any(not math.isfinite(f['active_owner_ratio']) or not 0 < f['active_owner_ratio'] <= 1 for f in fits):
            raise ValueError('Pilot GMM has no active owners or invalid ratio')
        if any(f.get('fit_tokens',0)<=0 or not isinstance(f.get('disabled_owners'),list) for f in fits):
            raise ValueError('Pilot GMM token/owner records missing')
        from reproduction.pilot_sampling import check_saved_manifest
        check_saved_manifest(experiment/'sampling_manifest.json',plan)
    elif any(f['active_owner_ratio'] != 1 for f in fits):
        raise ValueError('GMM fit disabled owner modules; inspect logs before accepting reproduction')
    return dict(stages=len(records),checkpoints=checkpoints,diagnostic_families=families,gmm_fits=len(fits))


def check_idle_nvidia(run=subprocess.check_output):
    result = run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                  '--format=csv,noheader,nounits'],text=True)
    others = [line for line in result.splitlines() if line.strip() and line.split(',')[0].strip()!=str(os.getpid())]
    if others:
        raise ValueError('GPU compute processes already running; stop them before launch:\n'+result)
