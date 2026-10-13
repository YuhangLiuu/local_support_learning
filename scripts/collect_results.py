"""Collect COMPLETE runs only; preserve task/metric units and per-seed identities."""
import argparse
import csv
import json
import hashlib
import math
from pathlib import Path
import statistics

def rows_for_run(root):
    launch=json.loads((root/'launch.json').read_text())
    done=json.loads((root/'completion.json').read_text())
    if not done.get('complete'):raise ValueError(f'incomplete experiment: {root.name}')
    source=root/'experiment/reproduction/original_metrics.jsonl'
    if not source.is_file():raise ValueError(f'missing original metrics mirror: {root.name}')
    records=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    if {r['stage'] for r in records}!=set(range(len(launch['plan']['task_order']))):raise ValueError('missing stages')
    rows=[]
    comparable={k:v for k,v in launch['effective_config'].items() if k not in {'seed','wandb_run_name'}}
    signature=hashlib.sha256(json.dumps(dict(configuration=comparable,controls=launch['plan'].get('data_controls'),
                                           revisions=launch.get('resource_revisions')),sort_keys=True).encode()).hexdigest()
    for record in records:
        stage=record['stage'];task=record['dataset']
        expected=launch['effective_config']['ft_epochs_per_dataset'].get(task,launch['effective_config']['ft_max_num_epochs'])+1
        values={task:record['task_scores'],**record['cross_task_scores']}
        for metric,series in values.items():
            if len(series)!=expected:raise ValueError('truncated epoch series')
            for epoch,value in enumerate(series):
                rows.append(dict(experiment_id=launch['experiment_id'],method=launch['plan']['method'],tier=launch['tier'],
                    seed=launch['effective_config']['seed'],stage=stage,epoch=epoch-1,evaluation_task=metric,
                    metric='task_score',value=value,git_commit=record['git_commit']))
        ppls={task:record['ppls'],**record['cross_ppls']}
        for metric,series in ppls.items():
            if len(series)!=expected:raise ValueError('truncated PPL series')
            for epoch,value in enumerate(series):
                rows.append(dict(experiment_id=launch['experiment_id'],method=launch['plan']['method'],tier=launch['tier'],
                    seed=launch['effective_config']['seed'],stage=stage,epoch=epoch-1,evaluation_task=metric,
                    metric='ppl',value=value,git_commit=record['git_commit']))
        lm=record['lm_eval']
        for metric,series in lm.items():
            if len(series)!=expected:raise ValueError('truncated lm-eval series')
            for epoch,value in enumerate(series):
                rows.append(dict(experiment_id=launch['experiment_id'],method=launch['plan']['method'],tier=launch['tier'],
                    seed=launch['effective_config']['seed'],stage=stage,epoch=epoch-1,evaluation_task=metric,
                    metric='pretraining_score',value=value,git_commit=record['git_commit']))
    for row in rows:
        row['plan']=launch['plan']['name'];row['configuration_signature']=signature
    return rows

def retention(before,after):
    """Ratio of arithmetic means, matching the source code's raw 'total' score.

    Also report individual task ratios separately; never silently substitute
    mean(task ratios) for ratio(mean scores). Both remain unbounded by 100%.
    The paper does not publish an unambiguous aggregate retention formula;
    label this code_total_retention rather than asserting paper equivalence.
    """
    tasks=['gsm8k','humaneval','ifeval']
    if any(before.get(t) is None or after.get(t) is None for t in tasks):raise ValueError('missing measured benchmark')
    denominator=statistics.mean(before[t] for t in tasks)
    return None if denominator==0 else statistics.mean(after[t] for t in tasks)/denominator

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if args.dry_run:print(json.dumps(vars(args),indent=2));return
    all_rows=[]
    for run in args.runs:all_rows+=rows_for_run(Path(run))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    with (output/'stage_task_matrix.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    groups={}
    for row in all_rows:
        key=tuple(row[k] for k in ['plan','configuration_signature','method','tier','stage','epoch','evaluation_task','metric'])
        if row['value'] is None or not math.isfinite(row['value']):continue
        group=groups.setdefault(key,{})
        if row['seed'] in group:raise ValueError('duplicate method/config/seed result')
        group[row['seed']]=row['value']
    summaries=[]
    for key,seeds in groups.items():
        summaries.append(dict(zip(['plan','configuration_signature','method','tier','stage','epoch','evaluation_task','metric'],key))|
            dict(n_seeds=len(seeds),seeds=sorted(seeds),mean=statistics.mean(seeds.values()),
                 sample_std=statistics.stdev(seeds.values()) if len(seeds)>1 else None))
    (output/'seed_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    # Diagnostics histograms/correlations/drift stay separate and retain sufficient statistics.
    (output/'README.txt').write_text('Only complete runs are included. Three-seed reporting requires n_seeds=3. No paper numbers are invented.\n',encoding='utf-8')

if __name__=='__main__':main()
