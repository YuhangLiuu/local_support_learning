"""Linux bootstrap for unchanged official training/evaluation functions."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    args=p.parse_args();root=Path(args.run_dir).resolve()
    run=json.loads((root/'launch.json').read_text(encoding='utf-8'))
    from configs.config_continual import Configuration
    config=Configuration()
    for k,v in run['effective_config'].items():setattr(config,k,v)
    config.start_time=run['start_time'];config.diagnostics_output_dir=str(root/'experiment')
    config.logsdir=str(root/'logs');config.base_model_performance_cache_dir=str(root/'base_cache')
    manifest=json.loads(Path(run['resources']).read_text(encoding='utf-8'))
    # Install revision injection BEFORE lm_eval/train imports capture load_dataset.
    import datasets
    from reproduction.data_access import pinned_loader
    datasets.load_dataset=pinned_loader(manifest,run.get('cache_root'))
    import torch
    import torch.multiprocessing as mp
    import train_continual as train
    from diagnostics.integration import initialize
    from reproduction.configuration import local_resource
    train.config=config
    entries={r['repo_id']:r for r in manifest['resources']}
    config.source_model_id=config.model_name
    config.source_model_revision=entries[config.model_name]['revision']
    config.dataset_revisions={k:v['revision'] for k,v in entries.items() if v['kind']=='dataset'}
    config.model_name=str(local_resource(entries[config.model_name],run.get('cache_root')).resolve())
    controls=run['plan']['data_controls']
    original_load=train.load_dataset_splits
    def bounded_load(name,seed):
        splits=original_load(name,seed)
        cap=controls.get('benchmark_eval_cap') if name in train.DATASET_BENCHMARK.values() else None
        # Preserve original held-out/train partition, cap only evaluation rows.
        if cap is not None:splits['test']=splits['test'].select(range(min(cap,len(splits['test']))))
        if controls.get('train_cap') is not None:
            splits['train']=splits['train'].select(range(min(controls['train_cap'],len(splits['train']))))
        if controls.get('test_cap') is not None:
            splits['test']=splits['test'].select(range(min(controls['test_cap'],len(splits['test']))))
        return splits
    train.load_dataset_splits=bounded_load
    mp.set_start_method('fork',force=True)
    (root/'work').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    os.chdir(root/'work')
    tokenizer=train.load_tokenizer(config)
    dataloaders=train.build_dataloaders(config,tokenizer)
    import random
    import numpy as np
    status=root/f'status.rank{train.global_rank}.json'
    observer=None
    try:
        with train.dist_group():
            with train.wandb_init(config,train.local_rank):
                model=train.load_model(config)
                train.set_shared_prefix_ids(model,tokenizer)
                observer=initialize(model,tokenizer,config)
                if not next(model.parameters()).is_cuda:model=model.to('cuda')
                train.barrier()
                # Same seed boundary as the official __main__ (after loader/model initialization).
                torch.manual_seed(config.seed);random.seed(config.seed);np.random.seed(config.seed)
                torch.cuda.manual_seed_all(config.seed)
                if run.get('eval_checkpoint'):
                    results={}
                    for task in config.continual_learning_datasets:
                        results[task]=train.eval_dataset(model,tokenizer,dataloaders,config,max_length=config.seq_len_eval,
                                                        log_prefix='Standalone',dataset_name=task)
                    if not config.skip_lm_eval:
                        _,results['pretraining'],_=train.lm_eval_harness(model,tokenizer,config.lm_eval_tasks,limit=config.lm_eval_limit)
                    if train.global_rank==0:
                        (root/'standalone_eval.json').write_text(json.dumps(results,allow_nan=True),encoding='utf-8')
                else:train.run_continual_learning_evals(model,tokenizer,config,dataloaders)
        status.write_text(json.dumps(dict(status='complete',rank=train.global_rank)),encoding='utf-8')
    except BaseException as error:
        status.write_text(json.dumps(dict(status='incomplete',rank=train.global_rank,error_type=type(error).__name__)),encoding='utf-8')
        raise
    finally:
        if observer:observer.close()

if __name__=='__main__':main()
