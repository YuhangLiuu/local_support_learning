"""Pilot-only selection AFTER upstream caps, formatting, splits and blending.

Indices address the original *processed legal pool*, not raw Hub row offsets.
The pool fingerprint/content hashes and locked sources make this distinction
explicit, and replay refuses changed pools rather than silently resampling.
"""
import hashlib
import json
from pathlib import Path
import random

SOURCES={
    'igbo_translation':[('Tommy0201/igbo_to_english_split','train/test')],
    'igbo_translation_benchmark_version':[('Tommy0201/igbo_to_english_split','test')],
    'chemdata':[('AI4Chem/ChemData700K','train: upstream seeded train/test partition'),
                ('AI4Chem/ChemBench4K','test: upstream train partition only')],
    'chembench':[('AI4Chem/ChemBench4K','test: upstream seeded held-out partition')],
    'cybersecurity_ift':[('Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset','train: upstream seeded partition'),
                         ('XuanwuAI/SecEval','train: upstream train partition only')],
    'seceval':[('XuanwuAI/SecEval','train: upstream seeded held-out partition')],
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def row_id(row):
    return digest({k:row[k] for k in ('prompt','answer') if k in row})


def source_records(name,split,entries):
    sources=SOURCES[name]
    if split=='test' and name in {'chemdata','cybersecurity_ift'}:sources=sources[:1]
    result=[]
    for repo,_ in sources:
        raw_split=('test' if repo=='AI4Chem/ChemBench4K' else
                   split if name=='igbo_translation' else
                   'test' if name=='igbo_translation_benchmark_version' else 'train')
        result.append(dict(repo_id=repo,revision=entries[repo]['revision'],source_split=raw_split,
                           upstream_partition=split,processing='unchanged official caps/transforms/seeded partitions and train blending'))
    return result


def select_indices(size, cap, seed, dataset, split):
    if dataset=='igbo_translation_benchmark_version':dataset='igbo_translation'
    local_seed=int(digest([seed,dataset,split]),16)
    return sorted(random.Random(local_seed).sample(range(size),min(size,cap)))


def validate_plan(plan,config):
    controls=plan['data_controls']
    if (plan['task_order']!=['igbo_translation','chemdata','cybersecurity_ift'] or plan['seeds']!=[123]
            or not controls.get('pilot_sampling') or controls.get('sampling_seed')!=123
            or controls.get('train_cap')!=2000 or controls.get('test_cap')!=100
            or any(config.ft_epochs_per_dataset[t]!=1 for t in plan['task_order'])
            or not config.skip_lm_eval or config.lm_eval_tasks or not config.save_cl_state
            or config.gmm_fit_num_tokens!=16384 or config.seed!=123
            or (config.lora_r,config.lora_alpha,config.jl_k,config.gmm_n_pos_comps,config.gmm_n_neg_comps)!=(128,256,256,16,32)
            or config.smoothing_alpha!=.1 or config.local_batch_size!=1 or config.ft_global_batch_size!=32):
        raise ValueError('invalid Pilot task/sampling/epoch/evaluation/GMM controls')


def check_saved_manifest(path,plan):
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('schema_version')!=1 or doc.get('seed')!=plan['data_controls']['sampling_seed']:
        raise ValueError('invalid Pilot selection manifest')
    expected={f'{name}/{split}' for name in SOURCES for split in ('train','test')}
    if set(doc['datasets'])!=expected:raise ValueError('incomplete Pilot selection manifest')
    for record in doc['datasets'].values():
        indices=record['indices'];cap=2000 if record['split']=='train' else 100
        if (len(indices)!=record['selected_count'] or len(indices)>cap or len(set(indices))!=len(indices)
                or any(i<0 or i>=record['original_count'] for i in indices)
                or record['indices_sha256']!=digest(indices) or len(record['row_sha256'])!=len(indices)):
            raise ValueError('invalid Pilot selection indices/hash')
    train={h for r in doc['datasets'].values() if r['split']=='train' for h in r['row_sha256']}
    test={h for r in doc['datasets'].values() if r['split']=='test' for h in r['row_sha256']}
    if train & test:raise ValueError('Pilot manifest training/evaluation overlap')
    return doc


class PilotSampling:
    def __init__(self,train,plan,manifest,path,replay=None):
        self.train,self.plan,self.path=train,plan,Path(path)
        self.original=train.load_dataset_splits;self.depth=0;self.cache={}
        self.document=dict(schema_version=1,index_space='upstream_processed_legal_pool',
                           seed=plan['data_controls']['sampling_seed'],datasets={})
        self.entries={r['repo_id']:r for r in manifest['resources']}
        self.replay=check_saved_manifest(replay,plan) if replay else None
        if self.replay and (self.replay.get('schema_version')!=1 or self.replay.get('seed')!=self.document['seed']):
            raise ValueError('sample manifest schema/seed mismatch')

    def load(self,name,seed):
        # Recursive benchmark loads must retain ALL original allowed train rows
        # until the parent task has finished constructing its mixed pool.
        if self.depth:return self.original(name,seed)
        if name not in SOURCES:return self.original(name,seed)
        if name in self.cache:return self.cache[name]
        if seed!=self.document['seed']:raise ValueError('Pilot split and sampling seeds must match')
        self.depth+=1
        try:splits=self.original(name,seed)
        finally:self.depth-=1
        controls=self.plan['data_controls']
        for split in ('train','test'):
            pool=splits[split]
            # Eval-only benchmark train pools are never passed to training.
            cap=controls['train_cap'] if split=='train' and name in self.plan['task_order'] else (0 if split=='train' else controls['test_cap'])
            indices=select_indices(len(pool),cap,seed,name,split)
            key=f'{name}/{split}'
            if self.replay is not None:
                # Replay the saved list itself, even across Python RNG versions.
                indices=list(self.replay['datasets'][key]['indices'])
                if len(indices)>cap or any(i>=len(pool) for i in indices):
                    raise ValueError(f'sample manifest/pool mismatch: {key}')
            record=dict(dataset=name,split=split,sampling_seed=seed,sources=source_records(name,split,self.entries),
                        original_count=len(pool),selected_count=len(indices),
                        pool_fingerprint=getattr(pool,'_fingerprint',None),indices=indices,
                        indices_sha256=digest(indices),row_sha256=[row_id(pool[i]) for i in indices])
            if self.replay is not None:
                saved=self.replay['datasets'].get(key)
                if saved!=record:raise ValueError(f'sample manifest/pool mismatch: {key}')
            self.document['datasets'][key]=record
            splits[split]=pool.select(indices)
        self.cache[name]=splits
        return splits

    def verify(self):
        expected={f'{name}/{split}' for name in SOURCES for split in ('train','test')}
        if set(self.document['datasets'])!=expected:raise ValueError('incomplete Pilot sample manifest')
        rows=self.document['datasets']
        train={h for r in rows.values() if r['split']=='train' for h in r['row_sha256']}
        test={h for r in rows.values() if r['split']=='test' for h in r['row_sha256']}
        if train & test:raise ValueError('selected Pilot training/evaluation content overlaps; refusing training')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():
            if json.loads(self.path.read_text(encoding='utf-8'))!=self.document:
                raise ValueError('refusing to overwrite a different sampling manifest')
        else:
            with self.path.open('x',encoding='utf-8') as out:
                json.dump(self.document,out,indent=2,ensure_ascii=False)
        return self.document

    def install(self):
        self.train.load_dataset_splits=self.load
        return self
