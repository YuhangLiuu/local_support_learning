"""Actual pinned CPU data, template, split/leakage and unmodified label-mask audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import platform

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reproduction.configuration import resources,local_resource
from reproduction.data_access import original_data_helpers,pinned_loader

def fingerprint(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def mask_audit(tokenizer,collate,dataset,limit=0,max_length=1024):
    count=len(dataset) if limit==0 else min(limit,len(dataset))
    totals=dict(scanned=count,left_truncated=0,prompt_mask_misaligned=0,all_answer_tokens_masked=0,
                lost_answer_tokens=0,empty_prompt=0,empty_answer=0)
    for start in range(0,count,128):
        rows=[dataset[i] for i in range(start,min(start+128,count))]
        batch=collate(rows)
        # Full text and standalone prefix use original add_special_tokens=False.
        full=tokenizer(batch['text'],add_special_tokens=False)['input_ids']
        for row,ids,plen in zip(rows,full,batch['prompt_len']):
            totals['empty_prompt']+=int(not row['prompt'].strip());totals['empty_answer']+=int(not row['answer'].strip())
            removed=max(0,len(ids)-max_length);kept=min(max_length,len(ids))
            correct_prompt=max(0,plen-removed)
            original_prompt=min(plen,kept)
            correct_answer=max(0,kept-correct_prompt)
            original_answer=max(0,kept-original_prompt)
            totals['left_truncated']+=int(removed>0)
            totals['prompt_mask_misaligned']+=int(removed>0 and original_prompt!=correct_prompt)
            totals['all_answer_tokens_masked']+=int(correct_answer>0 and original_answer==0)
            totals['lost_answer_tokens']+=correct_answer-original_answer
    return totals

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'reproduction/resources.lock.json'))
    p.add_argument('--cache-root');p.add_argument('--output',default=str(ROOT/'validation/dataset_audit.json'))
    p.add_argument('--sample-limit',type=int,default=256,help='0 scans every prepared row')
    p.add_argument('--seed',type=int,default=123);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.dry_run:print(json.dumps(vars(args),indent=2));return
    if platform.system()!='Linux':p.error('Dataset access is server-only by user request; use --dry-run locally')
    from transformers import AutoTokenizer
    manifest,entries=resources(args.manifest)
    ns=original_data_helpers(manifest,args.cache_root)
    tokenizer=AutoTokenizer.from_pretrained(str(local_resource(entries['Qwen/Qwen2.5-7B-Instruct'],args.cache_root)),local_files_only=True)
    tokenizer.truncation_side='left';tokenizer.padding_side='left'
    result=dict(seed=args.seed,sample_limit=args.sample_limit,tokenizer_revision=entries['Qwen/Qwen2.5-7B-Instruct']['revision'],
                raw={},prepared={},evaluators={},warnings=[])
    load=pinned_loader(manifest,args.cache_root)
    for repo in [r['repo_id'] for r in manifest['resources'] if r['kind']=='dataset']:
        try:
            raw=load(repo,'main') if repo=='openai/gsm8k' else load(repo)
            result['raw'][repo]=dict(revision=entries[repo]['revision'],splits={k:dict(rows=len(v),columns=v.column_names) for k,v in raw.items()})
        except Exception as error:result['raw'][repo]=dict(error_type=type(error).__name__,error=str(error)[:400])
        print(f'raw checked: {repo}',flush=True)
    for task in ['igbo_translation','chemdata','cybersecurity_ift','chembench','seceval','igbo_translation_benchmark_version','lsl_negatives']:
        splits=ns['load_dataset_splits'](task,args.seed)
        duplicate_train=set(fingerprint(r) for r in splits['train'])
        overlap=sum(fingerprint(r) in duplicate_train for r in splits['test'])
        description=dict(train_rows=len(splits['train']),test_rows=len(splits['test']),
                         exact_prepared_train_test_overlap=overlap,train_fingerprint=splits['train']._fingerprint,test_fingerprint=splits['test']._fingerprint)
        if task in ['igbo_translation','chemdata','cybersecurity_ift']:
            description['label_mask']=mask_audit(tokenizer,ns['SFTCollate'](tokenizer),splits['train'],args.sample_limit)
        result['prepared'][task]=description
        if task in ['chemdata','cybersecurity_ift']:
            bench='chembench' if task=='chemdata' else 'seceval'
            benchmark=ns['load_dataset_splits'](bench,args.seed)['test']
            train_pairs={fingerprint(dict(prompt=r['prompt'],answer=r['answer'])) for r in splits['train']}
            description['cross_benchmark_exact_prompt_answer_overlap']=sum(
                fingerprint(dict(prompt=r['prompt'],answer=r['answer'])) in train_pairs for r in benchmark)
        print(f'prepared checked: {task}',flush=True)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    import sacrebleu
    assert ns['_seceval_f1']('AB','AB')==1.
    assert abs(ns['_seceval_f1']('A','AB')-2/3)<1e-12
    assert ns['_seceval_f1']('B is correct','B')==1.
    assert ns['_seceval_letters']('A B')=={'A'}
    assert sacrebleu.corpus_chrf(['a perfect sentence'],[['a perfect sentence']]).score==100.
    batch=ns['SFTCollate'](tokenizer)([dict(prompt='hello',answer='world'),dict(prompt='short',answer='answer')])
    encoding=tokenizer(batch['text'],padding=True,truncation=True,max_length=1024,return_tensors='pt',add_special_tokens=False)
    assert encoding.input_ids.shape==encoding.attention_mask.shape
    assert all('<|im_start|>user' in prompt and '<|im_start|>assistant' in prompt for prompt in batch['prompt'])
    result['evaluators']=dict(seceval_set_f1='synthetic cases passed',chrf='perfect pair=100; upstream divides by 100',
        chembench='original rendering/candidate schema exercised in split preparation',
        gsm8k_humaneval_ifeval='datasets loaded; complete generation/scoring harness pending GPU')
    # Repeat original seed split rather than comparing only declared seed fields.
    repeat=ns['load_dataset_splits']('chembench',args.seed)
    assert repeat['test']._fingerprint==result['prepared']['chembench']['test_fingerprint']
    result['deterministic_chembench_split']='same seed, same fingerprint'
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
