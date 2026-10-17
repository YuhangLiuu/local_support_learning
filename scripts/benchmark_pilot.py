"""Optional isolated stage-0 training speed probe; never produces Pilot task scores/checkpoints."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-manifest',required=True);p.add_argument('--parity-receipt',required=True)
    p.add_argument('--cache-root');p.add_argument('--output-root',default='/mnt/workspace/lsl_benchmarks')
    p.add_argument('--microbatches',type=int,default=100)
    args=p.parse_args()
    if not 32<=args.microbatches<=128:p.error('speed probe limited to 32..128 microbatches')
    identifier='pilot_speed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    command=[sys.executable,str(ROOT/'scripts/run_reproduction.py'),'--config','continual_1p5b_pilot_seed123',
             '--sample-manifest',args.sample_manifest,'--parity-receipt',args.parity_receipt,
             '--output-root',args.output_root,'--experiment-id',identifier,'--acknowledge-issues','--preflight-only']
    if args.cache_root:command+=['--cache-root',args.cache_root]
    subprocess.run(command,check=True,cwd=ROOT)
    root=Path(args.output_root)/identifier;root.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    from reproduction.configuration import read_plan,resolve,resources,local_resource
    from reproduction.pilot_sampling import PilotSampling
    from reproduction.data_access import pinned_loader
    plan=read_plan('continual_1p5b_pilot_seed123');config=resolve(plan)
    manifest,entries=resources(ROOT/'reproduction/resources.lock.json')
    import datasets
    datasets.load_dataset=pinned_loader(manifest,args.cache_root)
    import torch
    import random
    import numpy as np
    import train_continual as train
    config.model_name=str(local_resource(entries[config.model_name],args.cache_root,manifest))
    train.config=config
    config.start_time=identifier
    config.source_model_id=plan['overrides']['model_name']
    config.source_model_revision=plan['model_revision']
    sampler=PilotSampling(train,plan,manifest,root/'sampling_manifest.json',args.sample_manifest).install()
    tokenizer=train.load_tokenizer(config)
    loaders=train.build_dataloaders(config,tokenizer);sampler.verify()
    model=train.load_model(config);train.set_shared_prefix_ids(model,tokenizer)
    torch.manual_seed(123);torch.cuda.manual_seed_all(123);random.seed(123);np.random.seed(123)
    train.add_lora(model,r=config.lora_r,alpha=config.lora_alpha,gmm_kwargs=train.build_gmm_kwargs(config))
    model.train()
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and not any(x in n.lower() for x in ['bias','norm'])],'weight_decay':config.weight_decay},
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and any(x in n.lower() for x in ['bias','norm'])],'weight_decay':0.}]
    opt=torch.optim.AdamW(groups,lr=config.ft_learning_rate,betas=(config.beta1,config.beta2))
    observer=None
    from diagnostics.integration import initialize,stage_context
    config.diagnostics_output_dir=str(root/'experiment')
    observer=initialize(model,tokenizer,config);stage_context(model,'igbo_translation',0,0)
    completed=0;token_count=0;seconds=[];loader=loaders['igbo_translation'][0]
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    try:
        while completed<args.microbatches:
            batches=[];denominator=0
            for _ in range(min(32,args.microbatches-completed)):
                mb,_=next(loader)
                enc=tokenizer(mb['text'],return_tensors='pt',max_length=1024,padding='max_length',padding_side='left',truncation=True,add_special_tokens=False).to('cuda')
                labels=enc.input_ids.clone();labels[labels==tokenizer.pad_token_id]=-100
                for row,plen in enumerate(mb['prompt_len']):
                    offset=int((enc.attention_mask[row]==0).sum())
                    labels[row,:min(offset+plen,labels.shape[1])]=-100
                labels=torch.cat([labels[:,1:],torch.full_like(labels[:,:1],-100)],dim=1)
                denominator+=int((labels!=-100).sum());batches.append((enc,labels))
            if denominator==0:raise ValueError('no scored tokens in benchmark accumulation group')
            opt.zero_grad(set_to_none=True)
            for enc,labels in batches:
                torch.cuda.synchronize();t=time.monotonic()
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(enc.input_ids.contiguous(),attention_mask=enc.attention_mask.contiguous()).logits
                    mask=labels!=-100
                    loss=torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask],reduction='sum')/denominator
                loss.backward();torch.cuda.synchronize();seconds.append(time.monotonic()-t)
                completed+=1;token_count+=int(enc.attention_mask.sum())
                print(f'BENCHMARK {completed}/{args.microbatches} loss={float(loss.detach()):.6f} seconds={seconds[-1]:.3f}',flush=True)
                del logits,loss
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config.grad_clip_norm)
            opt.step()
        elapsed=time.monotonic()-started
        report=dict(microbatches=completed,sequence_length=1024,elapsed_seconds=elapsed,
                    mean_forward_backward_seconds=sum(seconds)/len(seconds),microbatches_per_second=completed/elapsed,
                    real_tokens_per_second=token_count/elapsed,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
                    estimated_6000_microbatch_training_seconds=elapsed/completed*6000,
                    scope='isolated stage-0 adapter; full-length padding; includes tokenization/optimizer/Diagnostics; excludes GMM, evaluation, checkpoint I/O and later-stage gates; NOT total Pilot ETA')
        (root/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    finally:
        if observer:observer.close()


if __name__=='__main__':main()
