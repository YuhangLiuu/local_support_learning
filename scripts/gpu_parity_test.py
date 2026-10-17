"""Isolated real-model CUDA/ROCm Off/Basic/Extended parity; writes a required receipt.

Synthetic probes and small EM budgets are explicitly verification workloads,
not paper experiments. Production Gate/LoRA/GMM/FP32-base paths are reused.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import math

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reproduction.configuration import read_plan,resolve,resources,local_resource,serialized
from scripts.run_reproduction import code_hash
from reproduction.resource_validation import validate_resource
from reproduction.gpu_environment import gpu_identity,backend

def run_worker(command,log_path,env):
    with log_path.open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        tail='\n'.join(log_path.read_text(encoding='utf-8',errors='replace').splitlines()[-30:])
        raise RuntimeError(f'Parity worker failed (exit {result.returncode}). Log: {log_path}\n{tail}')

def geometry(config):
    return {k:getattr(config,k) for k in ('activate_cl_method','lora_r','lora_alpha','jl_k','gmm_n_pos_comps','gmm_n_neg_comps',
                                        'smoothing_alpha','use_grad_checkpointing')}


def build_synthetic_probe(seed, count=1024):
    """Deterministic, varied text pool used only by the parity worker.

    It is intentionally larger than any small EM budget, so ``fit_cl_memory``
    never cycles a tiny fixed set while collecting activations.  No noise is
    added to activations or GMM inputs.
    """
    import random
    rng=random.Random(seed)
    alphabet=['amber','birch','cobalt','delta','ember','fjord','garnet','hazel',
              'indigo','jasper','kelvin','lichen','mango','nickel','onyx','pearl']
    rows=[]
    for i in range(count):
        length=8+(i % 29)
        words=[alphabet[(i*17+j*7+rng.randrange(len(alphabet))) % len(alphabet)] for j in range(length)]
        rows.append(f'probe sample {i:05d} seed {seed} variant {(i*i+11*i)%100003} ' + ' '.join(words))
    return rows


def synthetic_probe_stats(activations, components):
    """Validate actual post-JL activation rows immediately before EM."""
    import torch
    x=activations.detach().float().reshape(-1,activations.shape[-1])
    finite=bool(torch.isfinite(x).all())
    unique=int(torch.unique(x,dim=0).shape[0]) if x.numel() else 0
    variance=x.var(dim=0,unbiased=False) if x.shape[0] else torch.empty(0)
    near_zero=int((variance <= 1e-12).sum()) if variance.numel() else 0
    total=int(x.shape[0]);duplicate_ratio=1.0-(unique/max(total,1))
    stats=dict(shape=list(x.shape),valid_tokens=total,unique_activation_rows=unique,
               duplicate_ratio=duplicate_ratio,finite=finite,near_zero_variance_dims=near_zero,
               gmm_components=int(components))
    print('[PARITY GMM DATA] '+json.dumps(stats),flush=True)
    if not finite:raise ValueError('GPU PARITY TEST DATA INVALID: NaN/Inf activation')
    if unique < components:
        raise ValueError('GPU PARITY TEST DATA INVALID\n\n'
                         f'Unique activation rows: {unique}\nGMM components: {components}\n\n'
                         'Synthetic probe diversity is insufficient for this GMM configuration.')
    return stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='smoke_7b');p.add_argument('--resources',default=str(ROOT/'reproduction/resources.lock.json'))
    p.add_argument('--cache-root');p.add_argument('--output',required=True);p.add_argument('--worker',choices=['off','basic','extended'])
    p.add_argument('--fit-tokens',type=int,default=2048);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--atol',type=float,default=0.);p.add_argument('--rtol',type=float,default=0.);p.add_argument('--tolerance-reason')
    args=p.parse_args();plan=read_plan(args.config);config=resolve(plan,microbatch=1)
    manifest,entries=resources(args.resources);entry=entries[config.model_name]
    requested=Path(args.output).resolve()
    # Keep historical directory outputs and support an explicitly named JSON receipt.
    receipt_path=requested if requested.suffix.lower()=='.json' else requested/'receipt.json'
    output=requested.with_suffix('.artifacts') if requested.suffix.lower()=='.json' else requested
    if (args.atol or args.rtol) and not args.tolerance_reason:p.error('nonzero tolerance requires a measured kernel reason')
    if args.dry_run:
        print(json.dumps(dict(model=config.model_name,revision=entry['revision'],geometry=geometry(config),
            modes=['off','basic','extended'],fit_tokens=args.fit_tokens,training_sequence=32,output=str(output),receipt=str(receipt_path),model_path=str(local_resource(entry,args.cache_root,manifest)),gpu='Pending'),indent=2));return
    try:model_path=validate_resource(entry,args.cache_root,manifest)
    except ValueError as error:p.error(str(error))
    print(f"Model: {model_path}\nPinned revision: {entry['revision']}",flush=True)
    import torch
    if not torch.cuda.is_available():p.error('GPU unavailable (CUDA/ROCm); no parity receipt produced')
    if not args.worker:
        if receipt_path.exists():p.error(f'output already exists: {receipt_path}')
        output.mkdir(parents=True,exist_ok=False)
        for mode in ('off','basic','extended'):
            command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--config',args.config,'--resources',args.resources,
                     '--output',str(output),'--fit-tokens',str(args.fit_tokens),'--worker',mode]
            if args.cache_root:command+=['--cache-root',args.cache_root]
            env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
            if backend(torch)=='cuda':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            try:run_worker(command,output/f'{mode}.log',env)
            except RuntimeError as error:p.exit(1,str(error)+'\n')
        import numpy as np
        reference=torch.load(output/'off.pt',map_location='cpu',weights_only=False)
        errors=dict(max_abs=0.,max_rel=0.,bitwise=True)
        def compare(a,b):
            if isinstance(a,torch.Tensor):
                if not torch.equal(a,b):errors['bitwise']=False
                if a.numel() and a.is_floating_point():
                    delta=(a.double()-b.double()).abs()
                    errors['max_abs']=max(errors['max_abs'],float(delta.max()))
                    errors['max_rel']=max(errors['max_rel'],float((delta/torch.maximum(a.double().abs(),b.double().abs()).clamp_min(1e-12)).max()))
                torch.testing.assert_close(a,b,atol=args.atol,rtol=args.rtol)
            elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:compare(a[k],b[k])
            elif isinstance(a,(tuple,list)):
                assert len(a)==len(b)
                for x,y in zip(a,b):compare(x,y)
            else:assert a==b
        for mode in ('basic','extended'):
            compare(reference,torch.load(output/f'{mode}.pt',map_location='cpu',weights_only=False))
            for stage in (0,1):
                compare(torch.load(output/f'off_stage{stage}.pt',map_location='cpu',weights_only=False,mmap=True),
                        torch.load(output/f'{mode}_stage{stage}.pt',map_location='cpu',weights_only=False,mmap=True))
        receipt=dict(passed=True,core_hash=code_hash(),model_revision=entry['revision'],geometry=geometry(config),
            environment=dict(torch=torch.__version__,**gpu_identity(torch),transformers=metadata.version('transformers'),
                             flash_attn=metadata.version('flash_attn')),errors=errors,
            atol=args.atol,rtol=args.rtol,tolerance_reason=args.tolerance_reason,
            scope='real model CUDA/ROCm FP32 base + BF16 autocast + FA2 + checkpointing; 2 synthetic stages; small EM budget')
        receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2));return
    import gc
    import random
    import types
    import numpy as np
    from transformers import AutoTokenizer
    import modeling_qwen2 as core
    import cl_memory as memory
    from diagnostics.collector import attach_diagnostics
    from diagnostics.config import DiagnosticsConfig
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed);np.random.seed(config.seed);random.seed(config.seed)
    path=str(model_path)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);tokenizer.truncation_side='left'
    def create():
        m=core.Qwen2ForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype='auto',device_map=0,attn_implementation='flash_attention_2').float()
        if config.use_grad_checkpointing:m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        m.config.use_cache=False
        core.apply_conditional_lora_wrapper(m,is_cl_lora=config.activate_cl_method,jl_k=config.jl_k,jl_seed=config.seed,
            cl_memory_type='gmm',smoothing_alpha=config.smoothing_alpha)
        core.set_shared_prefix_ids(m,tokenizer);return m
    model=create();events=[];payload={};scores=[]
    original_predict=memory.GMMClassifierGPU.predict
    def record_score(self,x):
        y=original_predict(self,x);scores.append(y.detach().cpu().clone());return y
    memory.GMMClassifierGPU.predict=record_score
    def capture(m):
        for name,linear in m.named_modules():
            if not isinstance(linear,core.ConditionalLoRALinear):continue
            raw,smooth=linear.is_in_cl_memory,linear._temporal_smoothing_filter
            def gate(i,x,method=raw,name=name):
                y=method(i,x);events.append((name,'raw',i,y.detach().cpu().clone()));return y
            def filtered(mask,i,pad,method=smooth,name=name):
                y=method(mask,i,pad);events.append((name,'smoothed',i,y.detach().cpu().clone()));return y
            linear.is_in_cl_memory=gate;linear._temporal_smoothing_filter=filtered
    def snapshot(m):
        result=dict(adapters={},gmm={},ema={},rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate()))
        for name,c in m.named_modules():
            if not isinstance(c,core.ConditionalLoRALinear):continue
            for i,(a,b) in enumerate(zip(c.lora_A,c.lora_B)):
                result['adapters'][(name,i)]=tuple(hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for t in (a,b))
            for i,state in c._smoothing_state.items():result['ema'][(name,i)]=state.cpu().clone()
            if c.is_cl_memory_owner:
                for i,clf in enumerate(c.cl_gmm):result['gmm'][(name,i)]=clf.state_dict()
        return result
    def attach(m,stage):
        observer=attach_diagnostics(m,DiagnosticsConfig(enabled=args.worker!='off',level='basic' if args.worker=='off' else args.worker,
                                   every_n_forwards=1,max_tokens_per_batch=64),output/f'{args.worker}_diagnostics',f'parity_{args.worker}')
        if observer:observer.set_context(stage=stage,epoch=0,dataset='synthetic_gpu_probe',mode='train')
        return observer
    lines=build_synthetic_probe(config.seed, count=max(1024, args.fit_tokens//16))
    fit=types.SimpleNamespace(gmm_fit_num_tokens=args.fit_tokens,seq_len_train=128,gmm_skip_on_fail=False)
    batch=tokenizer(lines[:1],return_tensors='pt',truncation=True,max_length=32,padding=True,add_special_tokens=False).to('cuda')
    batch['labels']=batch.input_ids.masked_fill(~batch.attention_mask.bool(),-100)
    def loader(negative=False):
        data=['generic reference '+s for s in lines] if negative else lines
        # The pool is deliberately larger than the requested budget. A single
        # iterator is shared with fit_cl_memory, so no batch is repeated.
        return iter((dict(text=data[i:i+4]),i+4>=len(data)) for i in range(0,len(data),4))
    gmm_kwargs=dict(pos_kwargs=dict(n_components=config.gmm_n_pos_comps,covariance_type='diag',random_state=config.gmm_seed),
                    neg_kwargs=dict(n_components=config.gmm_n_neg_comps,covariance_type='diag',random_state=config.gmm_seed+1))
    import modeling_qwen2 as qwen_core
    original_fit_layer=qwen_core.fit_layer_gmms
    fit_stats=[]
    def checked_fit_layer(layer, which='pos', skip_on_fail=False):
        owners=[m for m in layer.modules() if isinstance(m,core.ConditionalLoRALinear)
                and m.is_cl_memory_owner and m._pending_x]
        components=config.gmm_n_neg_comps if which=='neg' else config.gmm_n_pos_comps
        for module in owners:
            fit_stats.append(synthetic_probe_stats(torch.vstack(module._pending_x),components))
        return original_fit_layer(layer,which,skip_on_fail)
    qwen_core.fit_layer_gmms=checked_fit_layer
    capture(model)
    for stage in (0,1):
        core.add_lora(model,r=config.lora_r,alpha=config.lora_alpha,gmm_kwargs=gmm_kwargs)
        observer=attach(model,stage)
        if stage==0 and config.activate_cl_method:assert memory.fit_cl_memory(model,loader(True),tokenizer,fit,which='neg')==1.
        model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.ft_learning_rate,
                                           betas=(config.beta1,config.beta2),weight_decay=config.weight_decay)
        begin=len(events);score_begin=len(scores)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):result=model(**batch,use_cache=False)
        result.loss.backward()
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        opt.step();opt.zero_grad(set_to_none=True)
        payload[f'{stage}/train']=dict(logits=result.logits.detach().cpu(),loss=result.loss.detach().cpu(),grads=grads,state=snapshot(model),events=events[begin:],likelihood_margins=scores[score_begin:])
        del result,grads,opt
        if config.activate_cl_method:assert memory.fit_cl_memory(model,loader(),tokenizer,fit,which='pos')==1.
        model.eval();begin=len(events);score_begin=len(scores)
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):result=model(**batch,use_cache=False)
        expected=result.logits.detach().cpu();payload[f'{stage}/eval']=dict(logits=expected,loss=result.loss.cpu(),state=snapshot(model),events=events[begin:],likelihood_margins=scores[score_begin:])
        checkpoint=output/f'{args.worker}_stage{stage}.pt';core.save_cl_state(model,str(checkpoint))
        if observer:observer.close()
        del model,result;gc.collect();torch.cuda.empty_cache()
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
        model=create();core.load_cl_state(model,str(checkpoint));model.eval();capture(model)
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):restored=model(**batch,use_cache=False)
        assert torch.equal(expected,restored.logits.cpu())
        payload[f'{stage}/roundtrip']=dict(logits=restored.logits.cpu(),state=snapshot(model))
        del restored
    qwen_core.fit_layer_gmms=original_fit_layer
    with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
        cached=model(input_ids=batch.input_ids,attention_mask=batch.attention_mask,use_cache=True)
        cache=cached.past_key_values
        for step in range(2):
            cached=model(input_ids=torch.tensor([[42+step]],device='cuda'),attention_mask=torch.ones(1,batch.input_ids.shape[1]+step+1,device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True)
            payload[f'cache/{step}']=dict(logits=cached.logits.cpu(),kv=[(c.keys.cpu(),c.values.cpu()) for c in cache.layers],state=snapshot(model))
    payload['synthetic_probe_stats']=fit_stats
    payload['synthetic_probe_seed']=config.seed
    torch.save(payload,output/f'{args.worker}.pt')

if __name__=='__main__':main()
