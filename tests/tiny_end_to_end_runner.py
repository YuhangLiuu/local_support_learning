"""Standalone CPU integration worker: full repository imports, real EM, no downloads.

Baseline is imported from byte-for-byte git-show files in an isolated process.
Only synthetic tokenization/data and small configuration are test-owned.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "off", "basic", "extended"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314)
    args = parser.parse_args()
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path.insert(0, str(ROOT))
    if args.mode == "baseline":
        directory = out/"baseline_source";directory.mkdir(exist_ok=True)
        for name in ("modeling_qwen2.py", "configuration_qwen2.py", "cl_memory.py", "ddp.py"):
            data = subprocess.check_output(["git", "show", f"baseline-lsl-original:{name}"], cwd=ROOT)
            (directory/name).write_bytes(data)
        sys.path.insert(0, str(directory))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    import cl_memory as memory
    import modeling_qwen2 as core
    from configuration_qwen2 import Qwen2Config
    from types import SimpleNamespace

    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    def seed():
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    seed()
    config = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                  attention_dropout=.1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                  use_cache=False, attn_implementation="eager")
    fit_config = SimpleNamespace(gmm_fit_num_tokens=128, seq_len_train=16, gmm_skip_on_fail=False)
    kwargs = dict(pos_kwargs=dict(n_components=2,covariance_type="diag",random_state=0),
                  neg_kwargs=dict(n_components=2,covariance_type="diag",random_state=0))
    tok = Tokenizer(WordLevel({f"t{i}":i for i in range(256)},unk_token="t255"))
    tok.pre_tokenizer = Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token="t0",unk_token="t255",padding_side="left")
    def texts(task):
        rng=np.random.default_rng(1000+task)
        low,high = (130,230) if task==9 else ((10,90) if task==0 else (50,130))
        return [" ".join(f"t{t}" for t in [1,2,*rng.integers(low,high,size=10+i%5)]) for i in range(16)]
    def encode(lines):
        enc=tokenizer(lines,return_tensors="pt",padding=True,add_special_tokens=False)
        return dict(input_ids=enc.input_ids,attention_mask=enc.attention_mask,
                    labels=enc.input_ids.masked_fill(~enc.attention_mask.bool(),-100))
    def loader(lines):
        batches=[dict(text=lines[i:i+4]) for i in range(0,len(lines),4)]
        return iter((b,i==len(batches)-1) for i,b in enumerate(batches))
    def create():
        model=core.Qwen2ForCausalLM(Qwen2Config(**config)).float().cpu()
        core.apply_conditional_lora_wrapper(model,is_cl_lora=True,jl_k=8,jl_seed=17,
                                            cl_memory_type="gmm",smoothing_alpha=.3)
        model.model._shared_prefix_ids=torch.tensor([1,2])
        return model
    model=create()
    base_state=copy.deepcopy(model.state_dict())
    counts=dict(model_forward=0,backbone_forward=0,fit_pos=0,fit_neg=0,predict=0)
    handles=[]
    events=[]
    for key,method in (("fit_pos",memory.GMMClassifierGPU.fit_pos),("fit_neg",memory.GMMClassifierGPU.fit_neg),
                       ("predict",memory.GMMClassifierGPU.predict)):
        def counted(self,*a,_key=key,_method=method,**k):
            counts[_key]+=1
            return _method(self,*a,**k)
        setattr(memory.GMMClassifierGPU,key,counted)
    def wrapped(m):
        return [(n,c) for n,c in m.named_modules() if isinstance(c,core.ConditionalLoRALinear)]
    def owners(m):
        return [(n,c) for n,c in wrapped(m) if c.is_cl_memory_owner]
    def track(m):
        for key,module in (("model_forward",m),("backbone_forward",m.model)):
            def hook(*unused,_key=key):
                counts[_key]+=1
            handles.append(module.register_forward_hook(hook))
        for name,c in wrapped(m):
            raw,smooth=c.is_in_cl_memory,c._temporal_smoothing_filter
            def gate(i,x,_method=raw,_name=name):
                result=_method(i,x)
                events.append((_name,"raw",i,result.clone()))
                return result
            def filter_gate(mask,i,pad,_method=smooth,_name=name):
                result=_method(mask,i,pad)
                events.append((_name,"smooth",i,result.clone()))
                return result
            c.is_in_cl_memory=gate;c._temporal_smoothing_filter=filter_gate
    track(model)
    obs=None
    if args.mode in {"basic","extended"}:
        from diagnostics.collector import attach_diagnostics
        from diagnostics.config import DiagnosticsConfig
        obs=attach_diagnostics(model,DiagnosticsConfig(enabled=True,level=args.mode,every_n_forwards=1,
            max_batches_per_context=32,max_tokens_per_batch=256,save_token_traces=True,max_trace_rows=1000),
            out/"diagnostics","tiny-e2e")
    def context(stage,mode):
        if obs:
            obs.set_context(stage=stage,epoch=0,dataset=f"synthetic{stage}",mode=mode)
    snapshots={}
    stages=[]
    probe_batch=encode(texts(0)[:2])
    reference_rows=[]
    cosine_reference=None
    likelihood_differences=[]
    stage_paths=[]
    def state(m):
        result={"parameters":{n:p.detach().clone() for n,p in m.named_parameters()},"gmm":{},"ema":{},
                "rng":(torch.get_rng_state().clone(),np.random.get_state(),random.getstate())}
        for name,c in wrapped(m):
            for i,value in c._smoothing_state.items():
                result["ema"][(name,i)]=value.clone()
            for i,clf in enumerate(c.cl_gmm):
                for which,params in clf.params_vectorized.items():
                    result["gmm"][(name,i,which)]=None if params is None else tuple(t.clone() for t in params)
                for which in ("pos","neg"):
                    result["gmm"][(name,i,which,"real")]=memory.gmm_state(getattr(clf,f"{which}_gmm"))
        return result
    def assert_same(a,b):
        if isinstance(a,torch.Tensor):
            assert torch.equal(a,b)
        elif isinstance(a,np.ndarray):
            np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a: assert_same(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):assert_same(x,y)
        else:
            assert a==b
    def checkpoint(m,stage,expected):
        path=out/f"cl_stage{stage}.pt";core.save_cl_state(m,str(path));stage_paths.append(path)
        # Exact frozen base; CL State does not contain optimizer/RNG/EMA.
        rng=torch.get_rng_state().clone()
        restored=create();restored.load_state_dict(base_state);core.load_cl_state(restored,str(path));restored.eval()
        torch.set_rng_state(rng)
        track(restored)
        before=len(events)
        with torch.no_grad():actual=restored(**probe_batch,use_cache=False)
        assert torch.equal(expected.logits,actual.logits) and torch.equal(expected.loss,actual.loss)
        snapshots[f"stage{stage}/roundtrip"]={"logits":actual.logits,"loss":actual.loss,"state":state(restored),
                                              "gates":events[before:]}
        # Original scoring implementation and Pomegranate's own likelihood path.
        for (name,a),(_,b) in zip(owners(m),owners(restored)):
            for ca,cb in zip(a.cl_gmm,b.cl_gmm):
                z=torch.linspace(-1,1,40).reshape(5,8)
                assert torch.equal(ca.predict(z),cb.predict(z))
                for which in ("pos","neg"):
                    assert_same(memory.gmm_state(getattr(ca,f"{which}_gmm")),
                                memory.gmm_state(getattr(cb,f"{which}_gmm")))
                    vector=memory._diag_gmm_logprob(z,*cb.params_vectorized[which])
                    library=getattr(cb,f"{which}_gmm").log_probability(z)
                    delta=(vector-library).abs()
                    likelihood_differences.append(dict(stage=stage,module=name,which=which,
                        max_abs=float(delta.max()),max_rel=float((delta/torch.maximum(vector.abs(),library.abs()).clamp_min(1e-12)).max())))
                    torch.testing.assert_close(vector,library,atol=1e-3,rtol=1e-5)
        return restored
    for stage in range(2):
        previous={n:p.detach().clone() for n,p in model.named_parameters()}
        core.add_lora(model,r=4,alpha=8,gmm_kwargs=kwargs)
        assert len(wrapped(model))==14 and len(owners(model))==8
        trainable={n for n,p in model.named_parameters() if p.requires_grad}
        assert len(trainable)==28 and all(f"lora_A.{stage}" in n or f"lora_B.{stage}" in n for n in trainable)
        if stage==0:
            context(stage,"fit_neg")
            assert memory.fit_cl_memory(model,loader(texts(9)),tokenizer,fit_config,which="neg")==1.
            snapshots["stage0/negative"]=state(model)
        else:
            assert all(c.cl_gmm[1].neg_gmm is c.cl_gmm[0].neg_gmm for _,c in owners(model))
            assert all(torch.equal(c.cl_gmm[1].params_vectorized["neg"][j],c.cl_gmm[0].params_vectorized["neg"][j])
                       for _,c in owners(model) for j in range(3))
        context(stage,"train");model.train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,betas=(.9,.95),weight_decay=.1)
        losses=[]
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            data=encode(texts(stage)[step*4:step*4+4])
            begin=len(events)
            output=model(**data,use_cache=False);output.loss.backward()
            assert torch.isfinite(output.loss)
            gradients={n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}
            optimizer.step()
            for n,p in model.named_parameters():
                if n in previous:
                    assert torch.equal(p,previous[n]) and not p.requires_grad and p.grad is None
            snapshots[f"stage{stage}/step{step}"]={"logits":output.logits.detach(),"loss":output.loss.detach(),
                "gradients":gradients,"state":state(model),"gates":events[begin:]}
            losses.append(float(output.loss.detach()))
        assert any(not torch.equal(model.get_parameter(n),v) for n,v in snapshots[f"stage{stage}/step0"]["state"]["parameters"].items() if n in trainable)
        optimizer.zero_grad(set_to_none=True)
        context(stage,"fit_pos")
        assert memory.fit_cl_memory(model,loader(texts(stage)),tokenizer,fit_config,which="pos")==1.
        for _,c in owners(model):
            assert all(clf.is_fitted for clf in c.cl_gmm) and not any(c.is_disabled)
            for clf in c.cl_gmm:
                for params in clf.params_vectorized.values():
                    assert all(torch.isfinite(t).all() for t in params)
        snapshots[f"stage{stage}/positive"]=state(model)
        context(stage,"eval");model.eval()
        before=len(events)
        # Independent LoRA sum oracle for actual q-proj, using captured decisions.
        q=model.model.layers[0].self_attn.q_proj
        captured={}
        def capture(module,inputs,output):
            captured["x"]=inputs[0].detach().clone();captured["output"]=output.detach().clone()
        handle=q.register_forward_hook(capture)
        with torch.no_grad():evaluation=model(**probe_batch,use_cache=False)
        handle.remove()
        applicable={i:mask for name,kind,i,mask in events[before:] if name=="model.layers.0.self_attn.q_proj" and kind=="smooth"}
        with torch.no_grad():
            expected=q.base(captured["x"])
            for i in range(stage+1):
                update=q.scalings[i]*F.linear(F.linear(captured["x"]*applicable[i].unsqueeze(-1),q.lora_A[i]),q.lora_B[i])
                expected=expected+update
            assert torch.equal(expected,captured["output"])
        snapshots[f"stage{stage}/eval"]={"logits":evaluation.logits,"loss":evaluation.loss,
                                        "state":state(model),"gates":events[before:]}
        # Independent reference computed from the same forward tensors, no extra model call.
        keep=memory.drop_shared_prefix_mask(probe_batch["input_ids"],probe_batch["attention_mask"],torch.tensor([1,2]))
        raw_map={i:mask.reshape(keep.shape).bool() for name,kind,i,mask in events[before:]
                 if name=="model.layers.0.self_attn.q_proj" and kind=="raw"}
        for i in range(stage+1):
            update=q.scalings[i]*F.linear(F.linear(captured["x"]*applicable[i].unsqueeze(-1),q.lora_A[i]),q.lora_B[i])
            base=q.base(captured["x"])
            z=(captured["x"].reshape(-1,64)@q.jl_P).float()
            pos=memory._diag_gmm_logprob(z,*q.cl_gmm[i].params_vectorized["pos"])
            neg=memory._diag_gmm_logprob(z,*q.cl_gmm[i].params_vectorized["neg"])
            reference_rows.append(dict(stage=stage,adapter=i,count=int(keep.sum()),activation=int((applicable[i]&keep).sum()),
                raw=int((raw_map[i]&keep).sum()),margin=float((pos-neg).reshape(keep.shape)[keep].mean()),
                update_norm=float(update.detach().norm(dim=-1)[keep].mean())))
        if stage==1:
            with torch.no_grad():
                updates=[(q.scalings[i]*F.linear(F.linear(captured["x"]*applicable[i].unsqueeze(-1),
                          q.lora_A[i]),q.lora_B[i]))[keep].numpy() for i in range(2)]
            norms=np.linalg.norm(updates[0],axis=-1)*np.linalg.norm(updates[1],axis=-1)
            nonzero=norms>1e-12
            cos=(updates[0]*updates[1]).sum(-1)[nonzero]/norms[nonzero]
            cosine_reference=dict(numerator=float(cos.sum()),denominator=len(cos))
        restored=checkpoint(model,stage,evaluation)
        stages.append(dict(stage=stage,losses=losses,eval_loss=float(evaluation.loss),active_owners=8,adapters=stage+1))
        # Continue stage 1 from the restored CL State (a fresh optimizer is created).
        if obs:
            obs.close()
        model=restored
        if obs:
            obs=attach_diagnostics(model,obs.config,out/"diagnostics","tiny-e2e")
    # Actual HF DynamicCache with unpadded prefill avoids the upstream full-mask/decode alignment limitation.
    context(1,"cache");model.eval()
    data=encode([" ".join(f"t{i}" for i in range(1,17))])
    with torch.no_grad():
        prefill=model(input_ids=data["input_ids"],attention_mask=data["attention_mask"],use_cache=True)
        cache=prefill.past_key_values
        snapshots["cache/prefill"]={"logits":prefill.logits,"ema":state(model)["ema"],
            "kv":[(layer.keys.clone(),layer.values.clone()) for layer in cache.layers]}
        for step in range(2):
            decode=model(input_ids=torch.tensor([[32+step]]),attention_mask=torch.ones(1,17+step,dtype=torch.long),
                         past_key_values=cache,use_cache=True)
            snapshots[f"cache/decode{step}"]={"logits":decode.logits,"ema":state(model)["ema"],
                "kv":[(layer.keys.clone(),layer.values.clone()) for layer in cache.layers]}
    if obs:
        obs.close()
    formal_counts=counts.copy()
    if args.mode in {"basic","extended"}:
        # P2 stays independent: reload each saved stage, fixed probe, real ProbeRunner/history.
        from diagnostics.probe_runner import ProbeRunner
        from diagnostics.drift_metrics import DriftTracker
        tracker=DriftTracker()
        probe_snapshots={}
        for stage,path in enumerate(stage_paths):
            m=create();m.load_state_dict(base_state);core.load_cl_state(m,str(path));m.eval()
            track(m)  # Count independent probe work separately from inline work.
            observer=attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="probe",max_tokens_per_batch=256),
                                        out/"probe","tiny-e2e")
            runner=ProbeRunner(m);runner.tracker=tracker
            probe_state=state(m)
            result=runner.run("fixed",["sample0","sample1"],probe_batch,{"tokenizer":"synthetic-wordlevel-v1"},stage,0)
            probe_snapshots[stage]=result["snapshots"]["full"]
            assert_same(probe_state,state(m))
            assert not m.training and all(p.grad is None for p in m.parameters())
            observer.close();tracker.save(out/"probe_history.npz")
            tracker=DriftTracker();tracker.load(out/"probe_history.npz")
        torch.save(probe_snapshots,out/"probe_snapshots.pt")
    torch.save(snapshots,out/"snapshots.pt")
    summary=dict(mode=args.mode,seed=args.seed,model=config,jl_k=8,lora_rank=4,components=2,
                 gmm_fit_budget=128,stages=stages,counts=formal_counts,
                 independent_probe_counts={k:counts[k]-formal_counts[k] for k in counts},references=reference_rows,
                 likelihood_library_differences=likelihood_differences,
                 cosine_reference=cosine_reference,
                 source=str(Path(core.__file__).resolve()),cpu_threads=1)
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(dict(mode=args.mode,stages=stages,counts=formal_counts)))


if __name__ == "__main__":
    main()
