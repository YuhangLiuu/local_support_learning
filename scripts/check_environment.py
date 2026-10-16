"""Strict profile acceptance. No downloads, installers, model loading or Parity."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reproduction.environment_runtime import probe,profile,check_application_paths,atomic_json,fingerprint,sha,CATALOG,safe_child

def acceptance_hash():
    return fingerprint({name:sha(ROOT/name) for name in ['reproduction/environment_runtime.py','reproduction/lm_evaluation.py','reproduction/environment/profiles.json','reproduction/environment/artifacts.json','scripts/check_environment.py']})

def kernel_checks(torch):
    if not torch.cuda.is_available():raise ValueError('CUDA/ROCm GPU unavailable')
    if not torch.cuda.is_bf16_supported():raise ValueError('LSL requires native BF16 support')
    x=torch.randn(16,16,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    (x@x.T).float().square().mean().backward()
    if not torch.isfinite(x.grad).all():raise ValueError('BF16 backward non-finite')
    from flash_attn import flash_attn_func
    q,k,v=[torch.randn(2,16,4,32,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    output=flash_attn_func(q,k,v,dropout_p=0.0,causal=False)
    output.float().square().mean().backward();torch.cuda.synchronize()
    if not torch.isfinite(output).all() or any(t.grad is None or not torch.isfinite(t.grad).all() for t in (q,k,v)):
        raise ValueError('FlashAttention real kernel forward/backward failed')
    return dict(bf16='PASS',flash_attention_forward_backward='PASS')

def tiny_lsl_check(output):
    """Random CPU Tiny Qwen + real LoRA/Diagnostics; not experiment GPU Parity."""
    import torch
    from configuration_qwen2 import Qwen2Config
    from modeling_qwen2 import Qwen2ForCausalLM,apply_conditional_lora_wrapper,add_lora
    from diagnostics.collector import attach_diagnostics
    from diagnostics.config import DiagnosticsConfig
    model=Qwen2ForCausalLM(Qwen2Config(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,attn_implementation='eager'))
    apply_conditional_lora_wrapper(model,is_cl_lora=True,cl_memory_type='gmm',jl_k=8,jl_seed=0,smoothing_alpha=.1)
    kwargs=dict(pos_kwargs=dict(n_components=2,covariance_type='diag',random_state=0),
                neg_kwargs=dict(n_components=2,covariance_type='diag',random_state=1))
    add_lora(model,r=4,alpha=8,gmm_kwargs=kwargs)
    observer=attach_diagnostics(model,DiagnosticsConfig(enabled=True,every_n_forwards=1,max_batches_per_context=1,max_tokens_per_batch=8),output,'environment-self-test')
    try:
        observer.set_context(stage=0,epoch=0,dataset='synthetic_environment_probe',mode='train')
        ids=torch.tensor([[1,2,3,4,5,6,7,8]])
        result=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False)
        result.loss.backward()
        gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not torch.isfinite(result.loss) or not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Tiny LSL forward/backward failed')
    finally:observer.close()
    return 'PASS random CPU model + real LoRA + Basic Diagnostics; no weights/data downloaded'

def tiny_full_lm_check():
    """Call real HFLM on random CPU weights and an in-memory tokenizer only."""
    import torch
    from tokenizers import Tokenizer,models
    from transformers import PreTrainedTokenizerFast
    from configuration_qwen2 import Qwen2Config
    from modeling_qwen2 import Qwen2ForCausalLM
    from reproduction.lm_evaluation import load_lm_eval
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(
        {'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,**{f't{i}':i for i in range(4,64)}},unk_token='<unk>')),
        pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',unk_token='<unk>')
    model=Qwen2ForCausalLM(Qwen2Config(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,attn_implementation='eager'))
    wrapper,_=load_lm_eval()
    lm=wrapper(pretrained=model,tokenizer=tokenizer,backend='causal',device='cpu',dtype='float32',batch_size=1,max_length=16,add_bos_token=False)
    logits=lm._model_call(torch.tensor([[1,4,5]]))
    if logits.shape!=(1,3,64) or not torch.isfinite(logits).all():raise ValueError('real HFLM model call failed')
    return 'PASS real HFLM call on random CPU model; no benchmark dataset loaded'

def verify_report(path,requires_full=False):
    report=json.loads(Path(path).read_text())
    if not report.get('passed') or report.get('acceptance_hash')!=acceptance_hash():raise ValueError('environment report failed/stale')
    if requires_full and report['profile']!='full':raise ValueError('Full profile required for formal LM Evaluation')
    if Path(report['executable']).absolute()!=Path(sys.executable).absolute():raise ValueError('environment report belongs to a different interpreter')
    current=probe(report['identity']['image_id'])
    if current['identity']!=report['identity'] or current['instance_id']!=report['instance_id']:raise ValueError('GPU/Python/instance changed since environment acceptance')
    if current['gpu_imports']!=report['gpu_imports'] or current['executable_sha256']!=report['interpreter_sha256']:
        raise ValueError('GPU import paths or interpreter changed since acceptance')
    if current['python_full']!=report['python_full']:raise ValueError('Python build changed since acceptance')
    check_application_paths(report['profile'],report['env_root'])
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['minimal','full'],default='minimal');p.add_argument('--output')
    p.add_argument('--probe',action='store_true');p.add_argument('--image-id');p.add_argument('--env-root')
    p.add_argument('--cache-root',default=os.environ.get('LSL_CACHE_ROOT','/mnt/workspace/lsl_cache'))
    args=p.parse_args()
    if args.probe:print(json.dumps(probe(args.image_id)));return
    if not args.output or not args.env_root:p.error('--output and --env-root are required for acceptance')
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_EVALUATE_OFFLINE='1',PYTHONNOUSERSITE='1')
    report=dict(schema_version=1,profile=args.profile,passed=False,checked_at=datetime.now(timezone.utc).isoformat(),
        executable=str(Path(sys.executable).absolute()),env_root=str(Path(args.env_root).resolve()),checks={},acceptance_hash=acceptance_hash())
    report['scope']='environment only; model/data/experiment GPU Parity remain separate'
    report['allowed_experiments']='skip_lm_eval=True only' if args.profile=='minimal' else 'Full evaluations allowed after separate resource and Parity checks'
    # An interrupted check must not leave a previous PASS at the same report path.
    atomic_json(args.output,report)
    try:
        root=Path(args.env_root).resolve();expected=root/'bin/python'
        if not expected.is_file() or Path(sys.executable).absolute()!=expected.absolute() or sys.prefix==sys.base_prefix:
            raise ValueError('unexpected/broken venv interpreter; never accept a prompt label as proof')
        bound=json.loads((root/'bootstrap.json').read_text())
        wheelhouse=Path(bound['wheelhouse'])
        if sha(wheelhouse/'manifest.json')!=bound['wheelhouse_sha256']:raise ValueError('bound wheelhouse manifest changed')
        state=probe(args.image_id);report.update(identity=state['identity'],instance_id=state['instance_id'],gpu_imports=state['gpu_imports'],interpreter_sha256=state['executable_sha256'])
        if bound['profile']!=args.profile or bound['base']['identity']!=state['identity']:raise ValueError('environment ABI/GPU/profile binding mismatch')
        base=bound['base'];base_path=Path(base['executable'])
        report.update(python_full=state['python_full'],image_id_source=base['image_id_source'])
        if state['python_full']!=base['python_full']:raise ValueError('base Python build/version changed')
        if not base_path.is_file() or sha(base_path)!=base['executable_sha256'] or Path(state['base_executable'])!=base_path.resolve():
            raise ValueError('base interpreter link broken or replaced')
        if state['instance_id']!=base['instance_id']:raise ValueError('environment belongs to an old instance')
        report['checks']['application_paths']=check_application_paths(args.profile,root)
        for name,record in base['gpu_packages'].items():
            actual=state['gpu_packages'].get(name)
            if actual!=record:raise ValueError(f'GPU system package changed/shadowed: {name}')
        if state['gpu_imports']!=base['gpu_imports']:raise ValueError('GPU actual import paths changed/shadowed')
        import torch
        import pomegranate
        from modeling_qwen2 import Qwen2ForCausalLM,ConditionalLoRALinear
        from cl_memory import _diag_gmm_logprob
        from diagnostics.collector import attach_diagnostics
        import train_continual
        from cl_memory import GMMClassifierGPU
        # Real small CPU score, with no model weights, data or training.
        value=_diag_gmm_logprob(torch.zeros(2,2),torch.zeros(1),torch.zeros(1,2),torch.ones(1,2))
        if not torch.isfinite(value).all():raise ValueError('LSL GMM score non-finite')
        report['checks']['lsl_imports_and_gmm']='PASS'
        report['checks']['tiny_lsl_cpu']=tiny_lsl_check(Path(args.output).parent/'self-tests'/Path(args.output).stem)
        if args.profile=='full':
            from reproduction.lm_evaluation import load_lm_eval
            wrapper,_=load_lm_eval()
            report['checks']['real_hflm_cpu_call']=tiny_full_lm_check()
            import nltk
            bundle=json.loads((wheelhouse/'manifest.json').read_text())
            for item in bundle['nltk_files']:
                target=safe_child(root/'share',item['file'])
                if not target.is_file() or sha(target)!=item['sha256']:raise ValueError('installed NLTK resource integrity failure')
            report['nltk_data']=str(root/'share/nltk_data')
            for resource in ('tokenizers/punkt','tokenizers/punkt_tab/english'):nltk.data.find(resource)
            report['checks']['lm_eval_and_nltk']='PASS'
        else:report['checks']['lm_eval_and_nltk']='NOT_REQUIRED; Minimal is not formal reproduction ready'
        report['checks'].update(kernel_checks(torch))
        cache=Path(args.cache_root)
        for directory in (cache,cache/'model',cache/'dataset'):
            if not directory.is_dir() or not os.access(directory,os.R_OK|os.W_OK):raise ValueError(f'cache path inaccessible: {directory}')
        report['checks']['cache_paths']='PASS (directory access only; experiment resource validation remains separate)'
        report['passed']=True
    except Exception as error:report['error']=f'{type(error).__name__}: {error}'
    atomic_json(args.output,report)
    print('ENVIRONMENT PASS' if report['passed'] else 'ENVIRONMENT FAIL')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
