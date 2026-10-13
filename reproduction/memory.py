"""Analytical lower bounds plus explicitly uncertain activation/workspace ranges."""
def estimate(model,run,stages=3):
    h=model['hidden_size'];i=model['intermediate_size'];l=model['num_hidden_layers'];v=model['vocab_size']
    head=model.get('head_dim',h//model['num_attention_heads']);kv=head*model['num_key_value_heads']
    # Qwen2 biases in q/k/v, two per-layer RMSNorms, final norm, embeddings/head.
    base=v*h*(1 if model.get('tie_word_embeddings') else 2)+l*(2*h*h+2*h*kv+3*h*i+h+2*kv+2*h)+h
    per_adapter=l*run['lora_r']*(9*h+2*kv+3*i)
    jl=l*(3*h+i)*run['jl_k']*4
    gmm=l*4*(stages*run['gmm_n_pos_comps']+run['gmm_n_neg_comps'])*(2*run['jl_k']+1)*4
    base_bytes=base*4  # official load_model converts every base parameter to float32
    adapter_bytes=per_adapter*stages*4
    gradient_bytes=per_adapter*4;adam_bytes=per_adapter*8
    persistent=base_bytes+adapter_bytes+gradient_bytes+adam_bytes+jl+2*gmm
    logits=run['local_batch_size']*run['seq_len_train']*v*2
    fit_hidden=run['gmm_fit_num_tokens']*h*4
    fit_pending=run['gmm_fit_num_tokens']*run['jl_k']*2*4
    # Same scoring chunk default as original is_in_gmm: broadcast [N,K,JL].
    scoring=min(131072,run['local_batch_size']*run['seq_len_train'])*max(run['gmm_n_pos_comps'],run['gmm_n_neg_comps'])*run['jl_k']*4*3
    low=persistent+max(logits*2+scoring,fit_hidden+fit_pending)
    # Rough range, NOT a measured upper bound (attention/KV/allocator/EM workspace vary).
    high=low+max(8*2**30,run['local_batch_size']*run['seq_len_train']*h*l*8)
    saved_jl=l*(6*h+i)*run['jl_k']*4  # GPU->CPU copies may duplicate sibling projection storage
    return dict(base_parameters=base,adapter_parameters_per_stage=per_adapter,
        base_fp32_gib=base_bytes/2**30,adapter_gib=adapter_bytes/2**30,gradient_gib=gradient_bytes/2**30,
        adam_gib=adam_bytes/2**30,jl_gib=jl/2**30,gmm_model_and_vectors_gib=2*gmm/2**30,
        persistent_lower_bound_gib=persistent/2**30,training_and_fit_floor_gib=low/2**30,
        planning_range_gib=[low/2**30,high/2**30],diagnostics_budget='bounded copies + CPU summaries; GPU overhead must be measured',
        fit_hidden_gib=fit_hidden/2**30,cl_state_last_stage_gib=(adapter_bytes+saved_jl+gmm)/2**30,
        note='analytical estimate; excludes/approximates activation, KV, EM workspace, allocator and I/O peaks; not an OOM guarantee')
