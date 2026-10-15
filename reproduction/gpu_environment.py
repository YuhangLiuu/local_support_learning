"""GPU backend metadata only. HIP intentionally shares PyTorch's cuda APIs."""
def backend(torch):
    if getattr(torch.version,'hip',None):return 'rocm'
    if getattr(torch.version,'cuda',None):return 'cuda'
    return 'cpu'

def gpu_identity(torch):
    kind=backend(torch)
    available=torch.cuda.is_available()
    result=dict(backend=kind,cuda=getattr(torch.version,'cuda',None),hip=getattr(torch.version,'hip',None))
    if available:
        props=torch.cuda.get_device_properties(torch.cuda.current_device())
        result.update(gpu=props.name,capability=list(torch.cuda.get_device_capability()),
                      gcn_arch=getattr(props,'gcnArchName',None))
    return result

def inventory(torch):
    result=dict(torch=torch.__version__,**gpu_identity(torch),cuda_runtime=getattr(torch.version,'cuda',None),
                hip_runtime=getattr(torch.version,'hip',None),cuda_available=torch.cuda.is_available(),gpus=[])
    for n in range(torch.cuda.device_count() if result['cuda_available'] else 0):
        props=torch.cuda.get_device_properties(n)
        with torch.cuda.device(n):
            try:bf16=torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:bf16=torch.cuda.is_bf16_supported()
        result['gpus'].append(dict(index=n,name=props.name,memory_gib=props.total_memory/2**30,
            capability=list(torch.cuda.get_device_capability(n)),gcn_arch=getattr(props,'gcnArchName',None),bf16=bf16))
    return result

def platform_commands(kind):
    if kind=='rocm':return [(['rocm-smi'],'rocm_smi'),(['rocminfo'],'rocminfo'),(['hipcc','--version'],'hip_toolkit')]
    if kind=='cuda':return [(['nvidia-smi'],'nvidia_smi'),(['nvidia-smi','topo','-m'],'topology'),(['nvcc','--version'],'cuda_toolkit')]
    return []
