"""Optional offline diagnosis for a saved server-side GMM activation tensor.

This never changes the production classifier or initialization. It reports
whether the exact saved input reproduces an exception across seeds and compares
the same EM call with a deterministic diverse Gaussian control.
"""
import argparse
import json
from pathlib import Path
import sys


def run(path, components=32, seeds=(0,1,2)):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from cl_memory import GMMClassifierGPU
    saved=torch.load(path,map_location='cpu',weights_only=True)
    values=saved['X'] if isinstance(saved,dict) else saved
    if not isinstance(values,torch.Tensor) or values.ndim!=2 or not values.numel():
        raise ValueError('expected a nonempty rank-2 activation Tensor or {"X": Tensor}')
    values=values.float()
    if not torch.isfinite(values).all():raise ValueError('activation input contains NaN/Inf')
    def classifier(seed):
        kwargs=dict(n_components=components,covariance_type='diag',random_state=seed)
        return GMMClassifierGPU(kwargs,kwargs,device='cpu')
    result=dict(path=str(path),shape=list(values.shape),finite=bool(torch.isfinite(values).all()),runs=[])
    for seed in seeds:
        torch.manual_seed(seed)
        try:
            model=classifier(seed)
            model.fit_neg(values)
            result['runs'].append(dict(seed=seed,status='fit_ok'))
        except Exception as error:
            result['runs'].append(dict(seed=seed,status='failed',error_type=type(error).__name__,error=str(error)))
    generator=torch.Generator().manual_seed(123)
    control=torch.randn(min(max(values.shape[0],components*4),4096),values.shape[1],generator=generator)
    try:
        torch.manual_seed(123)
        model=classifier(123)
        model.fit_neg(control);result['diverse_control']='fit_ok'
    except Exception as error:result['diverse_control']=dict(status='failed',error_type=type(error).__name__,error=str(error))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);parser.add_argument('--components',type=int,default=32)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);args=parser.parse_args()
    if not args.path.is_file():parser.error(f'activation file not found: {args.path}')
    print(json.dumps(run(args.path,args.components,args.seeds),indent=2))


if __name__=='__main__':main()
