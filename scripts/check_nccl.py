"""Explicit target-host NCCL smoke; communication only, not model DDP parity."""
import argparse
import json
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.dry_run:print(json.dumps(dict(output=args.output,nccl='Pending',scope='communication only')));return
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():p.error('NCCL test requires target CUDA GPUs')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    try:
        dist.init_process_group('nccl')
        value=torch.tensor([dist.get_rank()+1.],device=f'cuda:{rank}')
        dist.all_reduce(value)
        assert value.item()==dist.get_world_size()*(dist.get_world_size()+1)/2
        if dist.get_rank()==0:
            Path(args.output).write_text(json.dumps(dict(passed=True,world_size=dist.get_world_size(),
                torch=torch.__version__,scope='NCCL all_reduce only; LSL multi-rank gate/GMM parity remains Pending')),encoding='utf-8')
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
