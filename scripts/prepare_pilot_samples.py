"""Materialize/recheck deterministic Pilot indices from already cached datasets; no model."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--cache-root')
    args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    from reproduction.configuration import read_plan,resources,resolve
    from reproduction.pilot_sampling import PilotSampling,SOURCES,validate_plan
    from reproduction.data_access import pinned_loader
    plan=read_plan('continual_1p5b_pilot_seed123');validate_plan(plan,resolve(plan))
    manifest,_=resources(ROOT/'reproduction/resources.lock.json')
    import datasets
    datasets.load_dataset=pinned_loader(manifest,args.cache_root)
    import train_continual as train
    path=Path(args.output).resolve()
    sampler=PilotSampling(train,plan,manifest,path,path if path.exists() else None).install()
    for name in SOURCES:sampler.load(name,123)
    doc=sampler.verify()
    print(json.dumps({k:r['selected_count'] for k,r in doc['datasets'].items()},indent=2))
    print(f'PILOT SAMPLES PASS: {path}')


if __name__=='__main__':main()
