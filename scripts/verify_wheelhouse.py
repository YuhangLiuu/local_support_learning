"""Read-only verification of the wheelhouse matching this base interpreter."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reproduction.environment_runtime import probe,verify_wheelhouse
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True)
    p.add_argument('--profile',choices=['minimal','full'],required=True);p.add_argument('--image-id');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.dry_run:print(json.dumps(vars(args)));return
    try:manifest=verify_wheelhouse(args.directory,probe(args.image_id),args.profile)
    except Exception as error:p.exit(2,f'WHEELHOUSE FAIL: {error}\n')
    print(json.dumps(dict(passed=True,profile=args.profile,wheels=len(manifest['wheels']))))
if __name__=='__main__':main()
