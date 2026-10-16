"""Persistent, profile-specific wheelhouse and current-instance venv recovery."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
import zipfile

from .environment_runtime import (ROOT,CATALOG,ARTIFACTS,catalog_digest,sha,fingerprint,atomic_json,profile,probe,check_base,
    safe_child,packaging_api,protected,check_dependency_closure,wheel_metadata,verify_wheelhouse)

def run(command,env=None):
    result=subprocess.run([str(c) for c in command],env=env,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(f'command failed ({result.returncode}): {command}\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}')
    return result.stdout

def offline_env(nltk=None):
    env={k:v for k,v in os.environ.items() if not k.startswith('PIP_') and k not in {'PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'}}
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_EVALUATE_OFFLINE='1')
    if nltk:env['NLTK_DATA']=str(nltk)
    return env

def read_url(url,json_data=True,limit=12*2**20):
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in {'pypi.org','files.pythonhosted.org','api.github.com','raw.githubusercontent.com'}:
        raise ValueError('unapproved artifact/metadata source')
    request=urllib.request.Request(url,headers={'User-Agent':'LSL-environment-bootstrap'})
    with urllib.request.urlopen(request,timeout=60) as stream:
        data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('metadata exceeds size limit')
    return json.loads(data) if json_data else data

def create_plan(state,name,allow_build=False):
    tags,requirements,utils,_,versions=packaging_api();supported=list(tags.sys_tags());rank={tag:i for i,tag in enumerate(supported)}
    pins=profile(name)['pins'];artifacts=[];packages={}
    catalog=json.loads(ARTIFACTS.read_text(encoding='utf-8'))
    for package,version in pins.items():
        data=catalog['packages'][package]
        if data['version']!=version:raise ValueError('artifact lock version mismatch')
        python_version='.'.join(map(str,sys.version_info[:3]))
        if data.get('requires_python') and versions.Version(python_version) not in requirements.Requirement('python'+data['requires_python']).specifier:
            raise ValueError(f'{package} does not support current Python')
        packages[package]=dict(version=version,requires=data.get('requires_dist') or [])
        candidates=[]
        for artifact in data['artifacts']:
            if artifact.get('yanked'):continue
            if artifact['filename'].endswith('.whl'):
                _,_,_,wheel_tags=utils.parse_wheel_filename(artifact['filename'])
                matches=[rank[t] for t in wheel_tags if t in rank]
                if matches:candidates.append((min(matches),artifact))
        if candidates:artifact=min(candidates,key=lambda c:c[0])[1]
        else:
            sources=[a for a in data['artifacts'] if a['packagetype']=='sdist' and not a.get('yanked')]
            if not allow_build or not sources:raise ValueError(f'no matching wheel for {package}=={version}; source build requires --allow-build')
            artifact=sources[0]
        artifacts.append(dict(package=package,version=version,file=artifact['filename'],url=artifact['url'],
            bytes=artifact['size'],sha256=artifact['sha256'],source_build=artifact['packagetype']=='sdist'))
    minor=state['identity']['python']
    check_dependency_closure(state.get('gpu_dependencies',{})|packages,state['identity']['gpu_versions'],
        dict(python_version=minor,python_full_version=minor+'.0',sys_platform='linux',platform_system='Linux',os_name='posix',platform_machine=state['identity']['machine']))
    nltk=catalog['nltk'] if name=='full' else []
    return dict(schema_version=1,profile=name,identity=state['identity'],catalog_sha256=catalog_digest(),artifacts=artifacts,nltk=nltk,
                total_download_bytes=sum(a['bytes'] for a in artifacts+nltk),sources=['official PyPI release artifacts','official revision-pinned NLTK data'])

def download_artifact(item,destination,git_blob=False):
    path=safe_child(destination,item['file']);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        valid=sha(path)==item.get('sha256')
        if git_blob:
            with path.open('rb') as stream:digest=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode()+stream.read()).hexdigest()
            valid=digest==item['git_blob_sha1']
        if valid:return path
    parsed=urllib.parse.urlparse(item['url'])
    if parsed.scheme!='https' or parsed.hostname not in {'files.pythonhosted.org','raw.githubusercontent.com'}:raise ValueError('unapproved download source')
    temporary=safe_child(destination,item['file']+'.partial')
    digest=hashlib.sha1(f'blob {item["bytes"]}\0'.encode()) if git_blob else hashlib.sha256()
    try:
        with urllib.request.urlopen(item['url'],timeout=90) as response,temporary.open('wb') as stream:
            total=0
            while True:
                chunk=response.read(2**20)
                if not chunk:break
                total+=len(chunk)
                if total>item['bytes']:raise ValueError('download exceeds planned size')
                digest.update(chunk);stream.write(chunk)
        if total!=item['bytes'] or digest.hexdigest()!=item.get('git_blob_sha1' if git_blob else 'sha256'):raise ValueError('download content hash/size mismatch')
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)
    return path

def prepare(directory,state,plan,max_gib):
    if plan['identity']!=state['identity'] or plan['catalog_sha256']!=catalog_digest():raise ValueError('download plan is stale/incompatible')
    if plan!=create_plan(state,plan['profile'],allow_build=True):raise ValueError('download plan differs from pinned official artifact metadata')
    if (directory/'manifest.json').is_file():
        try:return verify_wheelhouse(directory,state,plan['profile'])
        except ValueError:pass
    needed=plan['total_download_bytes']
    if needed>max_gib*2**30 or shutil.disk_usage(directory).free<needed*3+5*2**30:raise ValueError('download/build disk budget exceeded; inspect plan before increasing --max-gib')
    wheels=directory/'wheels';wheels.mkdir(exist_ok=True);sources=directory/'sources';sources.mkdir(exist_ok=True)
    built=[];pending=[]
    for item in plan['artifacts']:
        if protected(item['package']):raise ValueError('GPU artifact prohibited')
        path=download_artifact(item,sources if item['source_build'] else wheels)
        if item['source_build']:pending.append((item,path))
    if pending:
        build=directory/('build-'+uuid.uuid4().hex)
        run([sys.executable,'-I','-m','venv','--system-site-packages',build])
        python=build/'bin/python'
        tools=[a for a in plan['artifacts'] if a['package'] in {'pip','setuptools','wheel','packaging'}]
        lock=build/'tools.lock';lock.write_text('\n'.join(f'{a["package"]}=={a["version"]} --hash=sha256:{a["sha256"]}' for a in tools))
        run([python,'-I','-m','pip','install','--no-index','--no-deps','--ignore-installed','--require-hashes','--find-links',wheels,'-r',lock],offline_env())
        for item,path in pending:
            run([python,'-I','-m','pip','wheel','--no-index','--no-deps','--no-build-isolation','--wheel-dir',wheels,path],offline_env())
            built.append(dict(package=item['package'],source_sha256=item['sha256'],build_tools={a['package']:a['version'] for a in tools}))
    records=[];packages={};pins=profile(plan['profile'])['pins']
    for path in sorted(wheels.glob('*.whl')):
        meta=wheel_metadata(path)
        if meta['name'] not in pins or meta['version']!=pins[meta['name']]:raise ValueError('unexpected stale wheel in wheelhouse')
        records.append(dict(file='wheels/'+path.name,bytes=path.stat().st_size,sha256=sha(path),**meta))
        packages[meta['name']]=meta
    if set(packages)!=set(pins):raise ValueError('missing built/downloaded wheels')
    check_dependency_closure(state.get('gpu_dependencies',{})|packages,state['identity']['gpu_versions'])
    nltk_root=directory/'nltk_data';nltk_files=[]
    for item in plan.get('nltk',[]):
        archive=download_artifact(item,sources)
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                target=safe_child(nltk_root/'tokenizers',member.filename)
                if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('NLTK archive symlink prohibited')
                if member.is_dir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with stream.open(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    if plan['profile']=='full':
        for path in sorted(nltk_root.rglob('*')):
            if path.is_file():nltk_files.append(dict(file=path.relative_to(directory).as_posix(),sha256=sha(path)))
    manifest=dict(schema_version=1,profile=plan['profile'],identity=state['identity'],catalog_sha256=catalog_digest(),
                  wheels=records,nltk_files=nltk_files,built_from_sources=built,prepared_at=datetime.now(timezone.utc).isoformat())
    # Acceptance marker is written only after complete construction.
    atomic_json(directory/'manifest.json',manifest)
    verify_wheelhouse(directory,state,plan['profile'])
    return manifest

def install(root,directory,state,name,cache):
    if not (root/'envs').resolve().is_relative_to(root.resolve()):raise ValueError('managed env directory escapes persistent root')
    manifest=verify_wheelhouse(directory,state,name)
    binding=dict(base=state,profile=name,wheelhouse_sha256=sha(directory/'manifest.json'),wheelhouse=str(directory))
    identifier=fingerprint(binding)[:20]
    target=root/'envs'/f'{name}-{identifier}';env=offline_env(directory/'nltk_data')
    if not target.resolve().is_relative_to((root/'envs').resolve()):raise ValueError('managed environment target escapes root')
    report=root/'reports'/f'{name}-{identifier}.json'
    binding['acceptance_report']=str(report)
    pointer=root/f'current-{name}.json'
    if pointer.is_file():
        candidate=Path(json.loads(pointer.read_text())['env'])
        if not candidate.resolve().is_relative_to((root/'envs').resolve()):raise ValueError('current environment pointer escapes managed root')
        if (candidate/'bootstrap.json').is_file() and json.loads((candidate/'bootstrap.json').read_text())==binding:
            target=candidate
    def verify(path,output):
        atomic_json(output,dict(passed=False,profile=name,status='verification_in_progress'))
        text=run([path/'bin/python','-I',ROOT/'scripts/check_environment.py','--profile',name,'--env-root',path,
                  '--image-id',state['identity']['image_id'],'--cache-root',cache,'--output',output],env)
        accepted=json.loads(output.read_text())
        if not accepted.get('passed') or accepted.get('profile')!=name:raise RuntimeError('environment verifier did not produce a PASS report')
        return text
    if (target/'bootstrap.json').is_file() and (target/'bin/python').is_file() and json.loads((target/'bootstrap.json').read_text())==binding:
        try:
            verify(target,report)
            atomic_json(root/f'current-{name}.json',dict(env=str(target),report=str(report)))
            return target,report
        except (RuntimeError,OSError):pass
    if target.exists():target=target.with_name(target.name+'-rebuild-'+uuid.uuid4().hex[:8])
    # Never delete or move an old instance's venv.
    run([sys.executable,'-I','-m','venv','--system-site-packages',target])
    atomic_json(target/'bootstrap.json',binding)
    lock=target/'requirements.lock'
    lock.write_text('\n'.join(f'{a["name"]}=={a["version"]} --hash=sha256:{a["sha256"]}' for a in manifest['wheels'])+'\n')
    try:
        run([target/'bin/python','-I','-m','pip','install','--no-index','--find-links',directory/'wheels','--require-hashes',
             '--no-deps','--ignore-installed','--only-binary=:all:','-r',lock],env)
        if name=='full':shutil.copytree(directory/'nltk_data',target/'share/nltk_data')
        for kind in ('model','dataset'):(cache/kind).mkdir(parents=True,exist_ok=True)
        verify(target,report)
    except Exception as error:
        atomic_json(target/'installation_failed.json',dict(error=str(error)));raise
    atomic_json(root/f'current-{name}.json',dict(env=str(target),report=str(report)))
    return target,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    for flag in ('check','prepare','install','verify'):mode.add_argument('--'+flag,action='store_true')
    p.add_argument('--profile',choices=['minimal','full'],default='minimal')
    p.add_argument('--root',type=Path,default=Path('/mnt/workspace/lsl_environment'));p.add_argument('--image-id')
    p.add_argument('--cache-root',type=Path,default=Path(os.environ.get('LSL_CACHE_ROOT','/mnt/workspace/lsl_cache')))
    p.add_argument('--download',action='store_true',help='explicitly execute an already saved prepare plan')
    p.add_argument('--allow-build',action='store_true',help='allow hashed official sdists built offline with pinned tools')
    p.add_argument('--max-gib',type=float,default=8);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.dry_run:print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2));return
    root=args.root.resolve();report=root/'reports'/('bootstrap-'+uuid.uuid4().hex+'.json')
    try:
        state=probe(args.image_id);check_base(state)
        name=args.profile;directory=safe_child(root,'wheelhouses/'+name+'/'+fingerprint(dict(identity=state['identity'],lock=catalog_digest()))[:20])
        if args.check:
            atomic_json(report,dict(passed=True,scope='base inventory only; application environment not yet accepted',**state))
            print(json.dumps(dict(base_check='PASS',wheelhouse=str(directory),report=str(report)),indent=2));return
        if args.prepare:
            directory.mkdir(parents=True,exist_ok=True);plan_path=directory/'plan.json'
            if not args.download:
                plan=create_plan(state,name,args.allow_build);atomic_json(plan_path,plan)
                print(json.dumps(plan,indent=2));print(f'Plan saved: {plan_path}; inspect capacity/sources/hashes, then repeat with --download');return
            plan=json.loads(plan_path.read_text())
            if plan['profile']!=name:raise ValueError('download plan profile mismatch')
            prepare(directory,state,plan,args.max_gib)
            print(f'Wheelhouse verified: {directory}');return
        if args.install:
            target,acceptance=install(root,directory,state,name,args.cache_root.resolve())
            print(f'ENVIRONMENT PASS\nEnvironment: {target}\nReport: {acceptance}\nUse: source {target}/bin/activate');return
        pointer=json.loads((root/f'current-{name}.json').read_text());target=Path(pointer['env'])
        verify_wheelhouse(directory,state,name)
        atomic_json(pointer['report'],dict(passed=False,profile=name,status='verification_in_progress'))
        output=run([target/'bin/python','-I',ROOT/'scripts/check_environment.py','--profile',name,'--env-root',target,
                    '--image-id',state['identity']['image_id'],'--cache-root',args.cache_root,'--output',pointer['report']],offline_env(directory/'nltk_data'))
        print(output)
    except Exception as error:
        atomic_json(report,dict(passed=False,profile=args.profile,error=f'{type(error).__name__}: {error}'))
        print(f'ENVIRONMENT FAIL\n{error}\nReport: {report}',file=sys.stderr);raise SystemExit(2)

if __name__=='__main__':main()
