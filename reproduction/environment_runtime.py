"""Environment identity, offline wheel checks and acceptance policy. No installers."""
import hashlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import site
import sys
import sysconfig

ROOT=Path(__file__).resolve().parents[1]
CATALOG=ROOT/'reproduction/environment/profiles.json'
ARTIFACTS=ROOT/'reproduction/environment/artifacts.json'

def packaging_api():
    try:
        from packaging import tags,requirements,utils,markers,version
    except ImportError:
        from pip._vendor.packaging import tags,requirements,utils,markers,version
    return tags,requirements,utils,markers,version

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def catalog_digest():return fingerprint(dict(profiles=sha(CATALOG),artifacts=sha(ARTIFACTS)))

def atomic_json(path,data):
    import tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name);json.dump(data,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary:temporary.unlink(missing_ok=True)

def profile(name):
    data=json.loads(CATALOG.read_text(encoding='utf-8'))
    if name not in data['profiles']:raise ValueError('unknown environment profile')
    return data['profiles'][name]

def check_training_compatibility(skip_lm_eval):
    """Always-on lightweight checks; strict bootstrap provenance is optional."""
    _,requirements,_,_,_=packaging_api()
    managed=Path(sys.prefix)/'bootstrap.json'
    if managed.is_file() and json.loads(managed.read_text())['profile']=='minimal' and not skip_lm_eval:
        raise ValueError('Minimal profile only allows skip_lm_eval=True; restore Full for formal evaluation')
    checks=['transformers==4.56.2','pomegranate==1.1.2','tokenizers>=0.22,<0.23','huggingface-hub>=0.34,<1']
    if not skip_lm_eval:checks.append('lm-eval==0.4.10')
    for text in checks:
        requirement=requirements.Requirement(text)
        try:actual=md.version(requirement.name)
        except md.PackageNotFoundError as error:raise ValueError(f'missing {text}; restore the appropriate LSL environment profile') from error
        if actual not in requirement.specifier:raise ValueError(f'incompatible environment: {text} required, found {actual}')

def protected(name):
    return name in {'torch','torchvision','torchaudio','flash-attn','triton','pytorch-triton-rocm'} or name.startswith('nvidia-')

def probe(image_id=None):
    from reproduction.gpu_environment import gpu_identity
    gpu={}
    for distribution in md.distributions():
        name=distribution.metadata['Name'].lower().replace('_','-')
        if protected(name):gpu[name]=dict(version=distribution.version,origin=str(distribution.locate_file('').resolve()),requires=distribution.requires or [])
    gpu_imports={}
    try:
        import torch
        device=gpu_identity(torch)
        gpu_imports['torch']=str(Path(torch.__file__).resolve())
    except Exception as error:device=dict(backend='unavailable',error=str(error))
    try:gpu_imports['flash-attn']=str(Path(importlib.import_module('flash_attn').__file__).resolve())
    except Exception as error:gpu_imports['flash_attention_error']=str(error)
    _,requirements,_,_,_=packaging_api()
    pending=list(gpu);shared={}
    while pending:
        name=pending.pop()
        if name in shared:continue
        try:distribution=md.distribution(name)
        except md.PackageNotFoundError:continue
        record=dict(version=distribution.version,requires=distribution.requires or [])
        shared[name]=record
        for text in record['requires']:
            requirement=requirements.Requirement(text)
            if not requirement.marker or requirement.marker.evaluate({'extra':''}):pending.append(requirement.name.lower().replace('_','-'))
    supplied_image=image_id or os.environ.get('LSL_BASE_IMAGE_ID')
    release=Path('/etc/os-release')
    observed_image='observed-os-'+hashlib.sha256(release.read_bytes() if release.is_file() else platform.platform().encode()).hexdigest()[:16]
    identity=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',implementation=sys.implementation.name,
        soabi=sysconfig.get_config_var('SOABI'),os=platform.system(),machine=platform.machine(),libc=list(platform.libc_ver()),
        image_id=supplied_image or observed_image,gpu=device,
        gpu_versions={k:v['version'] for k,v in sorted(gpu.items())})
    executable=Path(sys.executable)
    boot=Path('/proc/sys/kernel/random/boot_id')
    # Containers can share the host boot ID; bind to the instance namespace/PID1 too.
    instance=dict(boot_id=boot.read_text().strip() if boot.is_file() else None,hostname=platform.node(),
                  platform_instance=os.environ.get('LSL_INSTANCE_ID'))
    try:
        instance['mount_namespace']=Path('/proc/self/ns/mnt').stat().st_ino
        fields=Path('/proc/1/stat').read_text().rsplit(')',1)[-1].split()
        instance['pid1_start_ticks']=fields[19]
        instance['pid1_cgroup']=Path('/proc/1/cgroup').read_text()
    except OSError:instance['namespace_evidence']='unavailable'
    return dict(identity=identity,executable=str(executable.absolute()),base_executable=str(Path(getattr(sys,'_base_executable',sys.executable)).resolve()),
        executable_sha256=sha(executable) if executable.is_file() else None,python_full=sys.version,
        image_id_source='explicit user/platform label' if supplied_image else 'observed OS-release fingerprint; registry image ID not supplied',base_prefix=sys.base_prefix,prefix=sys.prefix,
        in_venv=sys.prefix!=sys.base_prefix,gpu_packages=gpu,gpu_imports=gpu_imports,gpu_dependencies=shared,system_sites=site.getsitepackages(),
        instance_id=fingerprint(instance),instance_evidence=instance)

def check_base(state):
    identity=state['identity']
    if identity['os']!='Linux':raise ValueError('GPU environment bootstrap requires Linux; Windows supports dry-run/check only')
    if identity['python'] not in {'3.11','3.12'} or identity['implementation']!='cpython':raise ValueError('supported ABI: CPython 3.11 / 3.12 only')
    if state['in_venv']:raise ValueError('choose the current instance base Python, not an old/active venv (--python)')
    if not Path(state['executable']).is_file():raise ValueError('base interpreter is missing/broken')
    if identity['gpu']['backend'] not in {'cuda','rocm'} or 'gpu' not in identity['gpu']:raise ValueError('CUDA/ROCm GPU unavailable')
    if 'torch' not in state['gpu_packages'] or 'flash-attn' not in state['gpu_packages']:raise ValueError('verified system PyTorch and FlashAttention are required; bootstrap never installs them')
    if 'gpu_imports' in state and 'flash-attn' not in state['gpu_imports']:raise ValueError('system FlashAttention cannot import: '+str(state['gpu_imports']))

def safe_child(root,name):
    root=Path(root).resolve();relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(name) or ':' in str(name):raise ValueError('unsafe wheelhouse path')
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('wheelhouse path escape')
    return path

def wheel_metadata(path):
    import zipfile
    from email.parser import BytesParser
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names)!=1:raise ValueError('wheel METADATA missing/ambiguous')
        data=BytesParser().parsebytes(archive.read(names[0]))
    return dict(name=data['Name'].lower().replace('_','-'),version=data['Version'],requires=data.get_all('Requires-Dist',[]))

def check_dependency_closure(packages,gpu_versions,marker_environment=None):
    _,requirements,_,markers,_=packaging_api()
    available={name:record['version'] for name,record in packages.items()}|gpu_versions
    for name,record in packages.items():
        for text in record.get('requires',[]):
            requirement=requirements.Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}|(marker_environment or {})):continue
            if requirement.url:raise ValueError('unlocked direct URL dependency: '+text)
            dep=requirement.name.lower().replace('_','-')
            if dep not in available:raise ValueError(f'unlocked dependency: {name} requires {text}')
            if not requirement.specifier.contains(available[dep],prereleases=True):raise ValueError(f'incompatible dependency: {name} requires {text}, found {available[dep]}')

def verify_wheelhouse(directory,state,name):
    directory=Path(directory)
    if not (directory/'manifest.json').is_file():raise ValueError(f'matching Wheelhouse not prepared: {directory}; use --prepare explicitly; no network fallback')
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema_version')!=1 or manifest['profile']!=name or manifest['identity']!=state['identity']:
        raise ValueError('Wheelhouse mismatch: profile/Python ABI/OS/architecture/GPU stack/image changed')
    if manifest['catalog_sha256']!=catalog_digest():raise ValueError('Wheelhouse application lock changed; prepare a matching wheelhouse')
    tags,_,utils,_,_=packaging_api();supported=set(tags.sys_tags());packages={}
    pins=profile(name)['pins']
    trusted=json.loads(ARTIFACTS.read_text(encoding='utf-8'))['packages']
    for item in manifest['wheels']:
        path=safe_child(directory,item['file'])
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError(f'wheel integrity failure: {item["file"]}')
        parsed_name,ver,_,wheel_tags=utils.parse_wheel_filename(path.name)
        if not supported.intersection(wheel_tags):raise ValueError(f'incompatible wheel ABI: {path.name}')
        record=wheel_metadata(path)
        if record['name']!=parsed_name or record['version']!=str(ver) or pins.get(record['name'])!=record['version'] or record['name'] in packages:
            raise ValueError('wheel identity/version mismatch or duplicate')
        if protected(record['name']):raise ValueError('GPU wheels must never be installed by this bootstrap')
        reference=trusted.get(record['name'],{})
        if reference.get('version')!=record['version']:raise ValueError('missing trusted artifact version evidence')
        candidates=reference['artifacts']
        official=any(a['filename']==path.name and a['sha256']==item['sha256'] and a['size']==item['bytes'] for a in candidates)
        if not official:
            built=[a for a in manifest.get('built_from_sources',[]) if a['package']==record['name']]
            sources={a['sha256'] for a in candidates if a['packagetype']=='sdist'}
            expected_tools={p:pins[p] for p in ('pip','setuptools','wheel','packaging') if p in pins}
            if not built or any(a['source_sha256'] not in sources or a['build_tools']!=expected_tools for a in built):
                raise ValueError('wheel lacks pinned official artifact/source-build provenance')
        packages[record['name']]=record
    if set(packages)!=set(pins):raise ValueError('incomplete application wheelhouse')
    check_dependency_closure(state.get('gpu_dependencies',{})|packages,state['identity']['gpu_versions'])
    for item in manifest.get('nltk_files',[]):
        path=safe_child(directory,item['file'])
        if not path.is_file() or sha(path)!=item['sha256']:raise ValueError('NLTK resource integrity failure')
    if name=='full' and not manifest.get('nltk_files'):raise ValueError('Full wheelhouse lacks NLTK resources')
    return manifest

def check_application_paths(name,env_root):
    env_root=Path(env_root).resolve();pins=profile(name)['pins'];paths={}
    for package,expected in pins.items():
        distribution=md.distribution(package)
        origin=distribution.locate_file('').resolve()
        if distribution.version!=expected:raise ValueError(f'{package}=={expected} required, found {distribution.version}')
        if not origin.is_relative_to(env_root):raise ValueError(f'system application package leaked into environment: {package}: {origin}')
        paths[package]=str(origin)
    critical=[('transformers','transformers'),('tokenizers','tokenizers'),('huggingface_hub','huggingface-hub'),('pomegranate','pomegranate')]
    if name=='full':critical.append(('lm_eval','lm-eval'))
    for module,package in critical:
        obj=importlib.import_module(module)
        if not Path(obj.__file__).resolve().is_relative_to(env_root):raise ValueError(f'wrong actual import path: {module}')
        if getattr(obj,'__version__',pins[package])!=pins[package]:raise ValueError(f'wrong actual imported version: {module}')
    return paths
