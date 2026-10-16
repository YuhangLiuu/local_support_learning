"""Default online, application-only recovery. GPU stack remains owned by the image."""
import argparse
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from reproduction.environment_runtime import ROOT, profile, probe, check_base, atomic_json, fingerprint, protected


def requirements(name):
    catalog = profile(name)
    return {key: catalog['pins'][key] for key in catalog['roots']
            if key not in {'pip', 'setuptools', 'wheel', 'matplotlib'}}


def pending(required, installed):
    return {key: value for key, value in required.items() if installed.get(key) != value}


def gpu_component(name):
    name = name.lower().replace('_', '-')
    return protected(name) or name.startswith(('cuda-', 'rocm-', 'hip-', 'amd-', 'pytorch-triton', 'triton-', 'flash-attn'))


def checked_plan(report):
    entries = report.get('install', [])
    for entry in entries:
        name = entry['metadata']['name']
        if gpu_component(name):
            raise ValueError(f'Pip plan would replace/install GPU core component: {name}; use a compatible base image')
    return [f"{item['metadata']['name']}=={item['metadata']['version']}" for item in entries]


def application_check(name):
    required = requirements(name)
    paths = {}
    for key, expected in required.items():
        dist = md.distribution(key)
        if dist.version != expected:
            raise ValueError(f'{key}=={expected} required, found {dist.version}')
        paths[key] = str(dist.locate_file('').resolve())
    modules = [('transformers', 'transformers'), ('tokenizers', 'tokenizers'),
               ('huggingface_hub', 'huggingface-hub'), ('pomegranate', 'pomegranate')]
    if name == 'full':
        modules.append(('lm_eval', 'lm-eval'))
    for module, package in modules:
        obj = importlib.import_module(module)
        origin = Path(obj.__file__).resolve()
        dist = md.distribution(package)
        # Check the actual module, not only dist-info (e.g. PYTHONPATH shadowing).
        files = {Path(dist.locate_file(file)).resolve() for file in dist.files or []}
        if origin not in files:
            raise ValueError(f'{module} actual import is outside its selected distribution: {origin}')
        if getattr(obj, '__version__', required[package]) != required[package]:
            raise ValueError(f'{module} actual imported version mismatch')
        paths[module + ':import'] = str(origin)
    # Check the selected application's transitive closure; unrelated image packages
    # are deliberately excluded (a global pip check would reject harmless image tools).
    from reproduction.environment_runtime import packaging_api
    _, req_api, utils, _, _ = packaging_api()
    Requirement, canonicalize_name = req_api.Requirement, utils.canonicalize_name
    visited = set()
    def visit(package, extras=frozenset()):
        key = (canonicalize_name(package), extras)
        if key in visited:
            return
        visited.add(key)
        dist = md.distribution(package)
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not any(req.marker.evaluate({'extra': extra}) for extra in (extras or {''})):
                continue
            version = md.version(req.name)
            if req.url or (req.specifier and not req.specifier.contains(version, prereleases=True)):
                raise ValueError(f'incompatible dependency of {package}: {raw}; installed {version}')
            visit(req.name, frozenset(req.extras))
    for package in required:
        visit(package)
    return paths


def child_env():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER',
                'PIP_NO_DEPS', 'PIP_IGNORE_INSTALLED', 'PIP_REQUIRE_VIRTUALENV', 'PIP_DRY_RUN'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    # Prevent a stale pip.conf target/prefix from escaping the isolated venv.
    env['PIP_CONFIG_FILE'] = os.devnull
    env.pop('PIP_NO_INDEX', None)
    return env


def run(command, env):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def inventory(python, env):
    code = 'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}))'
    # distribution iteration can contain a system duplicate; md.version selects the
    # first on sys.path, so explicitly resolve each name through md.version.
    code = code.replace('d.version for d', 'm.version(d.metadata["Name"]) for d')
    return json.loads(subprocess.check_output([str(python), '-I', '-c', code], env=env, text=True))


def reusable(root, binding):
    try:
        if not (root / 'bin/python').is_file() or json.loads((root / 'bootstrap.json').read_text()) != binding:
            return False
        code = 'import sys,json; print(json.dumps([sys.prefix,sys._base_executable,sys.version]))'
        prefix, base, version = json.loads(subprocess.check_output([str(root / 'bin/python'), '-I', '-c', code],
                                                                  env=child_env(), text=True))
        return (Path(prefix).resolve() == root.resolve() and
                Path(base).resolve() == Path(binding['base']['executable']).resolve() and
                version == binding['base']['python_full'])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False


def install_missing(python, name, env, directory, index_url=None):
    required = requirements(name)
    installed = inventory(python, env)
    missing = pending(required, installed)
    if not missing:
        code = 'import sys; sys.path.insert(0,' + repr(str(ROOT)) + '); from reproduction.setup_environment import application_check; application_check(' + repr(name) + ')'
        result = subprocess.run([str(python), '-I', '-c', code], env=env)
        if result.returncode == 0:
            print('Application versions/imports/dependencies correct; skipping Pip installation.', flush=True)
            return
        print('Application import/dependency check failed; resolving repair plan.', flush=True)
    print('Missing/incompatible applications: ' + ', '.join(f'{k}=={v}' for k, v in missing.items()), flush=True)
    for key, expected in missing.items():
        print(f'  {key}: {installed.get(key, "MISSING")} -> {expected}', flush=True)
    # Resolve ALL roots together so correcting Tokenizers cannot break Transformers.
    # Constraints pin the application closure, but never ask Pip to install GPU packages.
    constraints = directory / 'application-constraints.txt'
    constraints.write_text(''.join(f'{k}=={v}\n' for k, v in profile(name)['pins'].items()
                                   if not gpu_component(k)), encoding='utf-8')
    report = directory / 'pip-plan.json'
    if report.exists():
        report.unlink()
    index = ['--index-url', index_url] if index_url else []
    pip = [str(python), '-I', '-m', 'pip', '--disable-pip-version-check']
    # Source-only application packages (e.g. langdetect) may need these. Install
    # wheel-only build tools first, with their own guarded plan, never an implicit
    # build-isolation environment that could download an unreviewed GPU stack.
    tools = {k: profile(name)['pins'][k] for k in ('setuptools', 'wheel')}
    absent_tools = pending(tools, inventory(python, env))
    if absent_tools:
        tools_report = directory / 'pip-build-tools-plan.json'
        run(pip + ['install', '--dry-run', '--only-binary=:all:', '--report', str(tools_report)] + index +
            [f'{k}=={v}' for k, v in absent_tools.items()], env)
        tool_plan = checked_plan(json.loads(tools_report.read_text()))
        if tool_plan:
            run(pip + ['install', '--no-deps', '--ignore-installed', '--only-binary=:all:'] + index + tool_plan, env)
    run(pip + ['install', '--no-build-isolation', '--dry-run', '--report', str(report), '-c', str(constraints)] + index +
        [f'{k}=={v}' for k, v in required.items()], env)
    plan = checked_plan(json.loads(report.read_text()))
    print('Approved application-only install plan: ' + ', '.join(plan), flush=True)
    if not plan:
        raise ValueError('Pip produced no repair plan for incompatible applications')
    # Install the exact approved plan without a second dependency resolution. This
    # prevents a changed index or resolver from silently adding a GPU package.
    # Outside-venv system packages are shadowed, never uninstalled.
    run(pip + ['install', '--no-deps', '--ignore-installed', '--no-build-isolation'] + index + plan, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=['minimal', 'full'])
    parser.add_argument('--root', default=os.environ.get('LSL_ENV_ROOT', '/mnt/workspace/lsl_environment'))
    parser.add_argument('--cache-root', default=os.environ.get('LSL_CACHE_ROOT', '/mnt/workspace/lsl_cache'))
    parser.add_argument('--index-url', default=os.environ.get('LSL_PIP_INDEX_URL'))
    args = parser.parse_args()
    try:
        state = probe()
        print('Current instance: ' + json.dumps({key: state[key] for key in
              ('executable', 'python_full', 'identity', 'gpu_imports') if key in state}, indent=2), flush=True)
        check_base(state)
        directory = Path(args.root).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        key = fingerprint({'base': state, 'profile': args.profile, 'requirements': requirements(args.profile)})[:20]
        target = directory / ('pip-' + args.profile + '-' + key)
        report = directory / 'reports' / (target.name + '.json')
        binding = dict(strategy='pip', base=state, profile=args.profile, acceptance_report=str(report))
        pointer = directory / ('current-pip-' + args.profile + '.json')
        if pointer.is_file():
            previous = Path(json.loads(pointer.read_text())['python']).parent.parent
            previous_binding = dict(binding, acceptance_report=str(directory / 'reports' / (previous.name + '.json')))
            if previous.parent == directory and reusable(previous, previous_binding):
                target, binding = previous, previous_binding
                report = Path(binding['acceptance_report'])
        env = child_env()
        if not reusable(target, binding):
            if target.exists():
                print('Old/broken environment detected; preserving it and creating a fresh venv.', flush=True)
                target = directory / (target.name + '-' + uuid.uuid4().hex[:8])
                report = directory / 'reports' / (target.name + '.json')
                binding['acceptance_report'] = str(report)
            run([sys.executable, '-I', '-m', 'venv', '--system-site-packages', target], env)
            atomic_json(target / 'bootstrap.json', binding)
        # Invalidate any previous receipt before attempting repairs.
        atomic_json(report, dict(passed=False, error='setup/acceptance in progress'))
        python = target / 'bin/python'
        install_missing(python, args.profile, env, target, args.index_url)
        if args.profile == 'full':
            env['NLTK_DATA'] = str(target / 'share/nltk_data')
            code = "import nltk\nfor n,r in [('punkt','tokenizers/punkt'),('punkt_tab','tokenizers/punkt_tab/english')]:\n try: nltk.data.find(r)\n except LookupError: nltk.download(n,download_dir=" + repr(env['NLTK_DATA']) + ",raise_on_error=True)"
            run([python, '-I', '-c', code], env)
        cache = Path(args.cache_root)
        for path in (cache, cache / 'model', cache / 'dataset'):
            path.mkdir(parents=True, exist_ok=True)
        # Fresh process: no pre-install module objects can contaminate acceptance.
        run([python, '-I', ROOT / 'scripts/check_environment.py', '--profile', args.profile,
             '--env-root', target, '--output', report, '--cache-root', cache], env)
        if not json.loads(report.read_text()).get('passed'):
            raise ValueError('acceptance report did not pass')
        atomic_json(directory / ('current-pip-' + args.profile + '.json'), {'python': str(python), 'report': str(report)})
        print(f'Activate: source {target}/bin/activate', flush=True)
        print('LSL ENVIRONMENT PASS', flush=True)
        return 0
    except Exception as error:
        print(f'LSL ENVIRONMENT FAIL: {error}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
