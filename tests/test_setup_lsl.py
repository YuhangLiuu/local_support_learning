"""CPU synthetic installer tests. Never install packages or claim GPU coverage."""
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest
from reproduction import setup_environment as setup


def test_profiles():
    minimal = setup.requirements('minimal')
    full = setup.requirements('full')
    assert 'lm-eval' not in minimal
    assert full['lm-eval'] == '0.4.10'
    assert minimal['transformers'] == '4.56.2'
    assert minimal['tokenizers'] == '0.22.2'
    assert minimal['huggingface-hub'] == '0.36.2'
    assert minimal['pomegranate'] == '1.1.2'
    assert not any(setup.gpu_component(name) for name in full)


@pytest.mark.parametrize('package,installed', [('transformers','5.0.0'),('tokenizers','0.21.0'),('pomegranate',None),('lm-eval',None)])
def test_repairs(package, installed):
    required = setup.requirements('full' if package == 'lm-eval' else 'minimal')
    actual = dict(required)
    if installed is None:
        actual.pop(package)
    else:
        actual[package] = installed
    assert setup.pending(required, actual) == {package: required[package]}


@pytest.mark.parametrize('name', ['torch','TORCH','nvidia-cublas-cu12','triton','flash_attn','pytorch-triton-rocm','rocm-core','cuda-runtime','hip-runtime'])
def test_protects_gpu(name):
    with pytest.raises(ValueError, match='GPU core'):
        setup.checked_plan({'install':[{'metadata':{'name':name,'version':'99'}}]})


def test_no_pip_when_satisfied(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, 'inventory', lambda *args: setup.requirements('minimal'))
    monkeypatch.setattr(setup.subprocess, 'run', Mock(return_value=Mock(returncode=0)))
    installer = Mock()
    monkeypatch.setattr(setup, 'run', installer)
    setup.install_missing('python','minimal',{},tmp_path)
    installer.assert_not_called()


@pytest.mark.parametrize('profile', ['minimal','full'])
def test_resolve_then_application_only_install(monkeypatch, tmp_path, profile):
    monkeypatch.setattr(setup, 'inventory', lambda *args: {k: setup.profile('minimal')['pins'][k] for k in ('setuptools','wheel')})
    commands = []
    def execute(command, env):
        commands.append(command)
        if '--dry-run' in command:
            (tmp_path/'pip-plan.json').write_text(json.dumps({'install':[{'metadata':{'name':'pomegranate','version':'1.1.2'}}]}))
    monkeypatch.setattr(setup, 'run', execute)
    setup.install_missing('python',profile,{},tmp_path,'https://pypi.org/simple')
    assert '--dry-run' in commands[0]
    assert '--no-deps' in commands[1] and '--ignore-installed' in commands[1]
    assert commands[1][-1] == 'pomegranate==1.1.2'
    assert ('lm-eval==0.4.10' in commands[0]) == (profile == 'full')


def test_unsafe_plan_never_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, 'inventory', lambda *args: {k: setup.profile('minimal')['pins'][k] for k in ('setuptools','wheel')})
    calls = []
    def execute(command, env):
        calls.append(command)
        (tmp_path/'pip-plan.json').write_text(json.dumps({'install':[{'metadata':{'name':'torch','version':'99'}}]}))
    monkeypatch.setattr(setup, 'run', execute)
    with pytest.raises(ValueError, match='GPU core'):
        setup.install_missing('python','minimal',{},tmp_path)
    assert len(calls) == 1


def test_network_failure_propagates(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, 'inventory', lambda *args: {k: setup.profile('minimal')['pins'][k] for k in ('setuptools','wheel')})
    monkeypatch.setattr(setup, 'run', Mock(side_effect=subprocess.CalledProcessError(1, 'pip')))
    with pytest.raises(subprocess.CalledProcessError):
        setup.install_missing('python','minimal',{},tmp_path)


def test_stale_or_broken_venv(tmp_path, monkeypatch):
    binding = {'base':{'instance_id':'new','executable':'base-python','python_full':'test-build'}}
    (tmp_path/'bootstrap.json').write_text(json.dumps(binding))
    assert not setup.reusable(tmp_path,binding)
    (tmp_path/'bin').mkdir()
    (tmp_path/'bin/python').write_text('synthetic')
    monkeypatch.setattr(setup.subprocess,'check_output',lambda *args,**kwargs: json.dumps([str(tmp_path),'base-python','test-build']))
    assert setup.reusable(tmp_path,binding)
    assert not setup.reusable(tmp_path,{'base':{'instance_id':'old'}})


def test_wrong_interpreter_rejected(tmp_path, monkeypatch):
    (tmp_path/'bin').mkdir()
    (tmp_path/'bin/python').write_text('synthetic')
    binding = {'base':{'executable':'right-python','python_full':'test-build'}}
    (tmp_path/'bootstrap.json').write_text(json.dumps(binding))
    monkeypatch.setattr(setup.subprocess,'check_output',lambda *args,**kwargs: json.dumps([str(tmp_path),'wrong-python','test-build']))
    assert not setup.reusable(tmp_path,binding)


def test_failure_never_pass(monkeypatch,capsys):
    monkeypatch.setattr(setup.sys,'argv',['setup_lsl','minimal'])
    monkeypatch.setattr(setup,'probe',Mock(side_effect=ValueError('first critical error')))
    assert setup.main() == 2
    captured = capsys.readouterr()
    assert 'LSL ENVIRONMENT PASS' not in captured.out
    assert 'first critical error' in captured.err


def test_sanitize_without_disabling_cache(monkeypatch):
    monkeypatch.setenv('PIP_TARGET','wrong')
    monkeypatch.setenv('PYTHONPATH','wrong')
    monkeypatch.setenv('PIP_CACHE_DIR','mycache')
    result = setup.child_env()
    assert 'PIP_TARGET' not in result and 'PYTHONPATH' not in result
    assert result['PIP_CACHE_DIR'] == 'mycache'


def test_real_inventory_uses_selected_distribution():
    result = setup.inventory(setup.sys.executable, setup.child_env())
    assert 'pip' in result


@pytest.mark.parametrize('bad_tokenizers', [False,True])
def test_fresh_process_resolves_isolated_packages_over_system_5(tmp_path,bad_tokenizers):
    """Real importlib/path selection using synthetic distributions, no pip install."""
    required = {k: setup.requirements('minimal')[k] for k in
                ('transformers','tokenizers','huggingface-hub','pomegranate')}
    local, system = tmp_path/'local', tmp_path/'system'
    for directory in (local,system):
        for package, version in required.items():
            if directory == system and package == 'transformers':version = '5.0.0'
            if directory == local and package == 'tokenizers' and bad_tokenizers:version = '0.21.0'
            module = package.replace('-','_')
            (directory/module).mkdir(parents=True)
            (directory/module/'__init__.py').write_text(f'__version__ = {version!r}\n')
            info = directory/f'{module}-{version}.dist-info'
            info.mkdir()
            (info/'METADATA').write_text(f'Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n')
            (info/'RECORD').write_text(f'{module}/__init__.py,,\n')
    code = ('import sys; sys.path[:0]=' + repr([str(local),str(system),str(setup.ROOT)]) +
            '; from reproduction import setup_environment as s; s.requirements=lambda name:' + repr(required) +
            '; paths=s.application_check("minimal"); import transformers; assert transformers.__version__=="4.56.2"; print(paths["transformers:import"])')
    result = subprocess.run([setup.sys.executable,'-I','-c',code],env=setup.child_env(),text=True,capture_output=True)
    if bad_tokenizers:
        assert result.returncode != 0 and 'tokenizers==0.22.2 required' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert str(local) in result.stdout


@pytest.mark.parametrize('backend', ['cuda','rocm'])
@pytest.mark.parametrize('profile', ['minimal','full'])
def test_bootstrap_orchestration_reuse_rebuild_and_failure(monkeypatch,tmp_path,capsys,backend,profile):
    root = tmp_path/'environments'
    state = {'identity':{'gpu':{'backend':backend}},'executable':setup.sys.executable,'python_full':setup.sys.version}
    monkeypatch.setattr(setup.sys,'argv',['setup_lsl',profile,'--root',str(root),'--cache-root',str(tmp_path/'cache')])
    monkeypatch.setattr(setup,'probe',lambda:state)
    monkeypatch.setattr(setup,'check_base',lambda state:None)
    commands = []
    def execute(command,env):
        command = list(map(str,command))
        commands.append(command)
        if 'venv' in command:
            target = Path(command[-1]); (target/'bin').mkdir(parents=True)
            (target/'bin/python').write_text('synthetic')
        elif '--output' in command:
            setup.atomic_json(command[command.index('--output')+1],{'passed':True})
    monkeypatch.setattr(setup,'run',execute)
    monkeypatch.setattr(setup,'install_missing',lambda *args:None)
    monkeypatch.setattr(setup.subprocess,'check_output',lambda command,**kwargs: json.dumps([
        str(Path(command[0]).parent.parent),setup.sys.executable,setup.sys.version]))
    assert setup.main() == 0
    assert any('check_environment.py' in ' '.join(c) and '-I' in c for c in commands)
    assert setup.main() == 0
    assert sum('venv' in c for c in commands) == 1
    pointer = json.loads((root/f'current-pip-{profile}.json').read_text())
    Path(pointer['python']).unlink()  # Broken previous-instance interpreter.
    assert setup.main() == 0
    assert sum('venv' in c for c in commands) == 2
    capsys.readouterr()
    monkeypatch.setattr(setup,'install_missing',Mock(side_effect=RuntimeError('Pip installation failed')))
    assert setup.main() == 2
    pointer = json.loads((root/f'current-pip-{profile}.json').read_text())
    assert not json.loads(Path(pointer['report']).read_text())['passed']
    assert 'LSL ENVIRONMENT PASS' not in capsys.readouterr().out
