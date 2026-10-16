"""Real train entry import with lm_eval blocked; only missing logging is stubbed."""
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import signal
import sys
import types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
class BlockLM(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='lm_eval' or fullname.startswith('lm_eval.'):raise ModuleNotFoundError('controlled missing lm_eval')
if sys.argv[1]!='protocol':sys.meta_path.insert(0,BlockLM())
if importlib.util.find_spec('wandb') is None:
    logging=types.ModuleType('wandb');logging.__spec__=importlib.machinery.ModuleSpec('wandb',loader=None)
    logging.run=None;sys.modules['wandb']=logging
if not hasattr(signal,'SIGQUIT'):signal.SIGQUIT=signal.SIGTERM
import train_continual as train
from reproduction.configuration import read_plan,resolve
train.config=resolve(read_plan('smoke_1p5b'));train.config.base_model_performance_cache_dir=''
if sys.argv[1]=='protocol':
    import ast,os,subprocess
    from types import SimpleNamespace
    from reproduction.lm_evaluation import load_lm_eval
    from lm_eval.models.huggingface import HFLM
    wrapper,_=load_lm_eval()
    def init(self,**kwargs):self.pretrained=kwargs['pretrained']
    HFLM.__init__=init
    tree=ast.parse(subprocess.check_output(['git','show','baseline-lsl-original:train_continual.py'],cwd=ROOT,text=True,encoding='utf-8'))
    old_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HFWithExternalCache')
    old_function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='lm_eval_harness');old_function.decorator_list=[]
    results=[]
    for index in (0,1):
        calls=[];casts=[]
        class Model:
            config=SimpleNamespace(num_hidden_layers=2,_attn_implementation='eager')
            def eval(self):return self
            def train(self):return self
            def to(self,dtype):casts.append(str(dtype));return self
        def evaluate(**kwargs):
            calls.append({k:v for k,v in kwargs.items() if k!='model'})
            assert kwargs['model'].max_gen_toks==1024 and kwargs['model'].num_layers==2
            task=kwargs['tasks'][0]
            metric,score={'gsm8k':('exact_match,flexible-extract',.5),'humaneval':('pass@1,none',.25),'ifeval':('prompt_level_strict_acc,none',.75)}[task]
            return dict(results={task:{'alias':task,metric:score}},samples={task:[]})
        cfg=resolve(read_plan('continual_7b_lsl'));cfg.activate_cl_method=False;cfg.start_time='test';cfg.base_model_performance_cache_dir=''
        train.config=cfg;train.world_size=1;train.local_rank=0;train.barrier=lambda:None;train.wandb_log=lambda value:None
        train.logger=SimpleNamespace(info=lambda *args:None);train.load_lm_eval=lambda:(wrapper,evaluate)
        namespace=dict(train.__dict__,HFLM=HFLM,simple_evaluate=evaluate)
        exec(compile(ast.Module(body=[old_class,old_function],type_ignores=[]),'original-evaluation','exec'),namespace)
        path=Path(sys.argv[2])/str(index);path.mkdir();os.chdir(path)
        fn=train.lm_eval_harness if index==0 else namespace['lm_eval_harness']
        output=fn(Model(),None,['gsm8k','humaneval','ifeval'],limit=2)
        results.append(dict(output=output,calls=calls,casts=casts))
    assert results[0]==results[1]
    assert results[0]['output'][1]['total']==.5
    print(json.dumps(dict(protocol_equal=True,calls=results[0]['calls'],synthetic_total=.5)))
elif sys.argv[1]=='skipped':
    result=train.lm_eval_harness(None,None,train.config.lm_eval_tasks,return_dummy=True)
    assert result[0]==0 and all(v==0 for v in result[1].values())
    assert not any(name=='lm_eval' or name.startswith('lm_eval.') for name in sys.modules)
    print(json.dumps(dict(imported=True,skipped=True,score=result[0])));
else:
    train.config.skip_lm_eval=False
    try:train.lm_eval_harness(None,None,train.config.lm_eval_tasks)
    except RuntimeError as error:
        assert '0.4.10' in str(error);print(json.dumps(dict(imported=True,error=str(error))))
    else:raise AssertionError('missing dependency not rejected')
