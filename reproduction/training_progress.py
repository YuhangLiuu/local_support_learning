"""Read-only logging adapters around existing training boundaries."""
from functools import wraps
import inspect
import json
import math
from pathlib import Path
import time


def bounded_lm_loader(loader, batch_size):
    """Explicit hardware batching only; leave prompts/tasks/generation/metrics intact."""
    def load():
        cls, evaluate = loader()
        def wrapper(*args, **kwargs):
            kwargs['batch_size'] = batch_size
            return cls(*args, **kwargs)
        def evaluation(*args, **kwargs):
            kwargs['batch_size'] = batch_size
            return evaluate(*args, **kwargs)
        return wrapper, evaluation
    return load


class TrainingProgress:
    def __init__(self, train, root, torch):
        self.train,self.root,self.torch = train,Path(root),torch
        self.stage=-1;self.epoch=-1;self.task=None;self.step=0;self.total=0
        self.epochs=0;self.steps_per_epoch=0;self.started=time.monotonic()
        self.originals={}

    def emit(self,event,**values):
        record=dict(event=event,stage=self.stage,epoch=self.epoch,dataset=self.task,
                    peak_cuda_gib=self.torch.cuda.max_memory_allocated()/2**30,
                    elapsed_seconds=time.monotonic()-self.started,**values)
        with (self.root/'progress.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(record,allow_nan=False)+'\n')
        if event=='training':
            fraction=values['step']/max(values['total_steps'],1)
            filled=min(20,int(20*fraction));bar='#'*filled+'-'*(20-filled)
            print(f'[LSL] Stage {self.stage+1}/3 {self.task} | Epoch {values["epoch_progress"]} | '
                  f'[{bar}] Step {values["progress"]} ({fraction:.1%}) | Loss {values["loss"]:.6f} | '
                  f'CUDA peak {record["peak_cuda_gib"]:.2f} GiB',flush=True)
        else:print('[LSL] '+json.dumps(record,ensure_ascii=False),flush=True)

    def replace(self,name,wrapper):
        old=getattr(self.train,name);self.originals[name]=old
        setattr(self.train,name,wrapper(old))

    def install(self):
        def main(old):
            signature=inspect.signature(old)
            @wraps(old)
            def wrapped(*args,**kwargs):
                bound=signature.bind(*args,**kwargs).arguments
                config=bound['config'];self.task=bound['tasks']['current']
                self.stage=config.task_idx;self.epochs=config.ft_max_num_epochs;self.epoch=-1;self.step=0
                size=bound['dataloaders'][self.task][2]
                self.steps_per_epoch=math.ceil(size/(config.ft_global_batch_size/self.train.world_size))
                self.total=self.steps_per_epoch*self.epochs
                self.emit('stage_start',stage_label=f'{self.stage+1}/3',total_steps=self.total)
                result=old(*args,**kwargs)
                self.emit('stage_complete',step=self.step,total_steps=self.total)
                return result
            return wrapped
        def context(old):
            @wraps(old)
            def wrapped(model,dataset,stage,epoch,mode='train'):
                self.stage,self.task,self.epoch=stage,dataset,epoch
                return old(model,dataset,stage,epoch,mode)
            return wrapped
        def log(old):
            @wraps(old)
            def wrapped(values,*args,**kwargs):
                result=old(values,*args,**kwargs)
                if 'Train/train_loss' in values:
                    self.step+=1
                    self.emit('training',step=self.step,total_steps=self.total,
                              progress=f'{self.step}/{self.total}',epoch_progress=f'{self.epoch+1}/{self.epochs}',
                              loss=float(values['Train/train_loss']),lr=float(values['Train/lr']))
                elif any(key.startswith('gmm') for key in values):
                    self.emit('gmm_statistics',metrics={key:float(value) for key,value in values.items() if key.startswith('gmm')})
                return result
            return wrapped
        def fit(old):
            @wraps(old)
            def wrapped(*args,**kwargs):
                which=kwargs.get('which','pos');self.emit('gmm_start',which=which)
                result=old(*args,**kwargs)
                self.emit('gmm_complete',which=which,active_owner_ratio=float(result))
                return result
            return wrapped
        def evaluation(old):
            @wraps(old)
            def wrapped(*args,**kwargs):
                task=kwargs.get('dataset_name','lm_eval');self.emit('evaluation_start',evaluation_task=task)
                result=old(*args,**kwargs)
                self.emit('evaluation_complete',evaluation_task=task)
                return result
            return wrapped
        def checkpoint(old):
            @wraps(old)
            def wrapped(model,config,task,epoch):
                result=old(model,config,task,epoch)
                self.emit('checkpoint',path=str(self.root/'experiment/checkpoints'/f'cl_state_stage{config.task_idx}_{task}_epoch{epoch}.pt'))
                return result
            return wrapped
        for name,wrapper in [('main',main),('stage_context',context),('wandb_log',log),
                             ('fit_cl_memory',fit),('eval_dataset',evaluation),('lm_eval_harness',evaluation),
                             ('save_diagnostic_checkpoint',checkpoint)]:self.replace(name,wrapper)
        return self

    def close(self):
        for name,original in self.originals.items():setattr(self.train,name,original)
        self.originals.clear()
