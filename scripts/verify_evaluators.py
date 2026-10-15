"""CPU synthetic scoring checks using installed lm-eval task implementations."""
import argparse
import json
import os
from pathlib import Path
import sys
import platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--output',default=str(ROOT/'validation/evaluator_audit.json'))
    p.add_argument('--manifest',default=str(ROOT/'reproduction/resources.lock.json'))
    p.add_argument('--cache-root');args=p.parse_args()
    if args.dry_run:print(json.dumps(vars(args),indent=2));return
    if platform.system()!='Linux':p.error('Evaluator resource downloads are server-only; use --dry-run locally')
    from reproduction.configuration import resources,resolve_cache_root
    manifest,_=resources(args.manifest)
    cache=resolve_cache_root(args.cache_root,manifest)
    os.environ['HF_HOME']=str(cache/'hf_runtime')
    os.environ['NLTK_DATA']=str(cache/'nltk_data')
    os.environ['HF_ALLOW_CODE_EVAL']='1'
    import nltk
    nltk.data.path.insert(0,os.environ['NLTK_DATA'])
    for name in ('punkt_tab','punkt'):
        assert nltk.download(name,download_dir=os.environ['NLTK_DATA'],quiet=True)
    import datasets
    from reproduction.data_access import pinned_loader
    datasets.load_dataset=pinned_loader(manifest,args.cache_root)
    from lm_eval.tasks import TaskManager,get_task_dict
    tasks=get_task_dict(['gsm8k','humaneval','ifeval'],task_manager=TaskManager())
    report=dict(status='CPU synthetic scoring only; model generation pending GPU',tasks={})
    for name,task in tasks.items():
        docs=list(task.eval_docs)
        report['tasks'][name]=dict(eval_samples=len(docs),dataset_path=task.config.dataset_path,
                                  output_type=task.config.output_type,metric_list=task.config.metric_list)
    gsm=tasks['gsm8k'];doc=list(gsm.eval_docs)[0]
    target=gsm.doc_to_target(doc)
    correct=gsm.process_results(doc,[target])
    assert any(v==1. for v in correct.values())
    report['gsm8k_perfect_extracted_answer']=correct
    ifeval=tasks['ifeval']
    # Select a real deterministic word-count instruction and provide a synthetic response.
    from lm_eval.tasks.ifeval import utils
    candidate=dict(key=0,prompt='Respond with at least two words.',instruction_id_list=['length_constraints:number_words'],
                   kwargs=[dict(num_words=2,relation='at least')])
    result=utils.process_results(candidate,['hello world'])
    assert result['prompt_level_strict_acc'] is True or result['prompt_level_strict_acc']==1.
    report['ifeval_word_constraint']=result
    human=tasks['humaneval'];doc=list(human.eval_docs)[0]
    # Use published canonical code, not an LLM or untrusted generated program.
    from lm_eval.tasks.humaneval.utils import pass_at_k
    result=pass_at_k([doc['test']+'\ncheck('+doc['entry_point']+')'],[[doc['prompt']+doc['canonical_solution']]],[1])
    report['humaneval_canonical_solution']=result
    assert result['pass@1']==1.
    from importlib.metadata import version
    report['versions']={name:version(name) for name in ('lm_eval','evaluate','nltk','datasets')}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__':main()
