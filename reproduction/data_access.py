"""Execute original data/collate helpers on CPU without importing CUDA training."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HELPERS={'is_cybersecurity_ift','is_seceval','is_sft','is_chembench','is_igbo_translation_benchmark',
 'load_dataset_splits','_chembench_render_question','_chembench_fewshot_prefix','_chembench_to_prepped',
 '_chembench_to_sft','_chemdata_to_prepped','_igbo_translation_to_prepped','_cybersecurity_ift_to_prepped',
 '_seceval_is_scorable','_seceval_render_question','_seceval_to_prepped','_seceval_letters','_seceval_f1',
 'SFTCollate','GenerativeCollate','PreppedQACollate'}
CONSTANTS={'PRETRAINING_DATASETS','DATASET_BENCHMARK','QA_DATASET_TAGS','SFT_DATASET_TAGS','PREPPED_QA_COLUMNS',
 'PREPPED_SFT_COLUMNS','DATASET_TEST_SIZE','CHEMDATA_MAX_TRAIN','BLEND_CHEMBENCH_TRAIN','DATASET_MAX_GEN_LENGTH',
 'TRANSLATE_IGBO_TO_ENG','IGBO_TRANSLATION_INSTRUCTION','IGBO_TRANSLATION_MAX_TRAIN','GENERATIVE_EVAL_NUM_SAMPLES',
 'CHEMBENCH_CHOICES','CHEMBENCH_INSTRUCTION','CHEMBENCH_ANSWER_PREFIX','CHEMBENCH_NUM_FEWSHOT',
 'CYBERSECURITY_IFT_MAX_TRAIN','BLEND_SECEVAL_TRAIN','CYBERSECURITY_IFT_MAX_PROMPT_CHARS',
 'CYBERSECURITY_IFT_MAX_ANSWER_CHARS','SECEVAL_CHOICES','SECEVAL_INSTRUCTION'}

def pinned_loader(manifest,cache_root=None):
    from datasets import load_dataset
    from .configuration import local_resource
    entries={r['repo_id']:r for r in manifest['resources'] if r['kind']=='dataset'}
    def load(repo=None,*args,**kwargs):
        if repo is None:repo=kwargs.pop('path')
        if repo not in entries:raise ValueError(f'unpinned dataset: {repo}')
        entry=entries[repo]
        if not entry.get('revision'):raise ValueError(f'missing revision: {repo}')
        kwargs['revision']=entry['revision']
        local=local_resource(entry,cache_root,manifest)
        # A local raw snapshot avoids network fallback and retains original split metadata.
        from .resource_validation import validate_resource
        validate_resource(entry,cache_root,manifest)
        kwargs.pop('revision')
        return load_dataset(str(local),*args,**kwargs)
    return load

def original_data_helpers(manifest,cache_root=None):
    from datasets import DatasetDict,concatenate_datasets
    ns=dict(load_dataset=pinned_loader(manifest,cache_root),DatasetDict=DatasetDict,concatenate_datasets=concatenate_datasets)
    chosen=[]
    for node in ast.parse((ROOT/'train_continual.py').read_text(encoding='utf-8')).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in HELPERS:
            chosen.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in CONSTANTS for t in node.targets):
            chosen.append(node)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),'train_continual.py:data-helpers','exec'),ns)
    return ns
