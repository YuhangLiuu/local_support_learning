"""Optional lm-eval dependency; the original wrapper/protocol is unchanged."""
from functools import lru_cache
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path
import sys

def require_lm_eval():
    managed=Path(sys.prefix)/'bootstrap.json'
    if managed.is_file() and json.loads(managed.read_text())['profile']=='minimal':
        raise RuntimeError('LM Evaluation requires Full profile; Minimal only permits skip_lm_eval=True')
    try:installed=version('lm_eval')
    except PackageNotFoundError as error:
        raise RuntimeError('LM Evaluation requires Full profile and lm-eval==0.4.10; restore the Full environment before launch') from error
    if installed!='0.4.10':raise RuntimeError(f'LM Evaluation requires lm-eval==0.4.10, found {installed}')

def require_nltk_resources(directory=None):
    try:
        import nltk
        managed_data=Path(sys.prefix)/'share/nltk_data'
        if managed_data.is_dir() and str(managed_data) not in nltk.data.path:nltk.data.path.insert(0,str(managed_data))
        if directory and str(directory) not in nltk.data.path:nltk.data.path.insert(0,str(directory))
        for resource in ('tokenizers/punkt','tokenizers/punkt_tab/english'):nltk.data.find(resource)
    except (ImportError,LookupError) as error:
        raise RuntimeError('Full evaluation requires NLTK punkt and punkt_tab resources; restore the Full wheelhouse/environment before launch') from error

@lru_cache(maxsize=1)
def load_lm_eval():
    require_lm_eval()
    try:
        from lm_eval.models.huggingface import HFLM
        from lm_eval import simple_evaluate
    except ImportError as error:
        raise RuntimeError(f'Full lm-eval==0.4.10 environment is incomplete: {error}') from error

    class HFWithExternalCache(HFLM):
        @property
        def max_gen_toks(self):
            return 1024

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            pretrained = self.pretrained.module if hasattr(self.pretrained, "module") else self.pretrained
            self.num_layers = pretrained.config.num_hidden_layers
            self.attention_type = pretrained.config._attn_implementation

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            from transformers.cache_utils import DynamicCache
            generation_kwargs["past_key_values"] = DynamicCache()
            return super()._model_generate(context, max_length, stop, **generation_kwargs)

    return HFWithExternalCache,simple_evaluate
