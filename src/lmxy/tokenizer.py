__all__ = ['get_tokenizer']

import os
from functools import partial
from typing import Protocol, TypedDict, Unpack, cast, runtime_checkable

import tiktoken
from glow import memoize
from transformers import AutoTokenizer, PreTrainedTokenizer

from ._types import Tokenize


class _TokenizerKwds(TypedDict, total=False):
    use_fast: bool  # default: true
    cache_dir: str | None  # default: use TRANSFORMERS_CACHE
    local_files_only: bool  # default: false
    trust_remote_code: bool  # default: false


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, *args, **kwargs) -> list: ...


@memoize(5, policy='lru')
def get_tokenizer(
    model_name: str,
    **kwargs: Unpack[_TokenizerKwds],
) -> Tokenize:
    """Load tokenizer.

    See: llama_index.core.utils.get_tokenizer
    """
    should_revert = False
    if 'TIKTOKEN_CACHE_DIR' not in os.environ:
        # set tokenizer cache temporarily
        should_revert = True
        os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '_static/tiktoken_cache',
        )

    try:
        enc = tiktoken.encoding_for_model(model_name)

    # Model is unknown for tiktoken. Use fallback
    except KeyError:
        tokenizer = get_tf_tokenizer(model_name, **kwargs)
        if isinstance(tokenizer, Tokenizer):
            return tokenizer.encode
        return cast('Tokenize', tokenizer)

    else:
        return partial(enc.encode, allowed_special='all')

    finally:
        if should_revert:
            del os.environ['TIKTOKEN_CACHE_DIR']


def get_tf_tokenizer(
    name: str,
    **kwargs: Unpack[_TokenizerKwds],
) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(name, **kwargs)
