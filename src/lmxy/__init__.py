from importlib import import_module
from typing import TYPE_CHECKING

from ._parsers import no_think
from ._queue import MulticastQueue
from ._types import (
    LlmFunction,
    LlmResponse,
    SparseEncode,
    Tokenize,
    VectorStore,
)

if TYPE_CHECKING:
    from ._responses import unpack_response
    from .fastembed import get_sparse_encoder
    from .openailike import OpenAiLike
    from .qdrant import QdrantVectorStore
    from .tokenizer import get_tokenizer
else:
    _exports = {
        '._responses': ['unpack_response'],
        '.fastembed': ['get_sparse_encoder'],
        '.openailike': ['OpenAiLike'],
        '.qdrant': ['QdrantVectorStore'],
        '.tokenizer': ['get_tokenizer'],
    }
    _submodule_by_name = {
        name: modname for modname, names in _exports.items() for name in names
    }

    def __getattr__(name: str):
        if mod := _submodule_by_name.get(name):
            mod = import_module(mod, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        msg = f'No attribute {name}'
        raise AttributeError(msg)

    def __dir__() -> list[str]:
        return __all__


__all__ = [
    'LlmFunction',
    'LlmResponse',
    'MulticastQueue',
    'OpenAiLike',
    'QdrantVectorStore',
    'SparseEncode',
    'Tokenize',
    'VectorStore',
    'get_sparse_encoder',
    'get_tokenizer',
    'no_think',
    'unpack_response',
]
