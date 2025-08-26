from importlib import import_module
from typing import TYPE_CHECKING

from ._types import SparseEncode, VectorStore

if TYPE_CHECKING:
    from .fastembed import get_sparse_encoder
    from .qdrant import QdrantVectorStore
else:
    _exports = {
        '.fastembed': ['get_sparse_encoder'],
        '.qdrant': ['QdrantVectorStore'],
    }
    _submodule_by_name = {
        name: modname for modname, names in _exports.items() for name in names
    }

    def __getattr__(name: str):
        if mod := _submodule_by_name.get(name):
            mod = import_module(mod, __package__)
            globals()[name] = obj = getattr(mod, name)
            return obj
        raise AttributeError(f'No attribute {name}')

    def __dir__():
        return __all__


__all__ = [
    'QdrantVectorStore',
    'SparseEncode',
    'VectorStore',
    'get_sparse_encoder',
]
