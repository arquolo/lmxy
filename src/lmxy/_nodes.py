__all__ = [
    'metadata_dict_to_node',
    'node_to_metadata_dict',
]

from collections.abc import Mapping

import orjson
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    IndexNode,
    Node,
    TextNode,
)


def node_to_metadata_dict(node: BaseNode, include_id: bool = False) -> dict:
    """Common logic for saving Node data into metadata dict."""
    # See: llama_index.core.vector_stores.utils:node_to_metadata_dict
    # NOTE: original greatly bloats qdrant because of JSON dump below

    # Using mode="json" here because BaseNode may have fields
    # of type bytes (e.g. images in ImageBlock),
    # which would cause serialization issues.
    node_dict = node.model_dump(mode='json')

    # Remove embedding from node_dict
    node_dict.pop('embedding', None)  # ! originally set None

    # Remove relationships, we directly use `doc_id` in metadata
    node_dict.pop('relationships', None)

    # Make metadata the top level
    metadata: dict = node_dict.pop('metadata', {})  # ! originally `get()`

    # Move to top level
    if (text := node_dict.pop('text', None)) is not None:
        metadata['text'] = text

    # store ref doc id at the top level for metadata filtering
    if (doc_id := node.ref_doc_id) is not None:
        metadata = {'doc_id': doc_id} | metadata

    if include_id and (id_ := node_dict.pop('id_', None)) is not None:
        metadata = {'id_': id_} | metadata

    return metadata | {
        # dump remainder of node_dict to json string
        '_node_content': orjson.dumps(node_dict).decode(),
        '_node_type': node.class_name(),
    }


def metadata_dict_to_node(
    metadata: Mapping, with_id: str | int | None = None
) -> BaseNode:
    """Load generic Node from metadata dict."""
    # See: llama_index.core.vector_stores.utils:metadata_dict_to_node
    # ! This one is altered to be compatible with above.
    mut = {**metadata}
    node_json = mut.pop('_node_content', None)
    node_type = mut.pop('_node_type', None)
    if node_json is None:
        msg = 'Node content not found in metadata dict.'
        raise ValueError(msg)

    text = mut.pop('text', None)

    # Discard IDs
    mut.pop('ref_doc_id', None)
    mut.pop('document_id', None)
    # mut.pop('doc_id', None)

    data = orjson.loads(node_json)
    data.setdefault('metadata', mut)
    data.pop('class_name', None)

    if with_id is not None:
        data['id_'] = with_id

    if text is not None:
        data['text'] = text

    # Reconstruct relationships
    if (
        'relationships' not in data
        and (parent_id := mut.pop('doc_id', None)) is not None
    ):
        data['relationships'] = {'1': {'node_id': parent_id}}

    tp = _TYPES.get(node_type, TextNode)
    return tp(**data)


_TYPES = {tp.class_name(): tp for tp in (Node, IndexNode, ImageNode)}
