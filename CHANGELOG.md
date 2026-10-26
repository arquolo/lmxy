# Changelog

## 0.8.x

### Qdrant

- Add payload indexes for keyword, integer, and UUID fields through `kw_fields`, `int_fields`, and `uuid_fields`.
- Add `delete_where()` for filtered deletion and support multiple values in `delete_by()`.
- Batch deletion and scrolling operations.
- Limit upsert batches by serialized request size as well as record count.
- Refine the query API, initialization, and LlamaIndex integration.

### Embeddings and reranking

- Automatically adapt requests to different embedding and reranking APIs.
- Add batching to reranking requests.
- Allow `Reranker` to accept plain text lists and return text/score pairs.
- Fix handling of `top_n=0` in reranking and empty input in BERT scoring.

### Response synthesis and infrastructure

- Integrate response synthesis with the LlamaIndex callback manager.
- Update Python support to 3.13 and 3.14 (`>=3.13, <3.15`).
- Update dependencies and fix HTTP retry handling.
- Improve tokenization, repetition handling, and logging.
- Extend the supported FastEmbed version range to `>=0.7.0, <0.10`.

## 0.7.x

Improve Qdrant performance and query composition, fix serialization of complex LlamaIndex nodes, and standardize asynchronous type annotations.

### Qdrant and vector search

- Replace the `_Request` operators `__add__` and `__mul__` with explicit `.hybrid()`, `.hsf()`, and `.fuse()` methods.
- Support three result fusion modes: `hsf`, `rrf`, and `dbsf`.
- Add `_resolve()` for efficient prefetch handling and query batching.
- Set `upsert_batch_size` and `query_batch_size` defaults to 64 instead of `None`.
- Standardize returned collections on `list` instead of `Sequence`.

### LlamaIndex node serialization

- Rewrite `llama_to_record()` and `record_to_llama()`.
- Store metadata, text, and `doc_id` at the top level and serialize binary content through JSON, fixing image serialization.
- Add explicit support for `ImageNode`, `IndexNode`, and `TextNode`.
- Move serialization logic from `_nodes.py` to `_llama.py`.

### Typing and asynchronous APIs

- Replace `AsyncIterator` with `AsyncGenerator` in function signatures.
- Accept `Iterable` instead of `Sequence` in `_types.py`, `_parsers.py`, and `util.py`.
- Extend response type checks in `_responses.py` to support `AsyncIterable`.
- Remove `reasoning_effort` from `OpenAiLike`.

### Utilities and infrastructure

- Temporarily set the tiktoken cache environment variable and handle unknown models through `encoding_name_for_model()`.
- Add a `padding` argument to prompt repacking in `synth.py`.
- Replace standard `logging` with Loguru.

## 0.6.x

Separate the native Qdrant client from LlamaIndex, introduce a unified `Tokens` response type, and add a response synthesizer with custom streaming.

### Tokens and LLM responses

- Introduce the frozen `Tokens` class in `_types.py` to unify strings and asynchronous streams, with support for awaiting and asynchronous iteration.
- Add `get_full_response()` to collect a token stream into a string.
- Make `unpack_response()` return `(Tokens, list[NodeWithScore])`.
- Add `tokens_from_response()` to extract tokens from LlamaIndex responses, including streaming, Pydantic, agent, and synthesizer responses.

### Native Qdrant storage

- Rename the storage class to `Qdrant` and remove its direct dependency on LlamaIndex.
- Simplify the API to `add(records)`, `retrieve(ids)`, `query(dense, sparse, fuse)`, `delete_by()`, `delete()`, and `clear()`.
- Introduce composable `_Request` objects with `+`, `*`, `.limit()`, and `.unit()` for score normalization.
- Remove legacy data formats, manual metadata conversion, and `_relative_score_fusion()`.

### LlamaIndex adapter

- Add `QdrantVectorStore` in `qdrant_llama.py` as an adapter around the native `Qdrant` client.
- Automatically convert between LlamaIndex nodes and queries and internal Qdrant records.

### Response synthesis

- Add asynchronous `synthesize()` to generate answers from retrieved nodes and return `Tokens`.
- Support response modes including `REFINE`, `COMPACT`, `TREE_SUMMARIZE`, and `ACCUMULATE`.
- Add custom streaming and context repacking to work around LlamaIndex issue #21740.

### Utilities and networking

- Add `_async.py` with `Areturn`, `ayield`, and `map_ctx()` for asynchronous generators and context managers.
- Add an optional `timeout` to `get_sparse_encoder()`; when specified, wrap results in a streaming generator. Return `list[SparseEncoding]`.
- Simplify HTTP transport exception handling and avoid unnecessary futures in `raise_for_status()`.
- Fix `min_max()` for empty lists and rename the transport option from `v2` to `aiohttp`.
- Apply embedding text truncation only when `max_chars` is explicitly set and standardize semaphore and embedding types.

## 0.5.x

### Embeddings

- Add `max_chars`, defaulting to 100,000, to limit text size per embedding API request.
- Dynamically size streaming batches through `_get_token_trimmer()`, accounting for the character limit.
- Add `handshake()` and `handshake_sync()` for explicit client initialization. Raise `RuntimeError` if a request is made before initialization.
- Return an empty list for empty text input and change default instructions from `None` to empty strings.

### Qdrant vector store

- Initialize `dense_config` and `sparse_config` with configuration objects instead of `None`, and expose the sparse `modifier` option.
- Store and retrieve multiple embeddings from `node.metadata['embeddings']` separately from `node.embedding`.
- Make `_parse_query()` return `{field_name: (vector, limit, threshold)}`. Apply absolute thresholds only to dense vectors.
- Remove `_is_legacy` and raise an error when a legacy collection configuration is detected.
- Deduplicate nodes explicitly and infer vector size from the first item, including multivector embeddings.

### Utilities and serialization

- Add `eager: bool = False` to `raise_for_status()`. When enabled, raise errors immediately for synchronous responses or closed streams without returning a future.
- Move `min_max()` to `util.py` for hybrid search score normalization.
- Exclude `embeddings` from serialized node metadata to avoid duplicating vectors in the payload.

### CI and dependencies

- Update tag triggers, add a workflow trigger for changes to `pyproject.toml`, and update GitHub Actions versions.
- Remove Twine and publish directly through OIDC.

## 0.4.x

- Add automatic embedding request batching through `Embedder.latency`. Positive values enable batching based on timeout, batch size, and worker availability.
- Rename `max_retries` to `retries` in `OpenAiLike` and `QdrantVectorStore`.
- Support `retries=None` for unlimited retries in `QdrantVectorStore` and clamp negative values to zero.
- Calculate total attempts as `1 + retries`: `retries=3` now means one initial attempt and three retries.
- Refactor embedding methods.

## 0.3.x

### Token streams

- Add `wordify()` to split streams into words and punctuation, `glue_reps()` to join alphanumeric characters and repeated punctuation, and `trim_repetitions_at_end()` to remove repeated trailing sequences.
- Improve `no_think()` compatibility with `AsyncIterable`.

### HTTP and asynchronous infrastructure

- Add `AiohttpTransport` for using aiohttp within httpx.
- Rework retries with Tenacity, including custom predicates and explicit delay and attempt limits.
- Add `MAX_CONNECTIONS` and `KEEP_ALIVE_TIMEOUT` environment settings, defaulting to 100 connections and 15 seconds.

### Embeddings

- Unify query and text requests with automatic instruction selection.
- Add semaphore-based request concurrency.
- Improve retries for HTTP 429 responses and read failures.
- Validate Ollama and OpenAI responses with Pydantic.

### LLM wrapper

- Improve conversion of `ImageBlock` and multiple text blocks to OpenAI messages.
- Rework tool calling with strict schema validation and explicit `tool_choice` settings (`required` and `auto`).
- Simplify parallel tool calls and argument extraction.

### Qdrant vector store

- Move node serialization to `_nodes.py`, optimize payloads, exclude embeddings and relationships, and use `pydantic_core`.
- Allow custom names for dense and sparse vector fields.
- Improve hybrid search score normalization, empty result handling, and dense/sparse query separation.
- Unify add and delete operations in `_ll_update()`.
- Add `qretrieve()` and `qquery()` for direct Qdrant requests.

### Reranking

- Rewrite `Reranker` independently of the LlamaIndex base class, simplify postprocessing, and improve callback manager integration.

## 0.2.x

- Require `hybrid_k > 0`.
- Keep `dense_k` and `sparse_k` independent instead of clamping them to `hybrid_k`.
- Limit `hybrid_k` to `dense_k + sparse_k` to retain more candidates when combining results.
- Return `hybrid_k=-1` when dense or sparse search is skipped (`alpha=0` or `alpha>=1`), disabling downstream hybrid processing and limits.
- Skip `adelete_nodes()` requests when the node ID list is empty.

## 0.1.x
