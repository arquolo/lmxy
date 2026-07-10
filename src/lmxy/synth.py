__all__ = ['synthesize']

from collections.abc import AsyncIterator

from glow import span_task
from llama_index.core import (
    BasePromptTemplate,
    PromptHelper,
    get_response_synthesizer,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_PROMPT_SEL,
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import MetadataMode, NodeWithScore
from loguru import logger
from pydantic import ValidationError

from ._responses import get_full_response, tokens_from_response
from ._types import Tokenize


@span_task
async def synthesize(
    *nodes: NodeWithScore,
    query: str,
    llm: LLM,
    tokenizer: Tokenize,
    text_qa_template: BasePromptTemplate = DEFAULT_TEXT_QA_PROMPT_SEL,
    refine_template: BasePromptTemplate = DEFAULT_REFINE_PROMPT_SEL,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    callback_manager: CallbackManager | None = None,
    # = .core.response_synthesizers.refine.DEFAULT_RESPONSE_PADDING_SIZE
    padding: int = 500,
) -> AsyncIterator[str] | str:
    """
    Response synthesizer has different modes to handle nodes:
    - SIMPLE_SUMMARIZE - O(1), truncates input.
        ```
        ctx = truncate(concat(nodes))
        LLM(ctx, query)
        ```
    - REFINE - O(n nodes)
        ```
        ret = LLM(nodes[0], query)
        for node in nodes[1:]:
            subs = repack_to_fit_remaining_ctx(node, initial=ret)
            for sub in subs:
                ret = LLM(sub, query, ret)
        ret
        ```
    - COMPACT - O(n tokens). Same as REFINE but with initial node repacking
    - TREE_SUMMARIZE - O(n log n).
        ```
        nodes = repack_to_fit_ctx(nodes)
        while len(nodes) > 1:
            nodes = [LLM(node, query) for node in nodes]
            nodes = repack_to_fit_ctx(nodes)
        LLM(nodes, query)
        ```
    - ACCUMULATE - concat of local summaries, O(n tokens), O(n nodes)
        ```
        concat(
            LLM(sub, query)
            for node in nodes
            for sub in repack_to_fit_ctx([node])
        )
        ```
    - COMPACT_ACCUMULATE - concat of local summaries, O(n tokens).
        Same as ACCUMULATE but with initial node repacking
    """
    prompt_helper = PromptHelper(
        context_window=llm.metadata.context_window,
        num_output=max(llm.metadata.num_output, 0) or 256,
        tokenizer=tokenizer,
    )
    if response_mode not in (ResponseMode.REFINE, ResponseMode.COMPACT):
        streaming = response_mode not in (
            ResponseMode.ACCUMULATE,
            ResponseMode.COMPACT_ACCUMULATE,
        )
        syn = get_response_synthesizer(
            llm=llm,
            prompt_helper=prompt_helper,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            response_mode=response_mode,
            callback_manager=callback_manager,
            streaming=streaming,
        )
        rsp = await syn.asynthesize(query, list(nodes))
        # text_chunks = [n.get_content(MetadataMode.LLM) for n in nodes]
        # rsp = syn._prepare_response_output(
        #     await syn.aget_response(query, text_chunks),
        #     list(nodes),
        # )
        return tokens_from_response(rsp)

    # See: https://github.com/run-llama/llama_index/issues/21740
    # Repack chunks, loop & stream for last chunk only
    text_qa_q = text_qa_template.partial_format(query_str=query)
    refine_q = refine_template.partial_format(query_str=query)

    max_prompt = max([text_qa_q, refine_q], key=_get_prompt_size)
    text_chunks = [n.get_content(MetadataMode.LLM) for n in nodes]
    if response_mode is ResponseMode.COMPACT:
        text_chunks = prompt_helper.repack(
            max_prompt, text_chunks, padding=padding, llm=llm
        )
    text_chunks = [
        c2
        for c in text_chunks
        for c2 in prompt_helper.repack(
            max_prompt, [c], padding=padding, llm=llm
        )
    ]
    text_chunks = text_chunks[::-1]  # First is last
    rstr: str | None = None
    while text_chunks:
        chunk = text_chunks.pop()

        if rstr is None:  # First chunk
            prompt = text_qa_q
            kwds = {'context_str': chunk}
        else:
            prompt = refine_q.partial_format(existing_answer=rstr)
            chunk, *repacked = prompt_helper.repack(prompt, [chunk], llm=llm)
            text_chunks += reversed(repacked)
            kwds = {'context_msg': chunk}

        try:
            agen = await llm.astream(prompt, **kwds)
            if not text_chunks:  # Last chunk, do stream
                return agen
            if answer := await get_full_response(agen):
                rstr = answer
        except (ValidationError, ValueError, TypeError) as e:
            logger.warning(f'LLM response error: {e}', exc_info=True)

    return rstr or ''


def _get_prompt_size(prompt: BasePromptTemplate) -> int:
    all_kwargs = dict.fromkeys(prompt.template_vars, '') | prompt.kwargs
    return len(prompt.format(llm=None, **all_kwargs))
