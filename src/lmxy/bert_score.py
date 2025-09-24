"""Fork of https://github.com/Tiiiger/bert_score

Performance optimized for transformers>4 & pytorch>2
"""

__all__ = ['score']

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from logging import getLogger
from math import log1p
from typing import NamedTuple

import torch
import transformers
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaTokenizer,
)

logger = getLogger(__name__)
_CACHE_DIR = transformers.utils.hub.TRANSFORMERS_CACHE


class _Embedding(NamedTuple):
    tokens: list[int]
    emb: Tensor


class _IdfEmbedding(NamedTuple):
    emb: Tensor
    idf: Tensor


def score(
    refs_and_hyps: Sequence[tuple[str, str]],
    model_name: str,
    num_layers: int,
    idf: bool = False,
    batch_size: int = 64,
    device: str | None = None,
    fast_tokenize: bool = False,
    verbose: bool = False,
    cache_dir: str | None = _CACHE_DIR,
    local_files_only: bool | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """BERTScore metric.

    Args:
    - refs_and_hyps - N pairs of reference and candidate sentences
    - num_layers - the layer of representation to use
    - idf - use idf weighting
    - batch_size - embedding batch size
    - device - embedding device. Default cuda if available
    - fast_tokenize - `use_fast` parameter passed to HF tokenizer

    Returns tensors of per-sample precisions, recalls & F1 scores. Size N.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = get_tokenizer(
        model_name,
        use_fast=fast_tokenize,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = get_model(
        model_name,
        num_layers,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    main_tq = tqdm(
        desc='processing sentence pairs',
        total=len(refs_and_hyps),
        disable=not verbose,
        smoothing=0,
    )
    # Normalize
    refs_and_hyps = [(ref.strip(), hyp.strip()) for ref, hyp in refs_and_hyps]

    # Dedup and sort for embeddings
    uniq = sorted(
        {s for tup in refs_and_hyps for s in tup},
        key=lambda s: len(s.split(' ')),
        reverse=True,
    )

    # Tokenize and embed
    embs: dict[str, _Embedding] = {}  # every (str) -> (tokens, emb)
    with tqdm(
        desc='embedding...',
        total=len(uniq),
        disable=not verbose,
        smoothing=0,
    ) as tq:
        for ubatch in batched(uniq, batch_size):
            embs |= _embed(ubatch, model, tokenizer, device=device)
            tq.update(len(ubatch))

    # IDFy and flatten
    idf_ = (
        Idf.train([embs[ref] for ref, _ in refs_and_hyps])
        if idf
        else Idf.default(tokenizer)
    )
    iembs = idf_.idfy(embs)
    all_iembs = [(iembs[ref], iembs[hyp]) for ref, hyp in refs_and_hyps]

    # Do cosine greedy matching
    all_ts: list[Tensor] = []
    with tqdm(
        all_iembs, 'greedy matching...', disable=not verbose, smoothing=0
    ) as tq:
        for batch in batched(tq, batch_size):
            all_ts.append(_greedy_cos_idf(batch))
            main_tq.update(len(batch))

    ps, rs = torch.cat(all_ts).unbind(dim=1)
    fs = 2 * ps * rs / (ps + rs)

    # Empty sentence is <bos><eos>, so only 2 tokens
    zs = torch.as_tensor(
        [2 in (e1.emb.shape[0], e2.emb.shape[0]) for e1, e2 in all_iembs],
        dtype=torch.bool,
    )
    if zs.any():
        logger.warning('Empty sentence detected; zeroing BERTscores')
        ps.masked_fill_(zs, 0.0)
        rs.masked_fill_(zs, 0.0)

    fs = fs.masked_fill_(fs.isnan(), 0.0)
    main_tq.close()

    return ps, rs, fs


def get_model(
    name: str,
    num_layers: int,
    /,
    *,
    device: str = 'cpu',
    cache_dir: str | None = _CACHE_DIR,
    local_files_only: bool | None = None,
) -> PreTrainedModel:
    assert num_layers >= 0
    logger.info(
        f'Loading model {name!r} @ layers={num_layers!r} on device={device!r}'
        f' ({cache_dir=}, {local_files_only=})'
    )
    model = AutoModel.from_pretrained(
        name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model.eval()
    model.to(device)

    if hasattr(model, 'decoder') and hasattr(model, 'encoder'):
        model = model.encoder

    if hasattr(model, 'n_layers'):  # xlm
        assert num_layers <= model.n_layers
        model.n_layers = num_layers
        return model

    mod: torch.nn.ModuleList
    if hasattr(model, 'layer'):  # xlnet
        mod = model.layer
    elif hasattr(model, 'encoder'):  # albert, t5, bert, roberta
        enc = model.encoder

        if hasattr(enc, 'albert_layer_groups'):  # albert
            assert num_layers <= enc.config.num_hidden_layers
            enc.config.num_hidden_layers = num_layers
            return model

        # `block` for T5, `layer` for bert/roberta
        mod = enc.block if hasattr(enc, 'block') else enc.layer

    elif hasattr(model, 'transformer'):  # bert, roberta
        mod = model.transformer.layer
    elif hasattr(model, 'layers'):  # bart
        mod = model.layers
    else:
        raise NotImplementedError(model)

    assert isinstance(mod, torch.nn.ModuleList)
    del mod[num_layers:]
    return model


def get_tokenizer(
    name: str,
    use_fast: bool = False,
    cache_dir: str | None = _CACHE_DIR,
    local_files_only: bool | None = None,
) -> PreTrainedTokenizer:
    logger.info(
        f'Loading tokenizer for model {name!r}'
        f' ({cache_dir=}, {local_files_only=})'
    )
    return AutoTokenizer.from_pretrained(
        name,
        use_fast=use_fast,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


@torch.inference_mode()
def _embed(
    sentences: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> dict[str, _Embedding]:
    # Tokenize
    tk_kwds = {
        'add_special_tokens': True,
        'max_length': tokenizer.model_max_length,
        'truncation': True,
    }
    if isinstance(tokenizer, GPT2Tokenizer | RobertaTokenizer):
        tk_kwds['add_prefix_space'] = True

    tokens = [
        (
            tokenizer.encode(s, **tk_kwds)
            if s
            else tokenizer.build_inputs_with_special_tokens([])
        )
        for s in sentences
    ]

    # Embed
    padded = pad_sequence(
        [torch.as_tensor(tks, dtype=torch.long) for tks in tokens],
        batch_first=True,
        padding_value=int(tokenizer.pad_token_id),
    )
    mask = _pad_mask([len(tks) for tks in tokens])

    out = model(
        padded.to(device=device, non_blocking=True),
        attention_mask=mask.to(device=device, non_blocking=True),
        output_hidden_states=False,
    )
    embs = out[0]  # b t d
    embs.div_(embs.norm(dim=-1, keepdim=True))  # L2-normalize each token

    return {
        txt: _Embedding(tokens=tks, emb=emb[: len(tks)])
        for txt, tks, emb in zip(sentences, tokens, embs.cpu().unbind())
    }


@dataclass(frozen=True, slots=True)
class Idf:
    idf: Mapping[int, float]

    @classmethod
    def default(cls, tokenizer: PreTrainedTokenizer) -> 'Idf':
        idf = defaultdict[int, float](lambda: 1.0)
        idf[tokenizer.sep_token_id] = 0
        idf[tokenizer.cls_token_id] = 0
        return cls(idf)

    @classmethod
    def train(cls, embs: Sequence[_Embedding]) -> 'Idf':
        """Map word piece index to its inverse document frequency."""
        idf_count = Counter[int](i for e in embs for i in set(e.tokens))
        w = log1p(len(embs))

        idf = defaultdict[int, float](lambda: w)
        idf.update({idx: w - log1p(c) for (idx, c) in idf_count.items()})
        return cls(idf)

    def idfy(self, embs: Mapping[str, _Embedding]) -> dict[str, _IdfEmbedding]:
        return {
            s: _IdfEmbedding(
                emb=e.emb,
                idf=torch.as_tensor(
                    [self.idf[i] for i in e.tokens], dtype=torch.float
                ),
            )
            for s, e in embs.items()
        }


@torch.inference_mode()
def _greedy_cos_idf(
    refs_and_hyps: Sequence[tuple[_IdfEmbedding, _IdfEmbedding]],
) -> Tensor:
    """Compute greedy matching based on cosine similarity."""

    def prepare(iembs: Sequence[_IdfEmbedding]) -> tuple[Tensor, Tensor]:
        # b k d
        embedding = pad_sequence([t.emb for t in iembs], batch_first=True)

        # b k
        idf = pad_sequence([t.idf for t in iembs], batch_first=True)
        idf.div_(idf.sum(dim=1, keepdim=True))  # unit norm

        return embedding, idf  # masked & zero-padded

    # bid, bi, bi, b
    hyp_embedding, hyp_idf = prepare([hyp for _, hyp in refs_and_hyps])

    # bjd, bj, bj, b
    ref_embedding, ref_idf = prepare([ref for ref, _ in refs_and_hyps])

    sim = torch.einsum('bid,bjd->bij', hyp_embedding, ref_embedding)  # bij
    word_ps = sim.amax(dim=2)  # bi, precision
    word_rs = sim.amax(dim=1)  # bj, recall

    ps = torch.einsum('bi,bi->b', word_ps, hyp_idf)
    rs = torch.einsum('bj,bj->b', word_rs, ref_idf)
    return torch.stack([ps, rs], -1)  # b 2


def _pad_mask(lens: Sequence[int], device: str | None = None) -> Tensor:
    return (  # (k maxlen)
        torch.as_tensor(lens, dtype=torch.long, device=device)[:, None]
        > torch.arange(max(lens), dtype=torch.long, device=device)[None, :]
    )
