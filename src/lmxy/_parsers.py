__all__ = ['no_think']

import re
from collections.abc import AsyncIterator

_THINK_START = '<think>'
_THINK_STOP = '</think>'
_THOUGHT = re.compile(f'{_THINK_START}.*{_THINK_STOP}', flags=re.DOTALL)
_DIRTY_END = re.compile(r' +\n')
_EOL_END = re.compile(r'\n*\Z')
_DUPLICATE_EMPTY_LINES = re.compile(r'\n\n\n+')
_TAG_CUT = {
    tag: re.compile('$|'.join(tag[: i + 1] for i in range(len(tag))) + '$')
    for tag in (_THINK_START, _THINK_STOP)
}
_NEXT_STATE = [  # answering -> (not answering, next tag, next tag part)
    (True, _THINK_START, _TAG_CUT[_THINK_START]),
    (False, _THINK_STOP, _TAG_CUT[_THINK_STOP]),
]


def no_think[S: AsyncIterator[str] | str](s: S) -> S:
    """Remove <think>...</think> blocks and dedent output."""
    if isinstance(s, str):
        txt = _THOUGHT.sub('', s)
        txt = _DIRTY_END.sub('\n', txt)
        txt = _DUPLICATE_EMPTY_LINES.sub('\n\n\n', txt)
        return txt.strip(' \n')  # type: ignore[return-value]

    return _astrip(_ahide_think(s))  # type: ignore[return-value]


async def _ahide_think(s: AsyncIterator[str]) -> AsyncIterator[str]:
    """Remove `<think>...</think>` blocks."""
    buf = ''
    answering, tag, partial_tag = _NEXT_STATE[False]
    async for token in s:
        if not token:  # Skip empty tokens
            continue
        buf += token

        while (pos := buf.find(tag)) != -1:  # Full tag present
            if answering:
                yield buf[:pos]
            buf = buf[pos + len(tag) :]
            answering, tag, partial_tag = _NEXT_STATE[answering]  # Step

        if not partial_tag.search(buf):  # Not even a part of tag
            if answering:
                yield buf
            buf = ''

    if buf and answering:
        yield buf


async def _astrip(s: AsyncIterator[str]) -> AsyncIterator[str]:
    """Strip chunked line.

    - Skips leading and trailing whitespaces and line breaks
    - Trims trailing whitespaces from all lines
    - Keeps at most 2 consecutive empty lines
    """
    buf = ''
    async for tok in s:
        buf += tok
        buf = buf.lstrip('\n ')  # Remove leading spaces and line breaks
        if buf:
            break  # Got initial data

    num_breaks = 0
    async for tok in s:
        buf += tok
        out, buf, num_breaks = _astrip_step(buf, num_breaks)
        if out:
            yield out

    line = buf.rstrip()  # Remove trailing spaces
    if line:
        yield '\n' * num_breaks + line


def _astrip_step(buf: str, num_breaks: int) -> tuple[str, str, int]:
    out = ''
    buf = _DIRTY_END.sub('\n', buf)  # Remove spaces before line break

    if '\n' in buf:
        buf = _DUPLICATE_EMPTY_LINES.sub('\n\n\n', buf)
        if m := _EOL_END.search(buf):  # Find position right before last \n
            pos = m.start()
            num_breaks = len(buf) - pos  # Final breaks at the end
            out, buf = buf[:pos], buf[pos:]

    # Buffer is not full of spaces but ends with them
    if (m := re.search(r' *\Z', buf)) and (pos := m.start()):
        num_breaks = 0  # Not empty chunk, reset counter
        out += buf[:pos]
        buf = buf[pos:]

    return out, buf, num_breaks
