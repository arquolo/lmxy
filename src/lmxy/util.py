__all__ = [
    'aretry',
    'is_true',
    'warn_immediate_errors',
]

import asyncio
import logging
import urllib.error
from collections.abc import Callable
from contextlib import suppress
from functools import update_wrapper
from inspect import iscoroutinefunction
from types import CodeType
from typing import Protocol, cast

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

_retriable_errors: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    urllib.error.HTTPError,
)

with suppress(ImportError):
    import requests

    _retriable_errors += (requests.HTTPError,)

with suppress(ImportError):
    import httpx

    _retriable_errors += (httpx.HTTPError,)

with suppress(ImportError):
    import aiohttp

    _retriable_errors += (aiohttp.ClientError,)


logger = logging.getLogger(__name__)
_ENV_VARS_TRUE_VALUES = frozenset({'1', 'ON', 'YES', 'TRUE'})


def is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.upper() in _ENV_VARS_TRUE_VALUES


class _Decorator(Protocol):
    def __call__[**P, R](self, f: Callable[P, R], /) -> Callable[P, R]: ...


def aretry(
    *extra_errors: type[BaseException],
    max_attempts: int = 10,
    wait: float = 1,
    override_defaults: bool = False,
) -> _Decorator:
    """Wrap sync or async function with a new `Retrying` object.

    By default retries only if:
    - asyncio.TimeoutError
    - urllib.error.HTTPError
    - requests.HTTPError
    - httpx.HTTPError
    - aiohttp.ClientError

    To add more add more.
    To disable default errors set `override_defaults`.
    """
    # Protect to not accidentally call aretry(fn)
    assert all(
        isinstance(tp, type) and issubclass(tp, BaseException)
        for tp in extra_errors
    )
    retry_ = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait),
        retry=retry_if_exception_type(
            (() if override_defaults else _retriable_errors) + extra_errors
        ),
        before_sleep=warn_immediate_errors,
        reraise=True,
    )

    def deco[**P, R](f: Callable[P, R]) -> Callable[P, R]:
        wrapped_f = retry_(f)

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
            try:
                return await wrapped_f(*args, **kwargs)  # type: ignore[misc]
            except BaseException as exc:
                _declutter_tb(exc, f.__code__)
                raise

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return wrapped_f(*args, **kwargs)
            except BaseException as exc:
                _declutter_tb(exc, f.__code__)
                raise

        w2 = async_wrapper if iscoroutinefunction(f) else wrapper
        return cast('Callable[P, R]', update_wrapper(w2, f))

    return deco


def _declutter_tb(e: BaseException, code: CodeType) -> None:
    tb = e.__traceback__
    # Drop outer to `code` frames
    while tb and tb.tb_frame.f_code is not code:
        tb = tb.tb_next
    e.__traceback__ = tb


def warn_immediate_errors(s: RetryCallState) -> None:
    if (
        not s.outcome
        or not s.next_action
        or not s.outcome.failed
        or (ex := s.outcome.exception()) is None
    ):
        return

    fn = s.fn
    if fn is None:
        qualname = '<unknown>'
    else:
        name = getattr(fn, '__qualname__', getattr(fn, '__name__', None))
        mod = getattr(fn, '__module__', None)
        qualname = (f'{mod}.{name}' if mod else name) if name else repr(fn)

    logger.warning(
        'Retrying %s in %.2g seconds as it raised %s: %s.',
        qualname,
        s.next_action.sleep,
        ex.__class__.__name__,
        ex,
    )
