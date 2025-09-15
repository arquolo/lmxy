__all__ = [
    'aretry',
    'get_clients',
    'raise_for_status',
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

from httpx import (
    AsyncByteStream,
    AsyncClient,
    AsyncHTTPTransport,
    Client,
    HTTPStatusError,
    HTTPTransport,
    Limits,
    Response,
)
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from ._env import env

# --------------------------------- retrying ---------------------------------

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

    # Drop frames until `code` frame is reached
    while tb:
        if tb.tb_frame.f_code is code:
            e.__traceback__ = tb
            return
        tb = tb.tb_next


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
        'Retrying %s #%d in %.2g seconds as it raised %s: %s.',
        qualname,
        s.attempt_number,
        s.next_action.sleep,
        ex.__class__.__name__,
        ex,
    )


_limits = Limits(max_connections=None, max_keepalive_connections=20)

# Use SSL_CERT_FILE envvar to pass `cafile`
_transport = HTTPTransport(
    verify=env.SSL_VERIFY, limits=_limits, retries=env.RETRIES
)
_atransport = AsyncHTTPTransport(
    verify=env.SSL_VERIFY, limits=_limits, retries=env.RETRIES
)


def get_clients(
    base_url: str = '', timeout: float | None = None
) -> tuple[Client, AsyncClient]:
    sc = Client(
        timeout=timeout,
        follow_redirects=True,
        base_url=base_url,
        transport=_transport,
    )
    ac = AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        base_url=base_url,
        transport=_atransport,
    )
    return sc, ac


def raise_for_status(resp: Response) -> asyncio.Future[Response]:
    if resp.is_success:
        f = asyncio.Future[Response]()
        f.set_result(resp)

    # opened asynchronous response
    elif isinstance(resp.stream, AsyncByteStream) and not resp.is_closed:

        async def _fail() -> Response:
            data = await resp.aread()
            exc = _failed_response(resp, data)
            raise exc from None

        f = asyncio.ensure_future(_fail())

    # closed asynchronous response or any synchronous response
    else:
        exc = _failed_response(resp, resp.read())

        f = asyncio.Future[Response]()
        f.set_exception(exc)

    return f


def _failed_response(resp: Response, content: bytes) -> HTTPStatusError:
    status_class = resp.status_code // 100
    error_type = _ERROR_TYPES.get(status_class, 'Invalid status code')
    message = (
        f"{error_type} '{resp.status_code} {resp.reason_phrase}' "
        f"for url '{resp.url}' failed with {content.decode()}"
    )
    return HTTPStatusError(message, request=resp.request, response=resp)


_ERROR_TYPES = {
    1: 'Informational response',
    3: 'Redirect response',
    4: 'Client error',
    5: 'Server error',
}
