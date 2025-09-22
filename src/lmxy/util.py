__all__ = [
    'aretry',
    'get_clients',
    'raise_for_status',
    'warn_immediate_errors',
]

import asyncio
import logging
import random
import urllib.error
from collections.abc import Callable
from contextlib import suppress
from datetime import timedelta
from functools import update_wrapper
from inspect import iscoroutinefunction
from types import CodeType
from typing import Any, cast

import tenacity as t
from httpx import (
    AsyncByteStream,
    AsyncClient,
    AsyncHTTPTransport,
    Client,
    HTTPError,
    HTTPStatusError,
    HTTPTransport,
    Limits,
    Response,
)

from ._env import env

# --------------------------------- retrying ---------------------------------

_retriable_errors: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    urllib.error.HTTPError,
    HTTPError,
)

with suppress(ImportError):
    import requests

    _retriable_errors += (requests.HTTPError,)

with suppress(ImportError):
    import aiohttp

    _retriable_errors += (aiohttp.ClientError,)


logger = logging.getLogger(__name__)


class aretry:  # noqa: N801
    """Wrap sync or async function with a new `Retrying` object.

    By default retries only if:
    - asyncio.TimeoutError
    - urllib.error.HTTPError
    - requests.HTTPError
    - httpx.HTTPError
    - aiohttp.ClientError

    To add more add more.
    To disable default errors set `override_defaults`.

    Defaults timeouts are from `stamina.retry`
    """

    def __init__(
        self,
        *extra_errors: type[BaseException],
        max_attempts: int | None = 10,
        timeout: float | timedelta | None = 45.0,
        wait_initial: float | timedelta = 0.1,
        wait_max: float | timedelta = 5.0,
        wait_jitter: float | timedelta = 1.0,
        wait_exp_base: float = 2.0,
        override_defaults: bool = False,
    ) -> None:
        # Protect to not accidentally call aretry(fn)
        assert all(
            isinstance(tp, type) and issubclass(tp, BaseException)
            for tp in extra_errors
        )

        stop = _make_stop(max_attempts, timeout)
        wait = _jittered_backoff_wait(
            initial=wait_initial,
            max_backoff=wait_max,
            jitter=wait_jitter,
            exp_base=wait_exp_base,
        )

        self.wrap = t.retry(
            stop=stop,
            wait=wait,
            retry=t.retry_if_exception_type(
                (() if override_defaults else _retriable_errors) + extra_errors
            ),
            before_sleep=warn_immediate_errors,
            reraise=True,
        )

    def __call__[**P, R](self, f: Callable[P, R], /) -> Callable[P, R]:
        wrapped_f = self.wrap(f)

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

        if iscoroutinefunction(f):
            return update_wrapper(cast('Callable[P, R]', async_wrapper), f)
        return update_wrapper(wrapper, f)


def _declutter_tb(e: BaseException, code: CodeType) -> None:
    tb = e.__traceback__

    # Drop frames until `code` frame is reached
    while tb:
        if tb.tb_frame.f_code is code:
            e.__traceback__ = tb
            return
        tb = tb.tb_next


def warn_immediate_errors(rcs: t.RetryCallState) -> None:
    if (
        rcs.next_action
        and rcs.outcome
        and (ex := rcs.outcome.exception()) is not None
    ):
        logger.warning(
            'Retrying %s #%d in %.2g seconds as it raised %s: %s.',
            guess_name(rcs.fn),
            rcs.attempt_number,
            rcs.next_action.sleep,
            ex.__class__.__name__,
            ex,
        )


def _make_stop(
    attempts: int | None, timeout: float | timedelta | None
) -> Callable[[t.RetryCallState], bool]:
    # See stamina._core:_make_stop
    if attempts and timeout:
        return t.stop_any(
            t.stop_after_attempt(attempts),
            t.stop_after_delay(timeout),
        )

    if attempts:
        return t.stop_after_attempt(attempts)

    if timeout:
        return t.stop_after_delay(timeout)

    return t.stop_never


def _jittered_backoff_wait(
    initial: float | timedelta = 0.1,
    max_backoff: float | timedelta = 5.0,
    jitter: float | timedelta = 1.0,
    exp_base: float = 2.0,
) -> Callable[[t.RetryCallState], float]:
    # See stamina._core:_compute_backoff
    initial = _to_seconds(initial)
    max_backoff = _to_seconds(max_backoff)
    jitter = _to_seconds(jitter)
    rng = random.Random()

    def wait(rcs: t.RetryCallState) -> float:
        num = rcs.attempt_number - 1
        jitter_ = rng.uniform(0, jitter) if jitter else 0
        return min(
            max_backoff,
            initial * (exp_base**num) + jitter_,
        )

    return wait


def _to_seconds(x: float | timedelta) -> float:
    return x.total_seconds() if isinstance(x, timedelta) else x


def guess_name(obj: object) -> str:
    if obj is None:
        return '<unknown>'
    name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
    mod = getattr(obj, '__module__', None)
    return (f'{mod}.{name}' if mod else name) if name else repr(obj)


def _get_transports() -> tuple[HTTPTransport, AsyncHTTPTransport]:
    limits = Limits(
        max_connections=env.MAX_CONNECTIONS,
        max_keepalive_connections=env.MAX_KEEP_ALIVE_CONNECTIONS,
    )

    # Use SSL_CERT_FILE envvar to pass `cafile`
    sync = HTTPTransport(
        verify=env.SSL_VERIFY, limits=limits, retries=env.RETRIES
    )
    async_ = AsyncHTTPTransport(
        verify=env.SSL_VERIFY, limits=limits, retries=env.RETRIES
    )
    return sync, async_


# Global pool for all HTTP requests
_transport, _atransport = _get_transports()


def get_clients(
    base_url: Any = '',
    timeout: float | None = None,
    follow_redirects: bool = True,
) -> tuple[Client, AsyncClient]:
    base_url = str(base_url)
    sync = Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        base_url=base_url,
        transport=_transport,
    )
    async_ = AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        base_url=base_url,
        transport=_atransport,
    )
    return sync, async_


def raise_for_status(resp: Response) -> asyncio.Future[Response]:
    """Raise status error if one occured.

    Adds more context to `Response.raise_for_status` (like response content).
    For sync response - call `.result()` on returned value first.
    For async response - DON'T FORGET to `await` first.
    """
    if resp.is_success:
        f = asyncio.Future[Response]()
        f.set_result(resp)
        return f

    # closed response or any synchronous response
    if resp.is_closed or not isinstance(resp.stream, AsyncByteStream):
        f = asyncio.Future[Response]()
        f.set_exception(_new_status_error(resp, resp.read()))
        return f

    # opened asynchronous response
    async def _fail() -> Response:
        exc = _new_status_error(resp, await resp.aread())
        raise exc from None

    return asyncio.ensure_future(_fail())


def _new_status_error(resp: Response, content: bytes) -> HTTPStatusError:
    status_cls = resp.status_code // 100
    error_type = _ERROR_TYPES.get(status_cls, 'Invalid status code')
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
