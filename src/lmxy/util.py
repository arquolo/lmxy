__all__ = [
    'aclient',
    'aretry',
    'client',
    'get_clients',
    'get_ip_from_response',
    'raise_for_status',
    'warn_immediate_errors',
]

import asyncio
import random
import sys
import urllib.error
from collections.abc import Callable
from datetime import timedelta
from functools import update_wrapper
from inspect import iscoroutinefunction
from types import CodeType, FrameType
from typing import Any, cast

import httpx
from tenacity import RetryCallState, retry
from glow import memoize, register_post_import_hook
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
from loguru import logger

from ._env import env

# --------------------------------- retrying ---------------------------------

_retriable_errors: list[type[BaseException]] = [
    TimeoutError,
    urllib.error.HTTPError,
    HTTPError,
]
register_post_import_hook(
    lambda mod: _retriable_errors.append(mod.HTTPError),
    'requests',
)
register_post_import_hook(
    lambda mod: _retriable_errors.append(mod.ClientError),
    'aiohttp',
)
_inf = float('inf')


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
        predicate: Callable[[BaseException], bool] | None = None,
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
        exc_tps = (
            extra_errors
            if override_defaults
            else (*_retriable_errors, *extra_errors)
        )
        self.wrap = retry(
            stop=_Stop(
                attempts=max_attempts or _inf,
                timeout=timeout or _inf,
            ),
            wait=_JitteredBackoffWait(
                initial=wait_initial,
                max_backoff=wait_max,
                jitter=wait_jitter,
                exp_base=wait_exp_base,
            ),
            retry=_Retry(exc_tps, predicate),
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


def warn_immediate_errors(rcs: RetryCallState) -> None:
    if (
        rcs.next_action
        and rcs.outcome
        and (ex := rcs.outcome.exception()) is not None
    ):
        f: FrameType | None = sys._getframe(1)

        depth = 2  # this frame + ? `tenacity` frames + `aretry` frame
        while f and 'tenacity' in f.f_code.co_filename:
            f = f.f_back
            depth += 1

        logger.opt(depth=depth).warning(
            f'#{rcs.attempt_number} in {rcs.next_action.sleep:.2g}s - '
            f'{ex.__class__.__name__}: {ex}'
        )


class _Stop:  # See stamina._core:_make_stop
    def __init__(
        self,
        attempts: float | int = 10,
        timeout: float | timedelta = 45,
    ) -> None:
        self.attempts = attempts
        self.timeout = _to_seconds(timeout)

    def __call__(self, rcs: RetryCallState) -> bool:
        assert rcs.seconds_since_start is not None
        return (
            rcs.attempt_number >= self.attempts
            or rcs.seconds_since_start >= self.timeout
        )


class _JitteredBackoffWait:  # See stamina._core:_compute_backoff
    def __init__(
        self,
        initial: float | timedelta = 0.1,
        max_backoff: float | timedelta = 5.0,
        jitter: float | timedelta = 1.0,
        exp_base: float = 2.0,
    ) -> None:
        self.initial = _to_seconds(initial)
        self.max_backoff = _to_seconds(max_backoff)
        self.jitter = _to_seconds(jitter)
        self.exp_base = exp_base
        self.rng = random.Random()

    def __call__(self, rcs: RetryCallState) -> float:
        num = rcs.attempt_number - 1
        jitter = self.rng.uniform(0, self.jitter) if self.jitter else 0
        return min(
            self.max_backoff,
            self.initial * (self.exp_base**num) + jitter,
        )


class _Retry:
    def __init__(
        self,
        exc_types: tuple[type[BaseException], ...],
        predicate: Callable[[BaseException], bool] | None,
    ) -> None:
        self.exc_types = exc_types
        self.predicate = predicate

    def __call__(self, rcs: RetryCallState) -> bool:
        assert rcs.outcome is not None
        ex = rcs.outcome.exception()
        return ex is not None and (
            isinstance(ex, self.exc_types)
            or (self.predicate(ex) if self.predicate else False)
        )


def _to_seconds(x: float | timedelta) -> float:
    return x.total_seconds() if isinstance(x, timedelta) else x


def guess_name(obj: object) -> str:
    if obj is None:
        return '<unknown>'
    name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
    mod = getattr(obj, '__module__', None)
    return (f'{mod}.{name}' if mod else name) if name else repr(obj)


@memoize()  # Global pool for all HTTP requests
def _get_transports() -> tuple[httpx.BaseTransport, httpx.AsyncBaseTransport]:
    limits = Limits(
        max_connections=env.MAX_CONNECTIONS,
        max_keepalive_connections=env.MAX_KEEP_ALIVE_CONNECTIONS,
        keepalive_expiry=env.KEEP_ALIVE_TIMEOUT,
    )

    # Use SSL_CERT_FILE envvar to pass `cafile`
    sync = HTTPTransport(
        verify=env.SSL_VERIFY,
        limits=limits,
        retries=env.RETRIES,
    )
    async_ = AsyncHTTPTransport(
        verify=env.SSL_VERIFY,
        limits=limits,
        retries=env.RETRIES,
    )
    return sync, async_


def get_clients(
    base_url: Any = '',
    timeout: float | None = None,
    follow_redirects: bool = True,
) -> tuple[Client, AsyncClient]:
    base_url = str(base_url)
    transport, atransport = _get_transports()
    sync = Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        base_url=base_url,
        transport=transport,
    )
    async_ = AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        base_url=base_url,
        transport=atransport,
    )
    return sync, async_


def get_ip_from_response(resp: Response) -> str | None:
    ns = resp.extensions.get('network_stream')
    if ns is None:
        return None
    return ns.get_extra_info('server_addr')


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

client, aclient = get_clients()
