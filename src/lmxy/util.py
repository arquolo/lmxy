__all__ = [
    'AiohttpTransport',
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
from contextlib import contextmanager
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import timedelta
from functools import update_wrapper
from inspect import iscoroutinefunction
from types import CodeType, FrameType
from typing import Any, cast
from urllib.parse import unquote

import aiohttp
import httpx
from tenacity import RetryCallState, retry
from glow import memoize, register_post_import_hook
from loguru import logger
from yarl import URL

from ._env import env

# --------------------------------- retrying ---------------------------------

_FutureResponse = asyncio.Future[httpx.Response]

_retriable_errors: list[type[BaseException]] = [
    TimeoutError,
    urllib.error.HTTPError,
    httpx.HTTPError,
    aiohttp.ClientError,
]
register_post_import_hook(
    lambda mod: _retriable_errors.append(mod.HTTPError),
    'requests',
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


# ---------------------------- aiohttp transport -----------------------------


class AiohttpTransport(httpx.AsyncBaseTransport):
    __slots__ = ('_get_session', '_session')

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        no_cookie: bool = True,
        verify: bool = False,
        keepalive_timeout: float = 15,
        max_connections: int = 100,
        max_connections_per_host: int = 0,
        proxy: str | None = None,
        retries: int = 0,
        force_close: bool = False,
    ) -> None:
        def get_session() -> aiohttp.ClientSession:
            connector = aiohttp.TCPConnector(
                ssl=verify,
                ttl_dns_cache=None,
                keepalive_timeout=None if force_close else keepalive_timeout,
                force_close=force_close,
                limit=max_connections,
                limit_per_host=max_connections_per_host,
                enable_cleanup_closed=True,
            )
            return aiohttp.ClientSession(
                connector=connector,
                headers=headers,
                proxy=proxy,
                cookie_jar=aiohttp.DummyCookieJar() if no_cookie else None,
                middlewares=[_RetryMiddleware(retries)] if retries else (),
            )

        # TCPConnector and ClientSession want running event loop in __init__.
        # Construct them in async method so they will get it.
        self._get_session = get_session
        self._session: aiohttp.ClientSession | None = None

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        if self._session is None:
            self._session = self._get_session()
        if self._session.closed:
            raise RuntimeError('Transport is closed')

        url = _httpx_to_yarl_url(request.url)
        url_ = url.with_query(())
        params = [(k, unquote(v)) for k, v in url.query.items()]

        timeout = request.extensions.get('timeout', {})
        sni_hostname = request.extensions.get('sni_hostname')

        with _map_aiohttp_exceptions():
            data: bytes | httpx.AsyncByteStream | None
            try:
                data = request.content or None
            except httpx.RequestNotRead:
                data = request.stream  # type: ignore
                request.headers.pop('transfer-encoding', None)

            rsp = await self._session.request(
                method=request.method,
                url=url_,
                params=params or None,
                data=data,
                headers=request.headers,
                allow_redirects=True,
                auto_decompress=False,
                compress=False,
                skip_auto_headers=_SKIP_AUTO_HEADERS,
                timeout=aiohttp.ClientTimeout(
                    sock_connect=timeout.get('connect'),
                    sock_read=timeout.get('read'),
                    connect=timeout.get('pool'),
                ),
                server_hostname=sni_hostname,
            ).__aenter__()

        extensions = {'http_version': b'HTTP/1.1'}
        if rsp.reason:
            extensions['reason_phrase'] = rsp.reason.encode()

        return httpx.Response(
            rsp.status,
            headers=rsp.raw_headers,
            content=_AiohttpResponseStream(rsp),
            request=request,
            extensions=extensions,
        )

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()


class _RetryMiddleware:
    def __init__(self, attempts: int = 0) -> None:
        self.attempts = attempts

    async def __call__(
        self, req: aiohttp.ClientRequest, handler: aiohttp.ClientHandlerType
    ) -> aiohttp.ClientResponse:
        rsp = await handler(req)
        if rsp.status not in _HTTP_RETRY_CODES:
            return rsp

        for _ in range(self.attempts):  # Try more
            rsp = await handler(req)
            if rsp.status not in _HTTP_RETRY_CODES:
                return rsp

        return rsp


class _AiohttpResponseStream(httpx.AsyncByteStream):
    CHUNK_SIZE = 1024 * 16

    def __init__(self, rsp: aiohttp.ClientResponse) -> None:
        self._rsp = rsp

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with _map_aiohttp_exceptions():
            async for chunk in self._rsp.content.iter_chunked(self.CHUNK_SIZE):
                yield chunk

    async def aclose(self) -> None:
        with _map_aiohttp_exceptions():
            await self._rsp.__aexit__(None, None, None)


def _httpx_to_yarl_url(url: httpx.URL) -> URL:
    return URL.build(
        scheme=url.scheme,
        user=url.username or None,
        password=url.password or None,
        host=url.host,
        port=url.port,
        path=url.path,
        query_string=url.query.decode(),
        fragment=url.fragment,
    )


@contextmanager
def _map_aiohttp_exceptions() -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        for aiohttp_exc, httpx_exc in _AIOHTTP_TO_HTTPX_EXCEPTIONS.items():
            if isinstance(exc, aiohttp_exc):
                raise httpx_exc(str(exc)) from exc

        if isinstance(exc, asyncio.TimeoutError):
            raise httpx.TimeoutException(str(exc)) from exc

        raise httpx.HTTPError(f'Unknown error: {exc!s}') from exc


_SKIP_AUTO_HEADERS = frozenset(
    {
        'accept',
        'accept-encoding',
        'connection',
        'content-encoding',
        'deflate',
        'user-agent',
    }
)
_HTTP_RETRY_CODES = (
    408,  # request timeout
    429,  # too many requests
    500,  # internal server error
    502,  # bad gateway
    503,  # service unavailable
    504,  # gateway timeout
)
_AIOHTTP_TO_HTTPX_EXCEPTIONS: dict[type[Exception], type[Exception]] = {
    # Order matters here, most specific exception first
    aiohttp.ClientSSLError: httpx.ProtocolError,
    aiohttp.ClientProxyConnectionError: httpx.ProxyError,
    aiohttp.ClientConnectorDNSError: httpx.ConnectError,
    aiohttp.ClientConnectorError: httpx.ConnectError,
    aiohttp.SocketTimeoutError: httpx.ReadTimeout,
    aiohttp.ServerTimeoutError: httpx.TimeoutException,
    aiohttp.ServerDisconnectedError: httpx.ReadError,
    aiohttp.ServerFingerprintMismatch: httpx.ProtocolError,
    aiohttp.TooManyRedirects: httpx.TooManyRedirects,
    aiohttp.ContentTypeError: httpx.ReadError,
    aiohttp.ClientHttpProxyError: httpx.ProxyError,
    aiohttp.ClientOSError: httpx.ConnectError,
    aiohttp.ClientConnectionResetError: httpx.ConnectError,
    aiohttp.ClientConnectionError: httpx.NetworkError,
    aiohttp.ClientPayloadError: httpx.ReadError,
    aiohttp.NonHttpUrlClientError: httpx.UnsupportedProtocol,
    aiohttp.InvalidUrlClientError: httpx.UnsupportedProtocol,
    aiohttp.InvalidURL: httpx.InvalidURL,
    aiohttp.ClientError: httpx.RequestError,
}

# ------------------------------ httpx clients -------------------------------


@memoize()  # Global pool for all HTTP requests
def _get_transports(
    v2: bool = False,
) -> tuple[httpx.BaseTransport, httpx.AsyncBaseTransport]:
    limits = httpx.Limits(
        max_connections=env.MAX_CONNECTIONS,
        max_keepalive_connections=env.MAX_KEEP_ALIVE_CONNECTIONS,
        keepalive_expiry=env.KEEP_ALIVE_TIMEOUT,
    )

    # Use SSL_CERT_FILE envvar to pass `cafile`
    sync = httpx.HTTPTransport(
        verify=env.SSL_VERIFY,
        limits=limits,
        retries=env.RETRIES,
    )
    if v2:
        async_ = AiohttpTransport(
            verify=env.SSL_VERIFY,
            max_connections=env.MAX_CONNECTIONS,
            keepalive_timeout=env.KEEP_ALIVE_TIMEOUT,
            retries=env.RETRIES,
        )
    else:
        async_ = httpx.AsyncHTTPTransport(
            verify=env.SSL_VERIFY,
            limits=limits,
            retries=env.RETRIES,
        )
    return sync, async_


def get_clients(
    base_url: Any = '',
    timeout: float | None = None,
    follow_redirects: bool = True,
    v2: bool = False,
) -> tuple[httpx.Client, httpx.AsyncClient]:
    base_url = str(base_url)
    transport, atransport = _get_transports(v2)
    sync = httpx.Client(
        timeout=timeout,
        follow_redirects=follow_redirects,
        base_url=base_url,
        transport=transport,
    )
    async_ = httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        base_url=base_url,
        transport=atransport,
    )
    return sync, async_


def get_ip_from_response(rsp: httpx.Response, /) -> str | None:
    ns = rsp.extensions.get('network_stream')
    if ns is None:
        return None
    return ns.get_extra_info('server_addr')


def raise_for_status(rsp: httpx.Response, /) -> _FutureResponse:
    """Raise status error if one occured.

    Adds more context to `Response.raise_for_status` (like response content).
    For sync response - call `.result()` on returned value first.
    For async response - DON'T FORGET to `await` first.
    """
    if rsp.is_success:
        f = _FutureResponse()
        f.set_result(rsp)
        return f

    # closed response or any synchronous response
    if rsp.is_closed or not isinstance(rsp.stream, httpx.AsyncByteStream):
        f = _FutureResponse()
        f.set_exception(_new_status_error(rsp, rsp.read()))
        return f

    # opened asynchronous response
    async def _fail() -> httpx.Response:
        exc = _new_status_error(rsp, await rsp.aread())
        raise exc from None

    return asyncio.ensure_future(_fail())


def _new_status_error(
    rsp: httpx.Response, content: bytes
) -> httpx.HTTPStatusError:
    status_cls = rsp.status_code // 100
    error_type = _ERROR_TYPES.get(status_cls, 'Invalid status code')
    message = (
        f"{error_type} '{rsp.status_code} {rsp.reason_phrase}' "
        f"for url '{rsp.url}' failed with {content.decode()}"
    )
    return httpx.HTTPStatusError(message, request=rsp.request, response=rsp)


_ERROR_TYPES = {
    1: 'Informational response',
    3: 'Redirect response',
    4: 'Client error',
    5: 'Server error',
}

client, aclient = get_clients()
