__all__ = [
    'AiohttpTransport',
    'aclient',
    'client',
    'get_clients',
    'get_ip_from_response',
    'raise_for_status',
]

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from typing import Any, Literal, Never, overload
from urllib.parse import unquote

import aiohttp
import httpx
from aiohttp.streams import EofStream
from glow import memoize
from yarl import URL

from ._async import Areturn
from ._env import env

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
                skip_auto_headers=_SKIP_AUTO_HEADERS,
                cookie_jar=aiohttp.DummyCookieJar() if no_cookie else None,
                auto_decompress=False,
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

        try:
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
                compress=False,
                timeout=aiohttp.ClientTimeout(
                    sock_connect=timeout.get('connect'),
                    sock_read=timeout.get('read'),
                    connect=timeout.get('pool'),
                ),
                server_hostname=sni_hostname,
            ).__aenter__()
        except Exception as exc:
            raise _reexcept_aiohttp_to_httpx(exc) from exc

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
    def __init__(self, retries: int = 0) -> None:
        self.retries = retries

    async def __call__(
        self, req: aiohttp.ClientRequest, handler: aiohttp.ClientHandlerType
    ) -> aiohttp.ClientResponse:
        for _ in range(self.retries):  # Try N extra times for non-retry code
            rsp = await handler(req)
            if rsp.status not in _HTTP_RETRY_CODES:
                return rsp
        return await handler(req)  # Final unconditional attempt


class _AiohttpResponseStream(httpx.AsyncByteStream):
    CHUNK_SIZE = 16 * 1024

    def __init__(self, rsp: aiohttp.ClientResponse) -> None:
        self._rsp = rsp

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        try:
            while chunk := await self._rsp.content.read(self.CHUNK_SIZE):
                yield chunk
        except EofStream:
            return
        except Exception as exc:
            raise _reexcept_aiohttp_to_httpx(exc) from exc

    async def aclose(self) -> None:
        try:
            await self._rsp.__aexit__(None, None, None)
        except Exception as exc:
            raise _reexcept_aiohttp_to_httpx(exc) from exc


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


def _reexcept_aiohttp_to_httpx(exc: Exception) -> Exception:
    for aiohttp_exc, httpx_exc in _AIOHTTP_TO_HTTPX_EXCEPTIONS.items():
        if isinstance(exc, aiohttp_exc):
            return httpx_exc(str(exc))

    if isinstance(exc, asyncio.TimeoutError):
        return httpx.TimeoutException(str(exc))

    return httpx.HTTPError(f'Unknown error: {exc!s}')


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
    *,
    aiohttp: bool = False,
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
    async_: httpx.AsyncBaseTransport
    if aiohttp:
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
    transport, atransport = _get_transports(aiohttp=v2)
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


# ------------------------------ error handling ------------------------------


@overload
def raise_for_status(rsp: httpx.Response, /) -> Awaitable[None]: ...
@overload
def raise_for_status(rsp: httpx.Response, /, eager: Literal[True]) -> None: ...


def raise_for_status(
    rsp: httpx.Response, /, eager: bool = False
) -> Awaitable[None] | None:
    """Raise status error if one occured.

    Adds more context to `Response.raise_for_status` (like response content).
    For sync response - returns or raises on call.
    For async response - returns or raises on call when response is read,
    otherwise DON'T FORGET to `await`.
    """
    if rsp.is_success:
        return None if eager else Areturn(None)

    # closed response or any synchronous response
    if rsp.is_closed or not isinstance(rsp.stream, httpx.AsyncByteStream):
        raise _new_status_error(rsp, rsp.read())

    # opened asynchronous response
    if eager:
        raise RuntimeError('Attempted sync error handling on async response')

    async def fail() -> Never:
        raise _new_status_error(rsp, await rsp.aread())

    return fail()


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
