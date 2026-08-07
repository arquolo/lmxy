__all__ = [
    'aretry',
    'warn_immediate_errors',
]

import random
import sys
import urllib.error
from collections.abc import Callable
from datetime import timedelta
from functools import update_wrapper
from inspect import iscoroutinefunction
from types import FrameType
from typing import cast

import aiohttp
import httpx
from glow import declutter_tb, register_post_import_hook
from loguru import logger
from tenacity import RetryCallState, retry

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
                declutter_tb(exc, f.__code__)
                raise

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return wrapped_f(*args, **kwargs)
            except BaseException as exc:
                declutter_tb(exc, f.__code__)
                raise

        if iscoroutinefunction(f):
            return update_wrapper(cast('Callable[P, R]', async_wrapper), f)
        return update_wrapper(wrapper, f)


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
        attempts: float = 10,
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
