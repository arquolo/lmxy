__all__ = ['guess_name', 'min_max']

from collections.abc import Iterable


def guess_name(obj: object) -> str:
    if obj is None:
        return '<unknown>'
    name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
    mod = getattr(obj, '__module__', None)
    return (f'{mod}.{name}' if mod else name) if name else repr(obj)


def min_max(xs: Iterable[float], /) -> list[float]:
    xs = list(xs)
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if ptp := hi - lo:
        return [(x - lo) / ptp for x in xs]  # scale to 0..1
    return [0.5] * len(xs)
