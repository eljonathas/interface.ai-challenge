from __future__ import annotations

import functools
from collections.abc import Callable, Coroutine
from typing import Any

from playwright.async_api import Error as PlaywrightError

from interface_ai_challenge.domain.errors import SurfaceError


def surface_errors[**P, T](function: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, T]]:
    """Translates driver exceptions into the domain's SurfaceError so no Playwright type leaks upward."""

    @functools.wraps(function)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await function(*args, **kwargs)
        except PlaywrightError as error:
            raise SurfaceError(error.message.splitlines()[0] if error.message else "surface error") from None

    return wrapper
