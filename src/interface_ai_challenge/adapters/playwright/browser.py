from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from playwright.async_api import Page, Route, async_playwright

from interface_ai_challenge.domain.geometry import Size


@asynccontextmanager
async def open_isolated_page(
    url_allowed: Callable[[str], bool],
    on_blocked: Callable[[str], None],
    viewport: Size,
    headless: bool = True,
) -> AsyncIterator[Page]:
    """Ephemeral browser context: no persistent profile, no service workers, no downloads, allowlisted requests only."""

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": viewport.width, "height": viewport.height},
            service_workers="block",
            accept_downloads=False,
        )

        async def enforce_allowlist(route: Route) -> None:
            url = route.request.url
            if url_allowed(url):
                await route.continue_()
                return
            on_blocked(_without_query(url))
            await route.abort("blockedbyclient")

        await context.route("**/*", enforce_allowlist)
        page = await context.new_page()

        async def close_popup(popup: Page) -> None:
            on_blocked(_without_query(popup.url))
            await popup.close()

        context.on("page", close_popup)
        try:
            yield page
        finally:
            await context.close()
            await browser.close()


def _without_query(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}" if parts.scheme else url[:64]
