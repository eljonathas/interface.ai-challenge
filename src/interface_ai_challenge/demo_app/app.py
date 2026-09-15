from __future__ import annotations

import asyncio
import re
import secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from interface_ai_challenge.demo_app.data import SYNTHETIC_MEMBERS
from interface_ai_challenge.demo_app.scenarios import DemoState, Scenario

_MEMBER_ID = re.compile(r"M-\d{5,6}")
_TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")


def create_app(scenario: Scenario = Scenario.NORMAL) -> FastAPI:
    """Legacy Credit Union Desk: a deliberately old-fashioned back office with synthetic data."""

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    state = DemoState(scenario)

    def render(request: Request, template: str, **context: Any) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, template, context)

    @app.get("/")
    async def root() -> Response:
        return RedirectResponse("/desk")

    @app.get("/desk", response_class=HTMLResponse)
    async def desk(request: Request) -> HTMLResponse:
        return render(request, "desk.html")

    @app.get("/w/search", response_class=HTMLResponse)
    async def search(request: Request) -> HTMLResponse:
        return render(
            request,
            "search.html",
            field=f"f_{secrets.token_hex(3)}",
            show_notice=scenario is Scenario.MAINTENANCE and state.first_time("maintenance"),
            injection=scenario is Scenario.INJECTION,
        )

    @app.get("/w/results", response_class=HTMLResponse)
    async def results(request: Request, k: str = "") -> HTMLResponse:
        if scenario is Scenario.SESSION_EXPIRED and state.first_time("session"):
            return render(request, "expired.html", error=None)
        if scenario is Scenario.SLOW:
            await asyncio.sleep(2.5)
        if scenario is Scenario.TRANSIENT_ERROR and state.first_time("transient"):
            return render(request, "transient.html", retry_url=f"{request.url.path}?{request.url.query}")
        value = request.query_params.get(k, "").strip() if k.startswith("f_") else ""
        if not _MEMBER_ID.fullmatch(value):
            return render(request, "results.html", error="Invalid member ID format.", rows=[])
        rows = [
            (member, state.token_for(member.id)) for member in SYNTHETIC_MEMBERS.values() if member.id.startswith(value)
        ]
        return render(request, "results.html", error=None, rows=rows)

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request, username: str = Form(""), password: str = Form("")) -> Response:
        if username == "demo" and password == "demo":
            return RedirectResponse("/w/search", status_code=303)
        return render(request, "expired.html", error="Invalid credentials.")

    @app.get("/w/m/{token}", response_class=HTMLResponse)
    async def member(request: Request, token: str) -> HTMLResponse:
        found = state.member_for(token)
        if found is None:
            return render(request, "missing.html")
        return render(
            request,
            "member.html",
            member=found,
            accounts_url=f"/w/m/{token}/accounts",
            shift=90 if scenario is Scenario.CANVAS_SHIFT else 0,
            duplicate=scenario is Scenario.DUPLICATE_CANVAS_CONTROL,
            dialog=scenario is Scenario.UNKNOWN_DIALOG and state.first_time("dialog"),
        )

    @app.get("/w/m/{token}/accounts", response_class=HTMLResponse)
    async def accounts(request: Request, token: str) -> HTMLResponse:
        found = state.member_for(token)
        if found is None:
            return render(request, "missing.html")
        if scenario is Scenario.PERMISSION_DENIED:
            return render(request, "denied.html")
        return render(request, "accounts.html", member=found, close_url=f"/w/m/{token}/close")

    @app.post("/w/m/{token}/close", response_class=HTMLResponse)
    async def close_account(request: Request, token: str) -> HTMLResponse:
        return render(request, "closed.html")

    return app
