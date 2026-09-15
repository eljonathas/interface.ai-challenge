from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from interface_ai_challenge.application.operator_service import OperatorService
from interface_ai_challenge.domain.computer import (
    ClickCommand,
    ComputerCommand,
    KeypressCommand,
    ScrollCommand,
    TypeCommand,
)
from interface_ai_challenge.domain.errors import ControlError, PolicyViolationError, StaleObservationError, SurfaceError
from interface_ai_challenge.domain.geometry import Point

_COOKIE = "operator_session"
_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@dataclass(frozen=True)
class _OperatorSession:
    operator_id: str
    csrf_token: str


class ClickBody(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    epoch: int
    revision: int


class TypeBody(BaseModel):
    text: str = Field(min_length=1, max_length=256)
    epoch: int


class KeyBody(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=3)
    epoch: int


class ScrollBody(BaseModel):
    x: int
    y: int
    delta_y: int
    epoch: int
    revision: int


def create_operator_app(service: OperatorService, access_token: str) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    sessions: dict[str, _OperatorSession] = {}

    def session_of(request: Request, mutating: bool = False) -> _OperatorSession:
        session = sessions.get(request.cookies.get(_COOKIE, ""))
        if session is None:
            raise HTTPException(status_code=401, detail="open the console link printed by the CLI")
        if mutating and not secrets.compare_digest(request.headers.get("x-csrf-token", ""), session.csrf_token):
            raise HTTPException(status_code=403, detail="missing or invalid CSRF token")
        origin = request.headers.get("origin")
        if mutating and origin is not None and origin != f"{request.url.scheme}://{request.url.netloc}":
            raise HTTPException(status_code=403, detail="cross-origin request refused")
        return session

    async def perform(request: Request, epoch: int, revision: int, command: ComputerCommand) -> dict[str, str]:
        session = session_of(request, mutating=True)
        try:
            await service.act(session.operator_id, epoch, revision, command)
        except (ControlError, StaleObservationError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        except PolicyViolationError as error:
            raise HTTPException(status_code=403, detail=error.reason) from None
        except SurfaceError:
            raise HTTPException(status_code=502, detail="the live session did not accept the action") from None
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, token: str | None = None) -> Response:
        if token is not None:
            if not secrets.compare_digest(token, access_token):
                raise HTTPException(status_code=401, detail="invalid access token")
            session_id = secrets.token_urlsafe(24)
            sessions[session_id] = _OperatorSession(
                operator_id=f"operator-{secrets.token_hex(3)}", csrf_token=secrets.token_urlsafe(24)
            )
            redirect = RedirectResponse("/", status_code=303)
            redirect.set_cookie(_COOKIE, session_id, httponly=True, samesite="strict")
            return redirect
        session = session_of(request)
        page = _PAGE.replace("__CSRF__", session.csrf_token).replace("__OPERATOR__", session.operator_id)
        return HTMLResponse(page, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY"})

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, object]:
        session = session_of(request)
        return {"operator_id": session.operator_id, **service.status()}

    @app.get("/api/screenshot")
    async def screenshot(request: Request) -> Response:
        session_of(request)
        try:
            shot = await service.screenshot()
        except SurfaceError:
            raise HTTPException(status_code=502, detail="live session unavailable") from None
        return Response(
            shot.png, media_type="image/png", headers={"X-Revision": str(shot.revision), "Cache-Control": "no-store"}
        )

    @app.post("/api/claim")
    async def claim(request: Request) -> dict[str, int]:
        session = session_of(request, mutating=True)
        try:
            token = await service.claim(session.operator_id)
        except ControlError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        return {"epoch": token.epoch}

    @app.post("/api/click")
    async def click(request: Request, body: ClickBody) -> dict[str, str]:
        return await perform(request, body.epoch, body.revision, ClickCommand(point=Point(x=body.x, y=body.y)))

    @app.post("/api/type")
    async def type_text(request: Request, body: TypeBody) -> dict[str, str]:
        return await perform(request, body.epoch, -1, TypeCommand(text=body.text))

    @app.post("/api/key")
    async def key(request: Request, body: KeyBody) -> dict[str, str]:
        return await perform(request, body.epoch, -1, KeypressCommand(keys=tuple(body.keys)))

    @app.post("/api/scroll")
    async def scroll(request: Request, body: ScrollBody) -> dict[str, str]:
        command = ScrollCommand(point=Point(x=body.x, y=body.y), delta_x=0, delta_y=body.delta_y)
        return await perform(request, body.epoch, body.revision, command)

    @app.post("/api/hand-back")
    async def hand_back(request: Request) -> dict[str, str]:
        session = session_of(request, mutating=True)
        try:
            await service.hand_back(session.operator_id)
        except ControlError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        return {"status": "returned"}

    @app.post("/api/abort")
    async def abort(request: Request) -> dict[str, str]:
        session = session_of(request, mutating=True)
        try:
            await service.abort(session.operator_id)
        except ControlError as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        return {"status": "aborted"}

    return app


class OperatorConsoleServer:
    def __init__(self, app: FastAPI, port: int) -> None:
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            if self._task.done():
                await self._task
                raise RuntimeError("operator console failed to start")
            await asyncio.sleep(0.05)

    async def stop(self) -> None:
        self._server.should_exit = True
        if self._task is not None:
            await self._task
