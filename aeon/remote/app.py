"""Authenticated FastAPI application for managing Aeon terminal tabs."""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import RemoteConfig
from .controller_lock import ControllerLock
from .instances import InstanceError, InstanceManager
from .security import (
    AuthService,
    AuthenticationError,
    LoginRateLimited,
)
from .store import RemoteStore
from .terminal import bridge_terminal


WEBSOCKET_SESSION_RECHECK_SECONDS = 1.0


async def _websocket_session(auth: AuthService, raw_session_token: str | None):
    """Read one durable WebSocket session without blocking the ASGI loop."""

    try:
        return await asyncio.to_thread(auth.session, raw_session_token)
    except Exception:
        # Authentication storage failures fail closed without exposing the raw
        # cookie or database details.
        return None


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)
    otp: str = Field(default="", max_length=32)
    remember: bool = False


class WorkspaceBody(BaseModel):
    root: str = Field(min_length=1, max_length=4096)
    name: str = Field(min_length=1, max_length=80)


class InstanceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["aeon", "codex", "claude", "grok"] = "aeon"
    name: str = Field(min_length=1, max_length=64)
    workspace: str = Field(min_length=1, max_length=4096)
    objective: str = Field(default="", max_length=20000)
    max_iterations: int | None = Field(default=None, ge=1, le=10000)


class TerminalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Nexus normally omits the name; the server allocates collision-safe
    # "Terminal N" labels. A custom legacy label remains accepted.
    name: str | None = Field(default=None, min_length=1, max_length=64)
    workspace: str = Field(min_length=1, max_length=4096)
    project_id: str | None = Field(default=None, pattern=r"^pr-[0-9a-f]{32}$")
    host_id: Literal[
        "192.168.0.177",
        "192.168.0.178",
        "192.168.0.179",
        "192.168.0.180",
    ] = "192.168.0.177"


class RenameInstanceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)


class EmptyActionBody(BaseModel):
    """Allow no client-controlled parameters for server-derived actions."""

    model_config = ConfigDict(extra="forbid")


class ActivateAgentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["aeon", "codex", "claude", "grok"]


class AgentSettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["aeon", "codex", "claude", "grok"]
    model: str = Field(max_length=160)
    effort: str = Field(max_length=32)
    harness: str | None = Field(default=None, max_length=64)


class ContinuousModeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    goal: str = Field(default="", max_length=20_000)


class AgentSkillBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=64 * 1024)
    expected_revision: str = Field(pattern=r"^[0-9a-f]{64}$")


class DeleteAgentSkillBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation: str = Field(min_length=1, max_length=200)


class AgentSkillTransferSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_path: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}/[A-Za-z0-9][A-Za-z0-9_-]{0,79}$"
    )
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")


class AgentSkillTransferBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_instance_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    skills: list[AgentSkillTransferSelection] = Field(min_length=1, max_length=64)
    include_knowledge: bool = True


class ForkConversationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(pattern=r"^msg-[A-Za-z0-9_-]{32}$")


class ConfirmationBody(BaseModel):
    confirmation: str = Field(min_length=1, max_length=64)


def _client_ip(request) -> str:
    return request.client.host if request.client else ""


def create_app(
    config: RemoteConfig | None = None,
    *,
    store: RemoteStore | None = None,
    manager: InstanceManager | None = None,
    auth: AuthService | None = None,
    static_dir: str | Path | None = None,
    title: str = "Aeon Remote",
    health_identity: str = "aeon-remote-v1",
    startup_initializer: Callable[[], None] | None = None,
) -> FastAPI:
    config = config or RemoteConfig.from_env()
    config.validate_server()
    config.prepare_state()
    store = store or RemoteStore(config.database_path)
    manager = manager or InstanceManager(store, config)
    auth = auth or AuthService(store, config)
    static_dir = (
        Path(static_dir).expanduser().resolve()
        if static_dir is not None
        else Path(__file__).with_name("static")
    )
    if not static_dir.is_dir():
        raise RuntimeError(f"Remote console static directory does not exist: {static_dir}")

    @asynccontextmanager
    async def controller_lifespan(application: FastAPI):
        # WebSocket generations and lifecycle locks are deliberately in-process.
        # One lifetime OS lock therefore makes a registry single-controller across
        # standalone Aeon Remote, Nexus, and accidental multi-worker starts.
        active_lock = ControllerLock.acquire(config.state_dir)
        application.state.controller_lock = active_lock
        guard_setter = getattr(store, "set_controller_guard", None)
        try:
            if callable(guard_setter):
                guard_setter(active_lock.assert_current)
            active_lock.assert_current()
            bootstrap = getattr(manager, "bootstrap", None)
            if callable(bootstrap):
                bootstrap()
            if startup_initializer is not None:
                active_lock.assert_current()
                startup_initializer()
            yield
        finally:
            active_lock.close()
            if callable(guard_setter):
                guard_setter(None)
            application.state.controller_lock = None

    app = FastAPI(
        title=title,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=controller_lifespan,
    )
    app.state.config = config
    app.state.store = store
    app.state.manager = manager
    app.state.auth = auth
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.allowed_hosts))
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        websocket_origins = [
            origin.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            for origin in config.allowed_origins
        ]
        connect_sources = " ".join(["'self'", *websocket_origins])
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(self), geolocation=(), payment=(), usb=()"
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers.setdefault("Content-Security-Policy", (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net "
            "'sha384-M169f14mRZOXm3hD/v2Ti0ThIT/RnAQagXA9nlE15yHAtrW19gdePJh/HaTzUOe/' "
            "'sha384-iF+jqbuti4XlB64clWgFWYEscb+UnSRv3VgVikGYZu+otNFnSHr7y7NcKfBnGizn'; "
            "style-src 'self' https://cdn.jsdelivr.net "
            "'sha384-8Xk9wy/gzEDUKrXtrmCFa2bBuK3BpjpDuL/p0SeKQX19Khl/M+lHOgD/CyYf7efP'; "
            "style-src-attr 'unsafe-inline'; "
            f"img-src 'self' data:; connect-src {connect_sources}; "
            "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        ))
        if not config.allow_insecure_http:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains"
            )
        return response

    def current_session(request: Request):
        session = auth.session(request.cookies.get(config.cookie_name))
        if not session:
            raise HTTPException(status_code=401, detail="Authentication required")
        return session

    def protected(request: Request, session=Depends(current_session)):
        origin = request.headers.get("origin")
        if origin not in config.allowed_origins:
            raise HTTPException(status_code=403, detail="Invalid request origin")
        supplied = request.headers.get("x-csrf-token", "")
        if not hmac.compare_digest(supplied, session["csrf_token"]):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
        return session

    def translate_instance_error(exc: InstanceError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/healthz")
    def health():
        return {
            "ok": True,
            "configured": store.admin_count() > 0,
            "identity": health_identity,
            "oidc_configured": bool(getattr(app.state, "oidc_configured", False)),
        }

    @app.post("/api/login")
    def login(body: LoginBody, request: Request, response: Response):
        origin = request.headers.get("origin")
        if origin not in config.allowed_origins:
            raise HTTPException(status_code=403, detail="Invalid request origin")
        if store.admin_count() == 0:
            raise HTTPException(
                status_code=503,
                detail="No administrator exists; run aeon-remote init-admin locally",
            )
        try:
            auth_kwargs = {
                "client_ip": _client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "remember": body.remember,
            }
            if config.require_totp:
                result = auth.authenticate(
                    body.username,
                    body.password,
                    body.otp,
                    **auth_kwargs,
                )
            else:
                result = auth.authenticate_password(
                    body.username,
                    body.password,
                    **auth_kwargs,
                )
        except LoginRateLimited as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        response.set_cookie(
            config.cookie_name,
            result.token,
            max_age=result.max_age,
            secure=not config.allow_insecure_http,
            httponly=True,
            samesite=config.cookie_samesite,
            path="/",
        )
        return {
            "authenticated": True,
            "username": result.username,
            "csrf_token": result.csrf_token,
            "expires_at": result.expires_at,
        }

    @app.get("/api/session")
    def session_info(session=Depends(current_session)):
        return {
            "authenticated": True,
            "username": session["username"],
            "csrf_token": session["csrf_token"],
            "expires_at": session["expires_at"],
        }

    @app.post("/api/logout")
    def logout(
        request: Request,
        response: Response,
        session=Depends(protected),
    ):
        auth.logout(
            request.cookies.get(config.cookie_name), client_ip=_client_ip(request)
        )
        response.delete_cookie(
            config.cookie_name,
            path="/",
            secure=not config.allow_insecure_http,
            httponly=True,
            samesite=config.cookie_samesite,
        )
        response.headers["Clear-Site-Data"] = '"cache", "cookies", "storage"'
        return {"authenticated": False}

    @app.get("/api/workspaces")
    def workspaces(session=Depends(current_session)):
        return manager.list_workspaces()

    @app.post("/api/workspaces")
    def create_workspace(
        body: WorkspaceBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            path = manager.create_workspace(body.root, body.name)
        except InstanceError as exc:
            translate_instance_error(exc)
        store.audit(
            "workspace_created",
            actor=session["username"],
            client_ip=_client_ip(request),
            details={"workspace": path},
        )
        return {"workspace": path}

    @app.get("/api/instances")
    def instances(session=Depends(current_session)):
        return {"instances": manager.list_instances()}

    @app.get("/api/terminal-hosts")
    def terminal_hosts(session=Depends(current_session)):
        return manager.list_terminal_hosts()

    @app.post("/api/instances")
    def create_instance(
        body: InstanceBody,
        request: Request,
        session=Depends(protected),
    ):
        # Browser-created work always begins as a fixed managed terminal. This
        # legacy endpoint is retained only as an explicit fail-closed response;
        # direct agents could otherwise bypass prompt/PGID identity, fresh
        # provider gating, and the in-place terminal lifecycle contract.
        raise HTTPException(
            status_code=400,
            detail="Create a terminal, then use activate-agent in that tab",
        )

    @app.post("/api/terminals")
    def create_terminal(
        body: TerminalBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            terminal = manager.create_terminal(
                name=body.name,
                workspace=body.workspace,
                host_id=body.host_id,
                project_id=body.project_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": terminal}

    @app.put("/api/instances/{instance_id}/name")
    def rename_instance(
        instance_id: str,
        body: RenameInstanceBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            instance = manager.rename_instance(
                instance_id,
                name=body.name,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": instance}

    @app.post("/api/instances/{terminal_id}/start-aeon-here")
    def start_aeon_here(
        terminal_id: str,
        request: Request,
        body: EmptyActionBody | None = None,
        session=Depends(protected),
    ):
        try:
            instance = manager.start_aeon_here(
                terminal_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": instance}

    @app.post("/api/instances/{instance_id}/activate-agent")
    def activate_agent(
        instance_id: str,
        body: ActivateAgentBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            value = manager.activate_agent(
                instance_id,
                kind=body.kind,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": value}

    @app.get("/api/instances/{instance_id}/agent-settings")
    def agent_settings(
        instance_id: str,
        session=Depends(current_session),
    ):
        try:
            return manager.get_agent_settings(instance_id)
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.post("/api/instances/{instance_id}/fork")
    def fork_instance_conversation(
        instance_id: str,
        body: ForkConversationBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            fork = manager.fork_agent_chat(
                instance_id,
                body.message_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": fork}

    @app.post("/api/instances/{instance_id}/close-fork")
    def close_instance_conversation_fork(
        instance_id: str,
        request: Request,
        body: EmptyActionBody | None = None,
        session=Depends(protected),
    ):
        try:
            manager.close_agent_chat_fork(
                instance_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"closed": True}

    @app.put("/api/instances/{instance_id}/agent-settings")
    def update_agent_settings(
        instance_id: str,
        body: AgentSettingsBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            return manager.update_agent_settings(
                instance_id,
                kind=body.kind,
                model=body.model,
                effort=body.effort,
                actor=session["username"],
                harness=body.harness,
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.get("/api/instances/{instance_id}/skills")
    def agent_created_skills(instance_id: str, session=Depends(current_session)):
        try:
            return manager.get_private_skills(instance_id)
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.put("/api/instances/{instance_id}/skills/{category}/{skill_name}")
    def update_agent_created_skill(
        instance_id: str,
        category: str,
        skill_name: str,
        body: AgentSkillBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            return manager.update_private_skill(
                instance_id,
                category=category,
                skill_name=skill_name,
                content=body.content,
                expected_revision=body.expected_revision,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.delete("/api/instances/{instance_id}/skills/{category}/{skill_name}")
    def delete_agent_created_skill(
        instance_id: str,
        category: str,
        skill_name: str,
        body: DeleteAgentSkillBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            return manager.delete_private_skill(
                instance_id,
                category=category,
                skill_name=skill_name,
                expected_revision=body.expected_revision,
                confirmation=body.confirmation,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.post("/api/instances/{instance_id}/skills/transfer")
    def transfer_agent_created_skills(
        instance_id: str,
        body: AgentSkillTransferBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            return manager.transfer_private_skills(
                instance_id,
                source_instance_id=body.source_instance_id,
                selections=[item.model_dump() for item in body.skills],
                include_knowledge=body.include_knowledge,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.get("/api/instances/{instance_id}/continuous-mode")
    def continuous_mode(
        instance_id: str,
        session=Depends(current_session),
    ):
        try:
            return manager.get_continuous_mode(instance_id)
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.put("/api/instances/{instance_id}/continuous-mode")
    def update_continuous_mode(
        instance_id: str,
        body: ContinuousModeBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            return manager.update_continuous_mode(
                instance_id,
                enabled=body.enabled,
                goal=body.goal,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)

    @app.post("/api/instances/{instance_id}/end-agent")
    def end_agent(
        instance_id: str,
        request: Request,
        body: EmptyActionBody | None = None,
        session=Depends(protected),
    ):
        try:
            value = manager.end_agent(
                instance_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": value}

    @app.post("/api/instances/{instance_id}/fresh-context")
    def fresh_agent_context(
        instance_id: str,
        request: Request,
        body: EmptyActionBody | None = None,
        session=Depends(protected),
    ):
        try:
            value = manager.fresh_restart_agent(
                instance_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": value}

    @app.post("/api/instances/{instance_id}/stop")
    def stop_instance(
        instance_id: str,
        request: Request,
        session=Depends(protected),
    ):
        try:
            value = manager.graceful_stop(
                instance_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": value}

    @app.post("/api/instances/{instance_id}/resume")
    def resume_instance(
        instance_id: str,
        request: Request,
        session=Depends(protected),
    ):
        try:
            value = manager.resume_instance(
                instance_id,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": value}

    @app.post("/api/instances/{instance_id}/force-stop")
    def force_stop_instance(
        instance_id: str,
        body: ConfirmationBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            value = manager.force_stop(
                instance_id,
                confirmation=body.confirmation,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"instance": value}

    @app.post("/api/instances/{instance_id}/kill")
    def kill_instance(
        instance_id: str,
        body: ConfirmationBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            manager.kill_instance(
                instance_id,
                confirmation=body.confirmation,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"deleted": True}

    @app.delete("/api/instances/{instance_id}")
    def delete_instance(
        instance_id: str,
        body: ConfirmationBody,
        request: Request,
        session=Depends(protected),
    ):
        try:
            manager.delete_instance(
                instance_id,
                confirmation=body.confirmation,
                actor=session["username"],
                client_ip=_client_ip(request),
            )
        except InstanceError as exc:
            translate_instance_error(exc)
        return {"deleted": True}

    @app.get("/api/resources")
    def resources(session=Depends(current_session)):
        return manager.resource_snapshot()

    @app.get("/api/audit")
    def audit_log(session=Depends(current_session)):
        rows = store.recent_audit(100)
        for row in rows:
            row.pop("client_ip", None)
        return {"events": rows}

    @app.websocket("/ws/instances/{instance_id}")
    async def terminal_socket(websocket: WebSocket, instance_id: str):
        origin = websocket.headers.get("origin", "")
        if origin not in config.allowed_origins:
            await websocket.close(code=4403)
            return
        raw_session_token = websocket.cookies.get(config.cookie_name)
        session = await _websocket_session(auth, raw_session_token)
        if not session:
            await websocket.close(code=4401)
            return
        protocols = [
            value.strip()
            for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if value.strip()
        ]
        csrf_values = [value[5:] for value in protocols if value.startswith("csrf.")]
        if (
            "aeon-v1" not in protocols
            or len(csrf_values) != 1
            or not hmac.compare_digest(csrf_values[0], session["csrf_token"])
        ):
            await websocket.close(code=4403)
            return
        try:
            prepared_attachment = await asyncio.to_thread(
                manager.prepare_terminal_attachment, instance_id
            )
        except InstanceError:
            await websocket.close(code=4404)
            return
        await websocket.accept(subprotocol="aeon-v1")
        try:
            await asyncio.to_thread(
                store.audit,
                "terminal_attached",
                actor=session["username"],
                instance_id=instance_id,
                client_ip=websocket.client.host if websocket.client else "",
            )
        except Exception:
            await websocket.close(code=1011)
            return

        async def revalidate_session() -> None:
            while True:
                await asyncio.sleep(WEBSOCKET_SESSION_RECHECK_SECONDS)
                valid = await _websocket_session(auth, raw_session_token)
                if valid:
                    continue
                try:
                    await websocket.close(code=4401)
                except Exception:
                    pass
                return

        bridge_task = asyncio.create_task(
            bridge_terminal(
                websocket,
                manager,
                instance_id,
                prepared=prepared_attachment,
            )
        )
        session_task = asyncio.create_task(revalidate_session())
        done, pending = await asyncio.wait(
            {bridge_task, session_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*done, return_exceptions=True)

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        if request.url.path.startswith(("/api/", "/__")):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return FileResponse(static_dir / "index.html")

    return app
