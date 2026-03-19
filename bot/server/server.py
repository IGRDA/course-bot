"""HTTP server with handler registration, health checks, and graceful shutdown.

"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from bot.slack.webhook_handler import SlackWebhookHandler

logger = logging.getLogger(__name__)

# Default shutdown timeout for waiting on background handlers.
DEFAULT_SHUTDOWN_TIMEOUT = 30.0


class AppServer:
    """HTTP server wrapping FastAPI with handler registration and graceful shutdown.

    Provides:
    - /health and /ready endpoints
    - register(handler) for self-registering webhook handlers
    - Graceful shutdown that waits for background tasks
    - Exception recovery middleware
    """

    def __init__(self, shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT) -> None:
        self._shutdown_timeout = shutdown_timeout
        self._handlers: list[Any] = []

        self._app = FastAPI(
            title="Course Bot",
            docs_url=None,
            redoc_url=None,
            lifespan=self._lifespan,
        )

        # Health and readiness endpoints
        self._app.add_api_route("/health", self._health, methods=["GET"])
        self._app.add_api_route("/ready", self._ready, methods=["GET"])

        # Exception recovery middleware
        self._app.middleware("http")(self._exception_recovery_middleware)

    @property
    def app(self) -> FastAPI:
        """Return the underlying FastAPI app (for uvicorn)."""
        return self._app

    def register(self, handler: Any) -> None:
        """Register a webhook handler.

        If the handler has a `router` attribute (e.g. SlackWebhookHandler),
        it is included as a FastAPI router.
        The handler is tracked for graceful shutdown.
        """
        self._handlers.append(handler)

        if hasattr(handler, "router"):
            self._app.include_router(handler.router)
            logger.info("Registered handler at path: %s", handler.path())
        else:
            logger.warning(
                "Handler %s has no router attribute — skipped route registration",
                type(handler).__name__,
            )

    # -- Lifespan (startup / shutdown) --------------------------------------

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage server lifecycle — setup signal handlers, cleanup on shutdown."""
        logger.info("Server starting up...")

        # Install SIGTERM handler for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)

        yield

        # Shutdown: wait for all registered handlers to finish
        logger.info("Server shutting down — waiting for handlers...")
        for handler in self._handlers:
            if hasattr(handler, "shutdown"):
                try:
                    await handler.shutdown(timeout=self._shutdown_timeout)
                except Exception:
                    logger.warning(
                        "Error shutting down handler %s",
                        type(handler).__name__,
                        exc_info=True,
                    )
        logger.info("Server shutdown complete.")

    def _handle_signal(self, sig: signal.Signals) -> None:
        logger.info("Received signal %s — initiating shutdown", sig.name)

    # -- Endpoints ----------------------------------------------------------

    async def _health(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _ready(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # -- Middleware ----------------------------------------------------------

    async def _exception_recovery_middleware(
        self,
        request: Request,
        call_next: Any,
    ) -> Any:
        """Catch unhandled exceptions and return 500 instead of crashing."""
        try:
            return await call_next(request)
        except Exception:
            logger.error(
                "Unhandled exception in %s %s",
                request.method,
                request.url.path,
                exc_info=True,
            )
            return JSONResponse(
                {"error": "internal server error"},
                status_code=500,
            )
