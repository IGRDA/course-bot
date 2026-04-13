"""Conversation handler — maps threads to workspaces.

Each Slack thread (identified by workspace_id = channel:thread_ts) gets
a dedicated git workspace via WorkspaceManager.

Concurrency control is handled externally by the API server's capacity
check (listing running Cloud Run Job executions), so no per-thread
locking is needed here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from bot.conversation.types import ConversationRequest, WorkspaceInfo
from bot.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class ConversationHandler:
    """Manages conversation sessions with workspace isolation."""

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self._workspace_manager = workspace_manager

    @asynccontextmanager
    async def handle(
        self,
        request: ConversationRequest,
    ) -> AsyncIterator[WorkspaceInfo]:
        workspace_id = request.workspace_id

        logger.info(
            "Handling conversation: workspace_id=%s requester=%s",
            workspace_id,
            request.requester_id,
        )

        workspace_info = await self._workspace_manager.setup(workspace_id)
        yield workspace_info
