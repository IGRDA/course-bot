"""Conversation types — ConversationRequest and WorkspaceInfo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationRequest:
    """Represents an incoming conversation event.

    Mirrors agents ConversationRequest. The workspace_id is derived from
    conversation_id (channel) and thread_id (thread timestamp) to
    deterministically map each Slack thread to a unique workspace.
    """

    conversation_id: str  # Slack channel ID (e.g. C12345)
    message_id: str  # Message timestamp (ts)
    thread_id: str  # Thread timestamp (thread_ts or ts for new threads)
    text: str  # User message text
    requester_id: str  # Slack user ID

    @property
    def workspace_id(self) -> str:
        """Deterministic workspace identifier: 'channelID:threadID'."""
        return f"{self.conversation_id}:{self.thread_id}"


@dataclass(frozen=True)
class WorkspaceInfo:
    """Describes a ready-to-use workspace for a conversation session.

    Returned by ConversationHandler.handle() after the workspace has been
    set up (cloned, branched, synced).
    """

    workspace_id: str
    workspace_dir: str
    branch_name: str
