"""Thin wrapper around claude-agent-sdk for response generation.

Uses Claude Code CLI (via the SDK) pointed at a directory containing
CLAUDE.md with the bot's instructions. The CLI reads CLAUDE.md
automatically and uses CLAUDE_CODE_OAUTH_TOKEN from the environment.

Claude has access to file-reading tools so it can examine documents
that users share in Slack (downloaded to local paths by SlackService).
"""

from __future__ import annotations

import asyncio
import logging
import os

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Generates responses via Claude Code with file-reading capability."""

    def __init__(self, agent_cwd: str, response_timeout: float = 3600.0) -> None:
        self._cwd = agent_cwd
        self._response_timeout = response_timeout
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        logger.info(
            "ClaudeClient init: cwd=%s token_present=%s token_len=%d response_timeout=%.0fs",
            agent_cwd,
            bool(token),
            len(token),
            response_timeout,
        )

    @property
    def cwd(self) -> str:
        """Return the working directory used by the Claude agent."""
        return self._cwd

    @staticmethod
    def _log_tool_result(block: ToolResultBlock) -> None:
        """Extract and log text from a ToolResultBlock.

        The block's ``content`` can be a plain string, a list of typed
        content dicts (e.g. ``[{"type": "text", "text": "..."}]``), or
        None.  We normalise all three forms so the actual output text
        reaches the application logs.
        """
        raw = block.content
        if raw is None:
            return
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, list):
            text = "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in raw)
        else:
            text = str(raw)

        for line in text.splitlines():
            if line.strip():
                logger.info("Claude tool_output: %s", line)

    async def generate_response(
        self,
        user_text: str,
        cwd: str | None = None,
    ) -> str:
        """Send the user's message to Claude and return a response.

        Wraps the SDK call with an overall timeout to prevent indefinite
        hangs from blocking tool calls (TaskOutput, long Bash commands,
        etc.).  Without this, a single stuck tool turn can block the
        entire response pipeline and the user never gets a reply.

        Args:
            user_text: The prompt to send to Claude.
            cwd: Working directory for this invocation.  When provided the
                 Claude agent runs inside this directory instead of the
                 default ``agent_cwd``.  This is essential for per-thread
                 workspace isolation so concurrent requests don't share
                 the same filesystem state.
        """
        try:
            return await asyncio.wait_for(
                self._execute_query(user_text, cwd),
                timeout=self._response_timeout,
            )
        except TimeoutError:
            logger.error(
                "Claude response timed out after %.0fs for prompt: %s",
                self._response_timeout,
                user_text[:200],
            )
            return (
                "Sorry, the request timed out after "
                f"{int(self._response_timeout // 60)} minutes. "
                "Please try a simpler request or break it into smaller steps."
            )

    async def _execute_query(
        self,
        user_text: str,
        cwd: str | None = None,
    ) -> str:
        """Run the Claude Code SDK query and collect the response."""
        effective_cwd = cwd or self._cwd
        options = ClaudeAgentOptions(
            max_turns=200,
            allowed_tools=[
                "Read",
                "Write",
                "Edit",
                "List",
                "Search",
                "Bash",
                "WebFetch",
                "Skill",
            ],
            setting_sources=["project"],
            permission_mode="acceptEdits",
            cwd=effective_cwd,
            model="sonnet",
            env={
                "BASH_DEFAULT_TIMEOUT_MS": "5400000",
                "BASH_MAX_TIMEOUT_MS": "5400000",
            },
            stderr=lambda line: logger.info("Claude CLI: %s", line),
        )
        logger.info(
            "Claude invocation starting: cwd=%s (override=%s) timeout=%.0fs",
            effective_cwd,
            cwd is not None,
            self._response_timeout,
        )
        parts: list[str] = []
        try:
            async for message in query(prompt=user_text, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            logger.info(
                                "Claude tool_use: %s — input=%s",
                                block.name,
                                str(getattr(block, "input", ""))[:300],
                            )
                elif isinstance(message, UserMessage):
                    blocks = message.content if isinstance(message.content, list) else []
                    for block in blocks:
                        if isinstance(block, ToolResultBlock):
                            self._log_tool_result(block)
                elif isinstance(message, ResultMessage):
                    logger.info(
                        "Claude result: turns=%d duration=%dms cost=$%s",
                        message.num_turns,
                        message.duration_ms,
                        message.total_cost_usd,
                    )
                    break
                else:
                    logger.debug("Claude message: type=%s", type(message).__name__)
        except Exception as exc:
            if parts:
                logger.warning(
                    "Claude CLI exited with error after producing output, using collected response: %s — %s",
                    type(exc).__name__,
                    str(exc)[:200],
                )
            else:
                logger.exception(
                    "Claude response generation failed with no output: %s — %s",
                    type(exc).__name__,
                    str(exc)[:500],
                )
                return "Hello!"

        result = "".join(parts) or "Hello!"

        if "authentication_error" in result or "Invalid bearer token" in result:
            logger.error("Claude returned an auth error as text: %s", result[:200])
            return "Hello! (I'm having trouble connecting to my AI brain right now.)"

        logger.info("Claude response result length=%d", len(result))
        return result
