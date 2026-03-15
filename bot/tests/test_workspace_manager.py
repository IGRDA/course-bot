"""Tests for workspace manager finalize and setup."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.conversation.types import WorkspaceInfo
from bot.workspace.manager import WorkspaceManager


@pytest.fixture
def manager():
    return WorkspaceManager(
        repo_url="https://github.com/test/repo.git",
        base_dir="/tmp/test-workspaces",
        default_branch="main",
    )


class TestFinalize:
    @pytest.mark.asyncio
    async def test_finalize_commits_and_pushes(self, manager):
        workspace_info = WorkspaceInfo(
            workspace_id="C123:1234.5678",
            workspace_dir="/tmp/test-workspaces/session-abcd1234",
            branch_name="session-abcd1234",
        )

        git_calls = []

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "status" in args:
                return (0, "M somefile.py\n", "")
            return (0, "", "")

        with patch.object(manager, "_run_git_checked", side_effect=mock_git_checked):
            with patch("os.path.isdir", return_value=True):
                await manager.finalize(workspace_info)

        commands = [" ".join(c) for c in git_calls]
        assert any("status" in c for c in commands)
        assert any("add -A" in c for c in commands)
        assert any("commit" in c for c in commands)
        assert any("push" in c for c in commands)

    @pytest.mark.asyncio
    async def test_finalize_skips_push_when_clean(self, manager):
        workspace_info = WorkspaceInfo(
            workspace_id="C123:1234.5678",
            workspace_dir="/tmp/test-workspaces/session-abcd1234",
            branch_name="session-abcd1234",
        )

        async def mock_git_checked(*args):
            if "status" in args:
                return (0, "", "")  # clean
            return (0, "", "")

        with patch.object(manager, "_run_git_checked", side_effect=mock_git_checked):
            with patch("os.path.isdir", return_value=True):
                await manager.finalize(workspace_info)

    @pytest.mark.asyncio
    async def test_finalize_noop_if_dir_missing(self, manager):
        workspace_info = WorkspaceInfo(
            workspace_id="C123:1234.5678",
            workspace_dir="/tmp/does-not-exist",
            branch_name="session-abcd1234",
        )
        await manager.finalize(workspace_info)  # should not raise


class TestSetup:
    @pytest.mark.asyncio
    async def test_setup_tries_remote_session_branch(self, manager):
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "fetch" in args and "session-" in " ".join(args):
                return (0, "", "")  # remote branch exists
            return (0, "", "")

        with patch.object(manager, "_run_git", side_effect=mock_git):
            with patch.object(manager, "_run_git_checked", side_effect=mock_git_checked):
                with patch("os.path.isdir", return_value=False):
                    info = await manager.setup("C123:1234.5678")

        assert info.branch_name.startswith("session-")
        commands = [" ".join(c) for c in git_calls]
        assert any("clone" in c for c in commands)

    @pytest.mark.asyncio
    async def test_setup_creates_new_branch_if_no_remote(self, manager):
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "fetch" in args and "session-" in " ".join(args):
                return (128, "", "fatal: couldn't find remote ref")
            return (0, "", "")

        with patch.object(manager, "_run_git", side_effect=mock_git):
            with patch.object(manager, "_run_git_checked", side_effect=mock_git_checked):
                with patch("os.path.isdir", return_value=False):
                    info = await manager.setup("C123:1234.5678")

        commands = [" ".join(c) for c in git_calls]
        assert any(f"origin/{manager._default_branch}" in c for c in commands)
