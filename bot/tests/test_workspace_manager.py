"""Tests for workspace manager — image-seed and clone-resume strategies."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bot.conversation.types import WorkspaceInfo
from bot.workspace.manager import WorkspaceManager


@pytest.fixture
def manager(tmp_path):
    image_dir = tmp_path / "image_source"
    image_dir.mkdir()
    (image_dir / "engine").mkdir()
    (image_dir / "engine" / "workflows").mkdir()
    (image_dir / "engine" / "CLAUDE.md").write_text("# Claude instructions")
    (image_dir / "bot").mkdir()
    (image_dir / "bot" / "main.py").write_text("app = ...")

    return WorkspaceManager(
        repo_url="https://github.com/test/repo.git",
        base_dir="/tmp/test-workspaces",
        default_branch="main",
        image_source_dir=str(image_dir),
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

        with (
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=True),
        ):
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

        with (
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=True),
        ):
            await manager.finalize(workspace_info)

    @pytest.mark.asyncio
    async def test_finalize_noop_if_dir_missing(self, manager):
        workspace_info = WorkspaceInfo(
            workspace_id="C123:1234.5678",
            workspace_dir="/tmp/does-not-exist",
            branch_name="session-abcd1234",
        )
        await manager.finalize(workspace_info)  # should not raise


class TestSetupImageSeed:
    """Tests for the image-seed path (new threads without a session branch)."""

    @pytest.mark.asyncio
    async def test_seeds_from_image_when_no_session_branch(self, manager):
        """When ls-remote finds no session branch, workspace is seeded from
        the Docker image instead of cloned."""
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "ls-remote" in args:
                return (0, "", "")  # no matching branch in output
            if "fetch" in args and manager._default_branch in " ".join(args):
                return (0, "", "")
            if "reset" in args:
                return (0, "", "")
            return (0, "", "")

        with (
            patch.object(manager, "_run_git", side_effect=mock_git),
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=False),
            patch("shutil.copytree") as mock_copytree,
        ):
            info = await manager.setup("C123:1234.5678")

        assert info.branch_name.startswith("session-")
        mock_copytree.assert_called_once()

        commands = [" ".join(c) for c in git_calls]
        assert not any("clone" in c for c in commands), "Should NOT clone when seeding from image"
        assert any("init" in c for c in commands)
        assert any("remote" in c and "add" in c and "origin" in c for c in commands)

    @pytest.mark.asyncio
    async def test_seed_grafts_onto_remote_main(self, manager):
        """When fetch succeeds, image code is grafted onto origin/main via
        ``git reset --soft`` for clean MR diffs."""
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "ls-remote" in args:
                return (0, "", "")  # no session branch
            return (0, "", "")  # fetch + reset both succeed

        with (
            patch.object(manager, "_run_git", side_effect=mock_git),
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=False),
            patch("shutil.copytree"),
        ):
            await manager.setup("C123:new-thread")

        commands = [" ".join(c) for c in git_calls]
        assert any("reset" in c and "--soft" in c for c in commands), "Should graft onto origin/main with reset --soft"

    @pytest.mark.asyncio
    async def test_seed_works_when_fetch_fails(self, manager):
        """If the remote is unreachable, the workspace is still usable —
        it just won't have upstream history for MR diffs."""
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "ls-remote" in args:
                return (0, "", "")  # no session branch
            if "fetch" in args:
                return (128, "", "fatal: unable to access")
            return (0, "", "")

        with (
            patch.object(manager, "_run_git", side_effect=mock_git),
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=False),
            patch("shutil.copytree"),
        ):
            info = await manager.setup("C123:offline-thread")

        assert info.branch_name.startswith("session-")
        commands = [" ".join(c) for c in git_calls]
        assert not any("reset" in c and "--soft" in c for c in commands), "Should skip reset --soft when fetch fails"
        assert any("commit" in c for c in commands), "Should still commit the seed"

    @pytest.mark.asyncio
    async def test_seed_creates_session_branch(self, manager):
        """The image-seed path must create the session branch."""
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "ls-remote" in args:
                return (0, "", "")
            return (0, "", "")

        with (
            patch.object(manager, "_run_git", side_effect=mock_git),
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=False),
            patch("shutil.copytree"),
        ):
            info = await manager.setup("C123:1234.5678")

        commands = [" ".join(c) for c in git_calls]
        assert any("checkout" in c and "-b" in c and info.branch_name in c for c in commands)


class TestSetupCloneResume:
    """Tests for the clone-resume path (existing session branch on remote)."""

    @pytest.mark.asyncio
    async def test_clones_when_session_branch_exists(self, manager):
        """When ls-remote finds the session branch, workspace is cloned
        from the remote (not seeded from image)."""
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "ls-remote" in args:
                branch = args[-1]
                return (0, f"abc123\trefs/heads/{branch}\n", "")
            return (0, "", "")

        with (
            patch.object(manager, "_run_git", side_effect=mock_git),
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=False),
            patch("shutil.copytree") as mock_copytree,
        ):
            info = await manager.setup("C123:1234.5678")

        mock_copytree.assert_not_called()

        commands = [" ".join(c) for c in git_calls]
        assert any("clone" in c for c in commands)
        assert any("checkout" in c and info.branch_name in c for c in commands)

    @pytest.mark.asyncio
    async def test_clone_resume_fetches_default_branch(self, manager):
        """The clone-resume path fetches both the default branch and the
        session branch."""
        git_calls = []

        async def mock_git(*args):
            git_calls.append(args)

        async def mock_git_checked(*args):
            git_calls.append(args)
            if "ls-remote" in args:
                branch = args[-1]
                return (0, f"abc123\trefs/heads/{branch}\n", "")
            return (0, "", "")

        with (
            patch.object(manager, "_run_git", side_effect=mock_git),
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=False),
        ):
            await manager.setup("C123:1234.5678")

        commands = [" ".join(c) for c in git_calls]
        assert any("fetch" in c and f"origin/{manager._default_branch}" in c for c in commands), (
            "Should fetch the default branch tracking ref"
        )


class TestSetupSync:
    """Tests for the sync path (workspace dir already exists on disk)."""

    @pytest.mark.asyncio
    async def test_sync_when_dir_exists(self, manager):
        """When the workspace directory already exists, sync is used
        instead of create."""

        async def mock_git_checked(*args):
            if "status" in args:
                return (0, "", "")  # clean
            return (0, "", "")

        with (
            patch.object(manager, "_run_git_checked", side_effect=mock_git_checked),
            patch("os.path.isdir", return_value=True),
        ):
            info = await manager.setup("C123:1234.5678")

        assert info.branch_name.startswith("session-")
