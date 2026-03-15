"""Workspace manager — git clone, branch creation, sync, and finalize.

Each Slack thread gets a deterministic workspace directory and git branch
derived from the workspace ID (channel:thread_ts).

With Cloud Run Jobs, each execution gets a fresh container.  The
``finalize()`` method commits any uncommitted changes and pushes the
session branch so the next execution can continue where this one left off.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os

from bot.conversation.types import WorkspaceInfo

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages per-session git workspaces.

    - Deterministic hash: sha1(workspace_id)[:8]
    - Clone on first use, fetch + checkout existing session branch if available
    - finalize() commits + pushes after processing
    """

    def __init__(
        self,
        repo_url: str,
        base_dir: str = "/tmp",
        default_branch: str = "main",
    ) -> None:
        self._repo_url = repo_url
        self._base_dir = base_dir
        self._default_branch = default_branch

    async def setup(self, workspace_id: str) -> WorkspaceInfo:
        """Set up a workspace for the given ID.

        Creates a git workspace with an isolated session branch.  If the
        session branch exists on the remote (from a previous execution),
        it is checked out to preserve continuity.
        """
        hash_str = self._hash_workspace_id(workspace_id)
        branch_name = f"session-{hash_str}"
        workspace_dir = os.path.join(self._base_dir, f"session-{hash_str}")

        if os.path.isdir(workspace_dir):
            await self._sync_workspace(workspace_dir, branch_name)
        else:
            await self._create_workspace(workspace_dir, branch_name)

        logger.info(
            "Workspace ready: workspace_id=%s branch=%s dir=%s",
            workspace_id,
            branch_name,
            workspace_dir,
        )

        return WorkspaceInfo(
            workspace_id=workspace_id,
            workspace_dir=workspace_dir,
            branch_name=branch_name,
        )

    async def finalize(self, workspace_info: WorkspaceInfo) -> None:
        """Commit uncommitted changes and push the session branch.

        Called after processing completes so the next execution for this
        thread can continue from the same state.  Errors are logged but
        do not propagate — finalization is best-effort.
        """
        workspace_dir = workspace_info.workspace_dir
        branch_name = workspace_info.branch_name

        if not os.path.isdir(workspace_dir):
            return

        try:
            rc, stdout, _ = await self._run_git_checked(
                "-C", workspace_dir, "status", "--porcelain",
            )
            if rc != 0:
                return

            if stdout.strip():
                await self._run_git_checked(
                    "-C", workspace_dir, "add", "-A",
                )
                await self._run_git_checked(
                    "-C", workspace_dir,
                    "commit", "-m", f"Auto-commit from {branch_name}",
                )
                logger.info("Auto-committed changes in %s", workspace_dir)

            rc, _, stderr = await self._run_git_checked(
                "-C", workspace_dir,
                "push", "origin", branch_name, "--force-with-lease",
            )
            if rc == 0:
                logger.info("Pushed session branch %s", branch_name)
            else:
                logger.warning(
                    "Failed to push session branch %s: %s",
                    branch_name,
                    stderr.strip(),
                )
        except BaseException:
            logger.warning(
                "Workspace finalization failed for %s",
                workspace_dir,
                exc_info=True,
            )

    # -- Hashing ------------------------------------------------------------

    @staticmethod
    def _hash_workspace_id(workspace_id: str) -> str:
        return hashlib.sha1(workspace_id.encode()).hexdigest()[:8]

    # -- Git operations -----------------------------------------------------

    async def _create_workspace(self, workspace_dir: str, branch_name: str) -> None:
        """Clone the repository and check out the session branch.

        If the session branch exists on the remote (from a previous
        execution), it is checked out.  Otherwise a new branch is
        created from origin/default_branch.
        """
        logger.info("Creating workspace: dir=%s branch=%s", workspace_dir, branch_name)

        await self._run_git(
            "clone", "--depth", "20", self._repo_url, workspace_dir,
        )

        await self._run_git(
            "-C", workspace_dir,
            "fetch", "origin",
            f"{self._default_branch}:refs/remotes/origin/{self._default_branch}",
        )

        # Try to check out the remote session branch (continuity from previous job)
        rc, _, _ = await self._run_git_checked(
            "-C", workspace_dir,
            "fetch", "origin", f"{branch_name}:refs/remotes/origin/{branch_name}",
        )
        if rc == 0:
            await self._run_git(
                "-C", workspace_dir,
                "checkout", "-b", branch_name, f"origin/{branch_name}",
            )
            logger.info("Resumed existing session branch: %s", branch_name)
        else:
            await self._run_git(
                "-C", workspace_dir,
                "checkout", "-b", branch_name, f"origin/{self._default_branch}",
            )

    async def _sync_workspace(self, workspace_dir: str, branch_name: str) -> None:
        """Sync an existing workspace with remote changes."""
        logger.info("Syncing workspace: dir=%s branch=%s", workspace_dir, branch_name)

        returncode, stdout, _ = await self._run_git_checked(
            "-C", workspace_dir, "status", "--porcelain",
        )
        if returncode != 0:
            logger.warning("git status failed in %s — skipping sync", workspace_dir)
            return

        if stdout.strip():
            logger.info("Workspace %s has uncommitted changes — skipping sync", workspace_dir)
            return

        await self._run_git_checked(
            "-C", workspace_dir, "checkout", branch_name,
        )

        returncode, _, stderr = await self._run_git_checked(
            "-C", workspace_dir, "fetch", "origin",
        )
        if returncode != 0:
            logger.warning("git fetch failed in %s: %s", workspace_dir, stderr)
            return

        returncode, _, stderr = await self._run_git_checked(
            "-C", workspace_dir,
            "rebase", "--autostash", f"origin/{self._default_branch}",
        )
        if returncode != 0:
            logger.warning("Rebase failed in %s — aborting and resetting", workspace_dir)
            await self._run_git_checked("-C", workspace_dir, "rebase", "--abort")
            await self._run_git_checked(
                "-C", workspace_dir,
                "reset", "--hard", f"origin/{self._default_branch}",
            )

    async def _run_git(self, *args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            cmd_str = " ".join(["git", *args])
            raise RuntimeError(
                f"git command failed (exit {proc.returncode}): {cmd_str}\n"
                f"stderr: {stderr.decode()}"
            )

    async def _run_git_checked(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout.decode(), stderr.decode()
