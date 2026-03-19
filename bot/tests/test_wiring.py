"""Smoke tests that verify constructor call sites match actual signatures.

These catch bugs like passing keyword arguments that a class no longer
accepts (e.g. the `conversation_handler=None` bug that broke both
main.py and worker.py in production).
"""

from __future__ import annotations

import ast
import inspect
import importlib
from pathlib import Path
from typing import Any

import pytest


APP_ROOT = Path(__file__).resolve().parent.parent


def _get_init_params(cls: type) -> set[str]:
    """Return the set of parameter names accepted by cls.__init__ (excluding self)."""
    sig = inspect.signature(cls.__init__)
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _find_constructor_calls(source: str, class_name: str) -> list[tuple[int, set[str]]]:
    """Parse source and find all `ClassName(...)` calls, returning (line, kwarg_names)."""
    tree = ast.parse(source)
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name) and func.id == class_name:
            name = func.id
        elif isinstance(func, ast.Attribute) and func.attr == class_name:
            name = func.attr
        if name:
            kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
            results.append((node.lineno, kwargs))
    return results


class TestSlackServiceWiring:
    """Verify every SlackService() call passes only accepted kwargs."""

    def test_init_signature_is_known(self):
        from bot.internal.slack_service import SlackService
        params = _get_init_params(SlackService)
        assert "platform" in params
        assert "bot_user_id" in params
        assert "claude_client" in params

    @pytest.mark.parametrize("module_path", [
        APP_ROOT / "main.py",
        APP_ROOT / "job" / "worker.py",
    ])
    def test_call_sites_match_signature(self, module_path: Path):
        from bot.internal.slack_service import SlackService
        accepted = _get_init_params(SlackService)

        source = module_path.read_text()
        calls = _find_constructor_calls(source, "SlackService")
        assert calls, f"No SlackService() calls found in {module_path.name}"

        for lineno, kwargs in calls:
            unexpected = kwargs - accepted
            assert not unexpected, (
                f"{module_path.name}:{lineno} passes unexpected kwargs "
                f"to SlackService: {unexpected}. Accepted: {accepted}"
            )


class TestJobDispatcherWiring:
    """Verify every JobDispatcher() call passes only accepted kwargs."""

    def test_call_sites_match_signature(self):
        from bot.job.dispatcher import JobDispatcher
        accepted = _get_init_params(JobDispatcher)

        source = (APP_ROOT / "main.py").read_text()
        calls = _find_constructor_calls(source, "JobDispatcher")
        assert calls, "No JobDispatcher() calls found in main.py"

        for lineno, kwargs in calls:
            unexpected = kwargs - accepted
            assert not unexpected, (
                f"main.py:{lineno} passes unexpected kwargs "
                f"to JobDispatcher: {unexpected}. Accepted: {accepted}"
            )


class TestWorkerBootstrapImports:
    """Verify the worker module can be imported without crashing.

    This catches top-level syntax errors, broken imports, and
    missing modules before a Cloud Run Job execution wastes time.
    """

    def test_worker_module_importable(self):
        mod = importlib.import_module("bot.job.worker")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_process_event")
        assert callable(mod.main)

    def test_dispatcher_module_importable(self):
        mod = importlib.import_module("bot.job.dispatcher")
        assert hasattr(mod, "JobDispatcher")


class TestConfigWiring:
    """Verify Config can be instantiated with the required env vars."""

    def test_config_requires_env_vars(self):
        from bot.config import Config
        with pytest.raises(RuntimeError, match="SLACK_SIGNING_SECRET"):
            Config()

    def test_config_instantiates_with_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "test")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

        from bot.config import Config
        cfg = Config()
        assert cfg.slack_signing_secret == "test"
        assert cfg.gcp_project_id == "test-project"
        assert cfg.max_concurrent_jobs == 5
