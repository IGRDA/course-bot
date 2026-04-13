"""
Utility functions shared across workflow nodes.
"""

from langchain_core.runnables import RunnableConfig

from workflows.output_manager import OutputManager


def get_output_manager(config: RunnableConfig | None) -> OutputManager | None:
    """Extract OutputManager from LangGraph config if present.

    Args:
        config: LangGraph RunnableConfig containing configurable options.

    Returns:
        OutputManager instance if found in config, None otherwise.
    """
    if config is None:
        return None
    return config.get("configurable", {}).get("output_manager")
