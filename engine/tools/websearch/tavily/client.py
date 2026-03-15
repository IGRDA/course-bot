"""Web search using Tavily.

langchain_tavily is imported lazily inside the search function so that
merely importing this module does not pull in that package.
"""

import os
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        
    Returns:
        String containing search results
    
    Note:
        Requires TAVILY_API_KEY environment variable.
    """
    try:
        if not os.environ.get("TAVILY_API_KEY"):
            logger.warning(
                "TAVILY_API_KEY environment variable is not set. "
                "Set it with: export TAVILY_API_KEY=your_key"
            )
            return "Error: TAVILY_API_KEY environment variable is not set."
        
        from langchain_tavily import TavilySearch

        search_tool = TavilySearch(max_results=max_results, topic="general")
        result = search_tool.invoke({"query": query})
        return result
    except Exception as e:
        logger.error(
            "Tavily search failed for query '%s': %s", 
            query, 
            str(e)
        )
        return f"Search failed: {str(e)}"
