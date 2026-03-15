"""Wikipedia search using LangChain.

langchain_community is imported lazily inside the search function so that
merely importing this module does not pull in that heavy package.
"""

import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search Wikipedia using LangChain.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)
        
    Returns:
        String containing Wikipedia search results
    """
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        # Initialize Wikipedia API wrapper with max_results
        wikipedia = WikipediaAPIWrapper(
            top_k_results=max_results,
            doc_content_chars_max=4000
        )
        search_tool = WikipediaQueryRun(api_wrapper=wikipedia)
        result = search_tool.run(query)
        return result
    except Exception as e:
        logger.error(
            "Wikipedia search failed for query '%s': %s",
            query,
            str(e)
        )
        return f"Wikipedia search failed: {str(e)}"


if __name__ == "__main__":
    query = "artificial inteligence"
    max_results = 10
    
    print(f"🔍 Searching Wikipedia for: '{query}'")
    print("-" * 80)
    
    result = web_search(query, max_results)
    
    print(f"\n📚 Wikipedia Results:\n")
    if result.startswith("Wikipedia search failed"):
        print(f"❌ {result}")
    else:
        print(result)
