"""
Web search using DuckDuckGo (region-aware).

langchain_community is imported lazily inside the search function so that
merely importing this module does not pull in that heavy package.
"""

import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def web_search(
    query: str,
    max_results: int = 5,
    region: str = "us-en",
) -> str:
    """
    Search the web using DuckDuckGo with region and language bias.

    Args:
        query: Search query string
        max_results: Maximum number of results
        region: DuckDuckGo region (e.g. 'us-en', 'es-es', 'uk-en')

    Returns:
        String containing search results
    """
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

        search = DuckDuckGoSearchAPIWrapper(
            region=region,
            safesearch="moderate",
            max_results=max_results,
        )

        return search.run(query)

    except Exception as e:
        logger.error(
            "DuckDuckGo search failed for query '%s' (region=%s): %s",
            query,
            region,
            str(e),
        )
        return f"Search failed: {str(e)}"


if __name__ == "__main__":
    query = "artificial intelligence"
    max_results = 10
    region = "us-en"  # change to "es-es" for Spain

    print(f"🔍 Searching DuckDuckGo for: '{query}' (region={region})")
    print("-" * 80)

    result = web_search(query, max_results=max_results, region=region)

    print("\n🦆 DuckDuckGo Results:\n")
    if result.startswith("Search failed"):
        print(f"❌ {result}")
    else:
        print(result)
