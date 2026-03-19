"""
CLI for testing article search providers.

Usage:
    python -m tools.articlesearch "quantum computing" --provider semanticscholar --max-results 5
    python -m tools.articlesearch "machine learning" --provider openalex --language es
    python -m tools.articlesearch "neural networks" --provider arxiv
    python -m tools.articlesearch "deep learning" --provider all --max-results 3
"""

import argparse
import json
import sys

from .factory import (
    create_article_search,
    available_article_search_providers,
    ArticleResult,
)


def format_article(article: ArticleResult, index: int) -> str:
    """Format a single article result for display."""
    lines = [
        f"\n{index}. {article['title']}",
        f"   Authors: {', '.join(article['authors'][:4])}{'...' if len(article['authors']) > 4 else ''}",
        f"   Year: {article['year'] or 'N/A'}",
    ]
    
    if article['citation_count'] is not None:
        lines.append(f"   Citations: {article['citation_count']}")
    
    if article['venue']:
        lines.append(f"   Venue: {article['venue']}")
    
    if article['language']:
        lines.append(f"   Language: {article['language']}")
    
    if article['doi']:
        lines.append(f"   DOI: {article['doi']}")
    
    lines.append(f"   URL: {article['url']}")
    lines.append(f"   Source: {article['source']}")
    
    if article['snippet']:
        # Truncate snippet for display
        snippet = article['snippet'][:150] + "..." if len(article['snippet'] or "") > 150 else article['snippet']
        lines.append(f"   Snippet: {snippet}")
    
    return "\n".join(lines)


def search_single_provider(
    query: str,
    provider: str,
    max_results: int,
    language: str | None,
    output_json: bool,
) -> list[ArticleResult]:
    """Search using a single provider."""
    print(f"\n🔍 Searching {provider} for: '{query}'")
    if language:
        print(f"   Language filter: {language}")
    print("-" * 60)
    
    try:
        search_fn = create_article_search(provider)
        results = search_fn(query, max_results, language)
        
        if not results:
            print(f"   No results found from {provider}")
            return []
        
        if output_json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            for i, article in enumerate(results, 1):
                print(format_article(article, i))
        
        print(f"\n   ✓ Found {len(results)} results from {provider}")
        return results
        
    except Exception as e:
        print(f"   ❌ Error searching {provider}: {e}")
        return []


def search_all_providers(
    query: str,
    max_results: int,
    language: str | None,
    output_json: bool,
) -> list[ArticleResult]:
    """Search across all available providers."""
    all_results: list[ArticleResult] = []
    providers = available_article_search_providers()
    
    print(f"\n🔍 Searching all providers ({', '.join(providers)}) for: '{query}'")
    if language:
        print(f"   Language filter: {language}")
    print("=" * 60)
    
    for provider in providers:
        results = search_single_provider(
            query, provider, max_results, language, output_json=False
        )
        all_results.extend(results)
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY: Found {len(all_results)} total results across {len(providers)} providers")
    
    if output_json:
        print("\n" + json.dumps(all_results, indent=2, ensure_ascii=False))
    
    return all_results


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Search for academic articles/papers across multiple providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tools.articlesearch "quantum computing" --provider semanticscholar
    python -m tools.articlesearch "machine learning" --provider openalex --language es
    python -m tools.articlesearch "neural networks" --provider arxiv --max-results 10
    python -m tools.articlesearch "deep learning" --provider all --json
        """,
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query string",
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["all"] + available_article_search_providers(),
        default="semanticscholar",
        help="Search provider to use (default: semanticscholar)",
    )
    
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=5,
        help="Maximum number of results per provider (default: 5)",
    )
    
    parser.add_argument(
        "--language", "-l",
        help="Filter by language (ISO 639-1 code, e.g., 'en', 'es', 'fr'). "
             "Only supported by OpenAlex.",
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )
    
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_providers:
        print("Available article search providers:")
        for provider in available_article_search_providers():
            print(f"  - {provider}")
        return 0
    
    if not args.query:
        parser.error("query is required when not using --list-providers")
    
    if args.provider == "all":
        results = search_all_providers(
            args.query,
            args.max_results,
            args.language,
            args.output_json,
        )
    else:
        results = search_single_provider(
            args.query,
            args.provider,
            args.max_results,
            args.language,
            args.output_json,
        )
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())

