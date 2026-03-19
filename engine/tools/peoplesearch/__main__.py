"""
CLI for testing people search tool.

Usage:
    python -m tools.peoplesearch "quantum physics" --max-results 5 --language es
    python -m tools.peoplesearch "international trade" --provider openai
"""

import argparse
import json
import logging
import sys

from .search import search_relevant_people


def main():
    parser = argparse.ArgumentParser(
        description="Search for relevant people on a topic using Wikipedia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.peoplesearch "quantum physics"
  python -m tools.peoplesearch "international trade policy" --max-results 3
  python -m tools.peoplesearch "machine learning" --language es --provider gemini
  python -m tools.peoplesearch "climate change" --json
        """
    )
    
    parser.add_argument(
        "topic",
        type=str,
        help="Topic to find relevant people for"
    )
    
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=5,
        help="Maximum number of people to return (default: 5)"
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code for descriptions (default: en)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="mistral",
        choices=["mistral", "gemini", "openai", "groq", "deepseek"],
        help="LLM provider for suggestions (default: mistral)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s"
    )
    
    if not args.json:
        print(f"\n🔍 Searching for people relevant to: '{args.topic}'")
        print(f"   Language: {args.language}, Provider: {args.provider}")
        print("-" * 60)
    
    try:
        people = search_relevant_people(
            topic=args.topic,
            max_results=args.max_results,
            language=args.language,
            llm_provider=args.provider,
        )
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        # Output as JSON
        output = [person.model_dump() for person in people]
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Human-readable output
        if not people:
            print("\n⚠️  No people found with Wikipedia images for this topic.")
            return
        
        print(f"\n📋 Found {len(people)} relevant people:\n")
        
        for i, person in enumerate(people, 1):
            print(f"{i}. {person.name}")
            print(f"   📝 {person.description[:150]}..." if len(person.description) > 150 else f"   📝 {person.description}")
            print(f"   🔗 {person.wikiUrl}")
            print(f"   🖼️  {person.image}")
            print()


if __name__ == "__main__":
    main()

