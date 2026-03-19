"""
CLI for Gemini Image Generator Agent.

Usage:
    # Generate images for a module
    python -m agents.image_generator output/Quantum_Theory_Gemini/module_0.json
    
    # Specify output path
    python -m agents.image_generator module_0.json -o module_0_with_images.json
    
    # Use a specific model
    python -m agents.image_generator module_0.json --model gemini-2.5-flash-image
    
    # Dry run (extract queries only, no generation)
    python -m agents.image_generator module_0.json --dry-run
    
    # List available queries without generating
    python -m agents.image_generator module_0.json --list-queries
"""

import argparse
import json
import sys
from pathlib import Path

from .agent import (
    extract_image_queries,
    process_module_images,
    DEFAULT_MODEL,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images for a course module using Gemini Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate images for a module
    python -m agents.image_generator output/Quantum_Theory_Gemini/module_0.json
    
    # Dry run to see what would be generated
    python -m agents.image_generator module_0.json --dry-run
    
    # List all image queries in a module
    python -m agents.image_generator module_0.json --list-queries
        """,
    )
    
    parser.add_argument(
        "module_path",
        type=str,
        help="Path to the module JSON file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for updated JSON (default: overwrite input)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model for image generation (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract queries without generating images"
    )
    
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="List all image queries and exit"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (default: GEMINI_API_KEY env var)"
    )
    
    return parser.parse_args()


def list_queries(module_path: Path) -> int:
    """List all image queries in a module."""
    if not module_path.exists():
        print(f"❌ File not found: {module_path}")
        return 1
    
    with open(module_path, 'r', encoding='utf-8') as f:
        module_data = json.load(f)
    
    locations = extract_image_queries(module_data)
    
    if not locations:
        print("⚠️ No images with queries found in module")
        return 0
    
    print(f"📝 Found {len(locations)} image queries:\n")
    
    for i, loc in enumerate(locations, 1):
        print(f"{i:3d}. [{loc.request_key}]")
        print(f"     Location: submodule[{loc.submodule_idx}].section[{loc.section_idx}]")
        print(f"     Query: {loc.query}")
        print()
    
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    module_path = Path(args.module_path)
    
    # List queries mode
    if args.list_queries:
        return list_queries(module_path)
    
    try:
        # Process the module
        updated_module, images_dir = process_module_images(
            module_path=module_path,
            output_json_path=args.output,
            api_key=args.api_key,
            model=args.model,
            dry_run=args.dry_run
        )
        
        print("\n✨ Done!")
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())




