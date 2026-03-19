"""CLI entry point for the Video HTML Generator Agent.

Usage:
    python -m agents.video_html_generator <input_json_path> [--output-dir <path>]

Examples:
    python -m agents.video_html_generator output/PDF_Course_20251226_204841/module_0.json
    python -m agents.video_html_generator output/Quantum_Theory_20251228_091909/module_0.json --output-dir ./custom_output
"""

import argparse
import json
import sys
from pathlib import Path

from .agent import simplify_module_from_path


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Simplify module JSON for video generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m agents.video_html_generator output/PDF_Course_20251226_204841/module_0.json
    python -m agents.video_html_generator module.json --output-dir ./video_html_generator
        """,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input module JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: video_html_generator/ in same folder as input)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print simplified JSON to stdout instead of saving",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    try:
        if args.preview:
            # Preview mode: just print to stdout
            with open(input_path, "r", encoding="utf-8") as f:
                module_data = json.load(f)
            
            from .agent import simplify_module
            simplified = simplify_module(module_data)
            print(json.dumps(simplified, indent=2, ensure_ascii=False))
        else:
            # Normal mode: save to file
            output_path = simplify_module_from_path(
                input_path,
                output_dir=args.output_dir,
            )
            print(f"✅ Simplified module saved to: {output_path}")
            
            # Print summary
            with open(output_path, "r", encoding="utf-8") as f:
                simplified = json.load(f)
            
            n_submodules = len(simplified.get("submodules", []))
            n_sections = sum(
                len(sm.get("sections", []))
                for sm in simplified.get("submodules", [])
            )
            print(f"   Module: {simplified.get('title', 'N/A')}")
            print(f"   Submodules: {n_submodules}")
            print(f"   Sections: {n_sections}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

