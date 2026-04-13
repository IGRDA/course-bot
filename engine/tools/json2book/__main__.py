"""
CLI entry point for PDF book generation.

Usage:
    python -m tools.json2book path/to/course.json [--template academic] [--output-dir path/to/output]
"""

import argparse
import sys
from pathlib import Path

from .factory import available_templates
from .generator import generate_pdf_book


def main():
    parser = argparse.ArgumentParser(
        description="Generate a PDF book from course.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate PDF in default location (course_dir/book/)
    python -m tools.json2book output/MyCourseName/course.json

    # Generate PDF with specific output directory
    python -m tools.json2book output/MyCourseName/course.json --output-dir ./my_book

    # Use a different template
    python -m tools.json2book output/MyCourseName/course.json --template academic

    # Skip image downloads
    python -m tools.json2book output/MyCourseName/course.json --no-images

    # Keep intermediate LaTeX files
    python -m tools.json2book output/MyCourseName/course.json --no-cleanup
""",
    )

    parser.add_argument("course_json", type=str, help="Path to course.json file")

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for PDF and intermediate files (default: course_dir/book/)",
    )

    parser.add_argument(
        "--template",
        "-t",
        type=str,
        default="academic",
        choices=available_templates(),
        help=f"LaTeX template to use (default: academic). Available: {', '.join(available_templates())}",
    )

    parser.add_argument("--no-images", action="store_true", help="Skip downloading images from URLs")

    parser.add_argument("--no-cleanup", action="store_true", help="Keep intermediate LaTeX files (.aux, .log, etc.)")

    args = parser.parse_args()

    # Validate input file
    course_path = Path(args.course_json)
    if not course_path.exists():
        print(f"❌ Error: Course file not found: {course_path}")
        sys.exit(1)

    if course_path.suffix != ".json":
        print(f"⚠️  Warning: Expected .json file, got: {course_path.suffix}")

    try:
        pdf_path = generate_pdf_book(
            course_json_path=course_path,
            output_dir=args.output_dir,
            template=args.template,
            download_images=not args.no_images,
            cleanup=not args.no_cleanup,
        )

        print(f"\n📗 Book generated: {pdf_path}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
