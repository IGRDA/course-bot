"""
Topic to PDF Book Workflow.

Generates a PDF book from a topic using a streamlined pipeline:
1. Index generation (course structure from topic)
2. Theory generation (section content)
3. Metadata calculation (IDs and indexes)
4. HTML formatting (needed for image placeholders)
5. Image generation (fetch images for sections)
6. Bibliography generation (book references)
7. PDF book generation (LaTeX compilation)

Skipped steps (not needed for PDF books):
- Activities generation (interactive elements don't work in PDF)
"""

from langgraph.graph import END, START, StateGraph

from workflows.nodes import (
    calculate_metadata_node,
    generate_bibliography_node,
    generate_html_node,
    generate_images_node,
    generate_index_node,
    generate_pdf_book_node,
    generate_theories_node,
)
from workflows.output_manager import OutputManager
from workflows.state import CourseConfig, CourseState


def build_book_generation_graph():
    """Build and return the PDF book generation graph.

    This graph has a streamlined pipeline for PDF book generation:
    1. generate_index - Creates course structure from topic
    2. generate_theories - Generates section content
    3. calculate_metadata - Sets IDs and indexes
    4. generate_html - Creates HTML elements (needed for image placeholders)
    5. generate_images - Fetches images for sections (included in PDF)
    6. generate_bibliography - Generates book references (fits academic PDFs)
    7. generate_pdf_book - Compiles PDF via LaTeX

    Skipped steps:
    - generate_activities (not useful in PDF format)
    """
    graph = StateGraph(CourseState)

    # Add nodes for PDF book generation pipeline
    graph.add_node("generate_index", generate_index_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_html", generate_html_node)
    graph.add_node("generate_images", generate_images_node)
    graph.add_node("generate_bibliography", generate_bibliography_node)
    graph.add_node("generate_pdf_book", generate_pdf_book_node)

    # Add edges for sequential execution
    graph.add_edge(START, "generate_index")
    graph.add_edge("generate_index", "generate_theories")
    graph.add_edge("generate_theories", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_html")
    graph.add_edge("generate_html", "generate_images")
    graph.add_edge("generate_images", "generate_bibliography")
    graph.add_edge("generate_bibliography", "generate_pdf_book")
    graph.add_edge("generate_pdf_book", END)

    return graph.compile()


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate a PDF book from a topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate a 10-page PDF book
    python -m main.workflow_book --total-pages 10

    # Generate in Spanish with different provider
    python -m main.workflow_book --total-pages 20 --language "Español" --provider gemini

    # Generate with custom title
    python -m main.workflow_book --title "Machine Learning Fundamentals" --total-pages 30
""",
    )
    parser.add_argument(
        "--title", type=str, default="Quantum Theory", help="Title/topic for the course (default: Quantum Theory)"
    )
    parser.add_argument("--total-pages", type=int, default=10, help="Total pages for the book (default: 10)")
    parser.add_argument(
        "--language", type=str, default="English", help="Language for content generation (default: English)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mistral",
        choices=["mistral", "gemini", "groq", "openai", "deepseek"],
        help="LLM provider (default: mistral)",
    )
    parser.add_argument(
        "--template", type=str, default="academic", help="LaTeX template for PDF generation (default: academic)"
    )
    parser.add_argument("--no-images", action="store_true", help="Skip image generation (faster, text-only PDF)")
    parser.add_argument("--no-bibliography", action="store_true", help="Skip bibliography generation")
    parser.add_argument(
        "--vision-ranking",
        action="store_true",
        help="Use Pixtral vision LLM to rank and select best images (slower but higher quality)",
    )
    args = parser.parse_args()

    from LLMs.text2text.health_check import validate_provider_keys

    validate_provider_keys(args.provider)

    # Build the graph
    app = build_book_generation_graph()

    # Create initial CourseState with config optimized for PDF book generation
    course_config = CourseConfig(
        title=args.title,
        text_llm_provider=args.provider,
        web_search_provider="ddg",
        total_pages=args.total_pages,
        words_per_page=400,
        language=args.language,
        description="",
        max_retries=8,
        concurrency=20,
        use_reflection=True,
        num_reflection_queries=5,
        # Research configuration (enabled for better content)
        enable_research=True,
        research_max_queries=5,
        research_max_results_per_query=5,
        # Image configuration
        generate_images=not args.no_images,
        image_search_provider="freepik",
        use_vision_ranking=args.vision_ranking,
        num_images_to_fetch=8,
        vision_llm_provider="pixtral",
        image_concurrency=5,
        imagetext2text_concurrency=5,
        vision_ranking_batch_size=8,
        # Bibliography configuration (enabled by default for books)
        generate_bibliography=not args.no_bibliography,
        bibliography_books_per_module=5,
        bibliography_articles_per_module=5,
        target_audience=None,
    )

    initial_state = CourseState(config=course_config, title=course_config.title, modules=[])

    # Create OutputManager for this run
    output_mgr = OutputManager(title=course_config.title)
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")
    print(f"📚 Generating PDF book: {course_config.title}")
    print(f"   Pages: {course_config.total_pages}")
    print(f"   Language: {course_config.language}")
    print(f"   Provider: {course_config.text_llm_provider}")
    print(f"   Bibliography: {'enabled' if not args.no_bibliography else 'disabled'}")
    print(f"   Vision ranking: {'enabled (Pixtral)' if args.vision_ranking else 'disabled'}")

    # Run the graph with OutputManager passed via configurable
    result = app.invoke(
        initial_state,
        config={"run_name": f"Book: {initial_state.title}", "configurable": {"output_manager": output_mgr}},
    )

    # Print completion message
    print("\nWorkflow completed successfully!")

    # Extract the final state from LangGraph result
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)

    # Save modules (PDF book node already saves course.json)
    output_mgr.save_modules(final_state)

    # Print summary
    total_sections = sum(len(s.sections) for m in final_state.modules for s in m.submodules)
    print("\n📊 Book Summary:")
    print(f"   Title: {final_state.title}")
    print(f"   Modules: {len(final_state.modules)}")
    print(f"   Total Sections: {total_sections}")
    print(f"   Language: {final_state.language}")
    if final_state.bibliography:
        print(f"   Bibliography: {len(final_state.bibliography.all_books)} books")
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")
    print(f"   📗 PDF book: {output_mgr.get_run_folder()}/book/book.pdf")
