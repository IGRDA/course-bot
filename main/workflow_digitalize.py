"""
Content Digitalization Workflow.

Reads a folder of structured markdown files (one per module) and produces
a fully enriched course with all the same elements as the main workflow:
activities, HTML formatting, images, videos, bibliography, people, mindmaps,
podcasts, and PDF book -- all selectable via CLI flags.

Pipeline:
    parse_markdown -> restructure_content -> calculate_metadata
    -> generate_activities -> generate_html -> generate_images
    -> inject_local_images -> generate_videos -> generate_bibliography
    -> generate_people -> generate_mindmap -> generate_podcasts
    -> generate_pdf_book
"""

from langgraph.graph import StateGraph, START, END

from main.state import CourseState, CourseConfig
from main.output_manager import OutputManager
from main.nodes.digitalize import detect_language_node
from main.nodes import (
    parse_markdown_folder_node,
    restructure_parsed_content_node,
    calculate_metadata_node,
    generate_activities_node,
    generate_html_node,
    generate_images_node,
    inject_local_images_node,
    generate_videos_node,
    generate_bibliography_node,
    generate_people_node,
    generate_mindmap_node,
    generate_podcasts_node,
    generate_pdf_book_node,
)


# ---------------------------------------------------------------------------
# Conditional wrapper nodes
# ---------------------------------------------------------------------------

_original_restructure_node = restructure_parsed_content_node
_original_activities_node = generate_activities_node
_original_html_node = generate_html_node
_original_images_node = generate_images_node
_original_podcasts_node = generate_podcasts_node
_original_pdf_node = generate_pdf_book_node

# Controlled by a module-level flag set from CLI args before graph compilation
_skip_restructure = False


def _conditional_restructure_node(state: CourseState, config=None) -> CourseState:
    if _skip_restructure:
        print("LLM restructuring skipped (--no-restructure)")
        detect_language_node(state, config)
        return state
    return _original_restructure_node(state, config)


def _conditional_activities_node(state: CourseState, config=None) -> CourseState:
    if not getattr(state.config, 'generate_activities', True):
        print("Activities generation skipped (--no-activities)")
        return state
    return _original_activities_node(state, config)


def _conditional_html_node(state: CourseState, config=None) -> CourseState:
    if not state.config.generate_html_output and not state.config.generate_pdf:
        print("HTML formatting skipped (no --html or --pdf flag)")
        return state
    return _original_html_node(state, config)


def _conditional_images_node(state: CourseState, config=None) -> CourseState:
    if not state.config.generate_html_output and not state.config.generate_pdf:
        print("Image generation skipped (no --html or --pdf flag)")
        return state
    return _original_images_node(state, config)


def _conditional_podcast_node(state: CourseState, config=None) -> CourseState:
    if not state.config.generate_podcast:
        print("Podcast generation skipped (no --podcast flag)")
        return state
    return _original_podcasts_node(state, config)


def _conditional_pdf_node(state: CourseState, config=None) -> CourseState:
    if not state.config.generate_pdf:
        print("PDF book generation skipped (no --pdf flag)")
        return state
    return _original_pdf_node(state, config)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_digitalize_graph_conditional():
    """Build the full digitalization graph with conditional nodes.

    Matches the main workflow node set, but starts from parsed markdown
    instead of LLM-generated index/theories.
    """
    graph = StateGraph(CourseState)

    graph.add_node("parse_markdown", parse_markdown_folder_node)
    graph.add_node("restructure_content", _conditional_restructure_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_activities", _conditional_activities_node)
    graph.add_node("generate_html", _conditional_html_node)
    graph.add_node("generate_images", _conditional_images_node)
    graph.add_node("inject_local_images", inject_local_images_node)
    graph.add_node("generate_videos", generate_videos_node)
    graph.add_node("generate_bibliography", generate_bibliography_node)
    graph.add_node("generate_people", generate_people_node)
    graph.add_node("generate_mindmap", generate_mindmap_node)
    graph.add_node("generate_podcasts", _conditional_podcast_node)
    graph.add_node("generate_pdf_book", _conditional_pdf_node)

    graph.add_edge(START, "parse_markdown")
    graph.add_edge("parse_markdown", "restructure_content")
    graph.add_edge("restructure_content", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_activities")
    graph.add_edge("generate_activities", "generate_html")
    graph.add_edge("generate_html", "generate_images")
    graph.add_edge("generate_images", "inject_local_images")
    graph.add_edge("inject_local_images", "generate_videos")
    graph.add_edge("generate_videos", "generate_bibliography")
    graph.add_edge("generate_bibliography", "generate_people")
    graph.add_edge("generate_people", "generate_mindmap")
    graph.add_edge("generate_mindmap", "generate_podcasts")
    graph.add_edge("generate_podcasts", "generate_pdf_book")
    graph.add_edge("generate_pdf_book", END)

    return graph.compile()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Digitalize markdown content into a complete course with HTML, podcast, and/or PDF book",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate HTML with all enrichments
    python -m main.workflow_digitalize --source ./my_course --html

    # Generate everything
    python -m main.workflow_digitalize --source ./my_course --html --podcast --pdf

    # HTML only, skip slow steps
    python -m main.workflow_digitalize --source ./my_course --html --no-images --no-videos

    # Podcast only (skips HTML/images automatically)
    python -m main.workflow_digitalize --source ./my_course --podcast

    # Skip LLM restructuring (faster, uses raw parsed structure)
    python -m main.workflow_digitalize --source ./my_course --html --no-restructure
""",
    )
    # --- Required ---
    parser.add_argument(
        "--source", required=True,
        help="Path to the markdown folder (one .md file per module)",
    )

    # --- Output selection ---
    output_group = parser.add_argument_group("output selection (at least one required)")
    output_group.add_argument("--html", action="store_true", help="Generate interactive HTML course")
    output_group.add_argument("--podcast", action="store_true", help="Generate podcast audio per module")
    output_group.add_argument("--pdf", action="store_true", help="Generate PDF book")

    # --- General ---
    parser.add_argument("--title", default="", help="Course title (default: derived from folder name)")
    parser.add_argument("--language", default="auto", help="Content language (default: auto-detect from content)")
    parser.add_argument(
        "--provider", default="mistral",
        choices=["mistral", "gemini", "groq", "openai", "deepseek"],
        help="LLM provider (default: mistral)",
    )

    # --- Enrichment opt-out flags ---
    enrich = parser.add_argument_group("enrichment control (all enabled by default)")
    enrich.add_argument("--no-restructure", action="store_true", help="Skip LLM-assisted structure improvement")
    enrich.add_argument("--no-activities", action="store_true", help="Skip quiz/glossary/meta-elements generation")
    enrich.add_argument("--no-images", action="store_true", help="Skip internet image search")
    enrich.add_argument("--no-videos", action="store_true", help="Skip YouTube video recommendations")
    enrich.add_argument("--no-bibliography", action="store_true", help="Skip book bibliography")
    enrich.add_argument("--no-people", action="store_true", help="Skip relevant people lookup")
    enrich.add_argument("--no-mindmap", action="store_true", help="Skip mind map generation")

    # --- Podcast options ---
    pod = parser.add_argument_group("podcast options")
    pod.add_argument(
        "--tts-engine", default="edge",
        choices=["edge", "coqui", "elevenlabs", "chatterbox", "openai_tts"],
        help="TTS engine (default: edge)",
    )
    pod.add_argument("--target-words", type=int, default=600, help="Words per podcast episode (default: 600)")

    args = parser.parse_args()

    if not (args.html or args.podcast or args.pdf):
        parser.error("Specify at least one output: --html, --podcast, or --pdf")

    # Set module-level flag so conditional node can check it
    _skip_restructure = args.no_restructure

    # Language: use "English" as fallback when auto-detect is requested
    # (the restructurer will detect and override)
    initial_language = "English" if args.language == "auto" else args.language

    needs_html = args.html or args.pdf

    course_config = CourseConfig(
        title=args.title,
        md_source_path=args.source,
        language=initial_language,
        text_llm_provider=args.provider,
        # Output flags
        generate_html_output=needs_html,
        generate_podcast=args.podcast,
        generate_pdf=args.pdf,
        # Activities
        generate_activities=not args.no_activities,
        activities_concurrency=30,
        activity_selection_mode="deterministic",
        sections_per_activity=1,
        # HTML formatting
        html_concurrency=15,
        select_html="LLM",
        html_formats="paragraphs|accordion|tabs|carousel|flip|timeline|conversation",
        include_quotes_in_html=True,
        include_tables_in_html=True,
        # Image search (internet)
        image_search_provider="freepik",
        use_vision_ranking=False,
        num_images_to_fetch=8,
        image_concurrency=3,
        # Videos
        generate_videos=not args.no_videos,
        videos_per_module=3,
        # Bibliography
        generate_bibliography=not args.no_bibliography,
        bibliography_books_per_module=5,
        bibliography_articles_per_module=5,
        # People
        generate_people=not args.no_people,
        people_per_module=3,
        # Mindmap
        generate_mindmap=not args.no_mindmap,
        mindmap_max_nodes=20,
        # Podcast
        podcast_target_words=args.target_words,
        podcast_tts_engine=args.tts_engine,
        podcast_speaker_map=(
            {"host": "es-ES-AlvaroNeural", "guest": "es-ES-XimenaNeural"}
            if initial_language.lower().startswith("espa")
            else None
        ),
        max_retries=8,
        concurrency=10,
    )

    initial_state = CourseState(
        config=course_config,
        title=args.title,
        modules=[],
    )

    output_mgr = OutputManager(title=args.title or "Digitalized_Course")

    enabled_outputs = [f for f, v in [("HTML", args.html), ("Podcast", args.podcast), ("PDF", args.pdf)] if v]
    skipped = [s for s, v in [
        ("restructure", args.no_restructure), ("activities", args.no_activities),
        ("images", args.no_images), ("videos", args.no_videos),
        ("bibliography", args.no_bibliography), ("people", args.no_people),
        ("mindmap", args.no_mindmap),
    ] if v]

    print(f"Output folder: {output_mgr.get_run_folder()}")
    print(f"Source: {args.source}")
    print(f"Outputs: {' + '.join(enabled_outputs)}")
    print(f"Language: {args.language}  |  Provider: {args.provider}")
    if skipped:
        print(f"Skipped enrichments: {', '.join(skipped)}")

    app = build_digitalize_graph_conditional()

    result = app.invoke(
        initial_state,
        config={
            "run_name": "Content Digitalization",
            "configurable": {"output_manager": output_mgr},
        },
    )

    print("\nWorkflow completed successfully!")

    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)

    output_mgr.save_final(final_state)
    output_mgr.save_modules(final_state)

    total_sections = sum(len(s.sections) for m in final_state.modules for s in m.submodules)
    modules = final_state.modules
    print(f"\nCourse Summary:")
    print(f"   Title: {final_state.title}")
    print(f"   Modules: {len(modules)}")
    print(f"   Total Sections: {total_sections}")
    print(f"   Language: {final_state.language}")

    # Enrichment status
    has_videos = sum(1 for m in modules if m.video is not None)
    has_bib = sum(1 for m in modules if m.bibliography is not None)
    has_people = sum(1 for m in modules if m.relevant_people)
    has_mindmap = sum(1 for m in modules if m.mindmap is not None)
    has_summary = sum(
        1 for m in modules for sm in m.submodules for s in sm.sections if s.summary
    )

    if final_state.bibliography:
        print(f"   Bibliography: {len(final_state.bibliography.all_books)} books")
    print(f"   Videos: {has_videos}/{len(modules)} modules")
    print(f"   Bibliography (module): {has_bib}/{len(modules)} modules")
    print(f"   People: {has_people}/{len(modules)} modules")
    print(f"   Mindmaps: {has_mindmap}/{len(modules)} modules")
    print(f"   Summaries: {has_summary}/{total_sections} sections")
    print(f"\nAll outputs saved to: {output_mgr.get_run_folder()}")
