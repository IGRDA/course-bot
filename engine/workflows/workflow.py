"""
Topic-based Course Generation Workflow.

Generates a complete course from a topic using:
1. Research phase (optional)
2. Index/skeleton generation
3. Theory generation
4. Activities generation
5. HTML formatting
6. Image generation
7. Video recommendations (optional)
8. Bibliography generation (optional)
9. Relevant people generation (optional)
"""

from langgraph.graph import END, START, StateGraph

from workflows.nodes import (
    calculate_metadata_node,
    generate_activities_node,
    generate_bibliography_node,
    generate_html_node,
    generate_images_node,
    generate_index_node,
    generate_mindmap_node,
    generate_people_node,
    generate_theories_node,
    generate_videos_node,
)
from workflows.output_manager import OutputManager
from workflows.state import CourseConfig, CourseState


def build_course_generation_graph():
    """Build and return the course generation graph."""
    graph = StateGraph(CourseState)

    # Add nodes for complete course generation pipeline
    graph.add_node("generate_index", generate_index_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("generate_activities", generate_activities_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_html", generate_html_node)
    graph.add_node("generate_images", generate_images_node)
    graph.add_node("generate_videos", generate_videos_node)
    graph.add_node("generate_bibliography", generate_bibliography_node)
    graph.add_node("generate_people", generate_people_node)
    graph.add_node("generate_mindmap", generate_mindmap_node)

    # Add edges for sequential execution
    graph.add_edge(START, "generate_index")
    graph.add_edge("generate_index", "generate_theories")
    graph.add_edge("generate_theories", "generate_activities")
    graph.add_edge("generate_activities", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_html")
    graph.add_edge("generate_html", "generate_images")
    graph.add_edge("generate_images", "generate_videos")
    graph.add_edge("generate_videos", "generate_bibliography")
    graph.add_edge("generate_bibliography", "generate_people")
    graph.add_edge("generate_people", "generate_mindmap")
    graph.add_edge("generate_mindmap", END)

    return graph.compile()


if __name__ == "__main__":
    import argparse

    from LLMs.text2text.health_check import validate_provider_keys

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a course")
    parser.add_argument("--total-pages", type=int, default=2, help="Total pages for the course (default: 2)")
    parser.add_argument(
        "--theory-only",
        action="store_true",
        help="No web search; content from theory only (research and reflection disabled).",
    )
    args = parser.parse_args()

    validate_provider_keys("mistral")

    # Build the graph
    app = build_course_generation_graph()

    # Create initial CourseState with config and minimal content
    course_config = CourseConfig(
        title="Quantum Theory",
        text_llm_provider="mistral",  # LLM provider: mistral | gemini | groq | openai | deepseek
        web_search_provider="ddg",  # Web search provider: ddg | tavily | wikipedia
        total_pages=args.total_pages,  # Total pages for the course
        words_per_page=400,  # Target words per page
        language="Español",
        description="",
        max_retries=8,
        concurrency=20,  # Number of concurrent section theory generations
        use_reflection=True,  # Enable reflection pattern for fact verification (default: False)
        num_reflection_queries=7,  # Number of verification queries per section (default: 3)
        # Research configuration
        enable_research=True,  # Enable research phase before index generation
        research_max_queries=7,  # Number of search queries to generate
        research_max_results_per_query=5,  # Results per search query
        # Activities configuration
        activities_concurrency=40,  # Number of concurrent activity generations
        activity_selection_mode="deterministic",  # "random" or "deterministic"
        sections_per_activity=1,  # Generate activities every N sections within each submodule (1 = every section)
        # HTML configuration
        html_concurrency=20,  # Number of concurrent HTML generations
        select_html="LLM",  # "LLM" | "random"
        html_formats="paragraphs|accordion|tabs|carousel|flip|timeline|conversation",  # Pipe-separated list
        html_random_seed=42,  # Seed for deterministic random selection
        include_quotes_in_html=True,  # Include quote elements
        include_tables_in_html=True,  # Include table elements
        # Image generation configuration
        image_search_provider="freepik",  # Image search provider: bing | freepik | ddg | google
        use_vision_ranking=False,  # Use vision LLM to rank images (slower but better quality)
        num_images_to_fetch=8,  # Number of images to fetch for ranking
        vision_llm_provider="pixtral",  # Vision LLM provider for image ranking
        image_concurrency=5,  # Number of image blocks to process in parallel
        imagetext2text_concurrency=5,  # Number of Pixtral vision LLM calls in parallel
        vision_ranking_batch_size=8,  # Number of images per batch for Pixtral ranking
        target_audience=None,  # Target audience: None | "kids" | "general" | "advanced"
        # Video generation configuration
        generate_videos=True,  # Enable video recommendations
        videos_per_module=3,  # Number of videos per module
        # Bibliography generation configuration
        generate_bibliography=True,  # Enable book bibliography
        bibliography_books_per_module=5,  # Number of books per module
        bibliography_articles_per_module=5,  # Number of articles per module
        # People generation configuration
        generate_people=True,  # Enable relevant people
        people_per_module=3,  # Number of people per module
        # Mind map generation configuration
        generate_mindmap=True,  # Enable mind map generation
        mindmap_max_nodes=20,  # Maximum nodes per mind map
    )

    initial_state = CourseState(
        config=course_config,
        title=course_config.title,  # Initialize from config, can be refined during generation
        modules=[],  # Will be populated by skeleton generation
    )

    # Create OutputManager for this run (creates timestamped folder)
    output_mgr = OutputManager(title=course_config.title)
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")

    # Run the graph with OutputManager passed via configurable
    result = app.invoke(
        initial_state, config={"run_name": f"{initial_state.title}", "configurable": {"output_manager": output_mgr}}
    )

    # Print the final course state
    print("\nWorkflow completed successfully!")

    # Extract the final state from LangGraph result (which is a dictionary)
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)

    # Save final outputs: course.json, course.html, and individual modules
    output_mgr.save_final(final_state)
    output_mgr.save_modules(final_state)

    # Print summary
    total_sections = sum(len(s.sections) for m in final_state.modules for s in m.submodules)
    print("\n📊 Course Summary:")
    print(f"   Title: {final_state.title}")
    print(f"   Modules: {len(final_state.modules)}")
    print(f"   Total Sections: {total_sections}")
    print(f"   Language: {final_state.language}")
    print(f"   HTML Selection: {final_state.config.select_html}")
    print(f"   Available Formats: {final_state.config.html_formats}")
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")
