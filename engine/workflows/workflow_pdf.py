"""
PDF Course Generation Workflow.

Generates a complete course from a PDF syllabus using:
1. PDF extraction (modules, submodules, sections)
2. Research for enrichment
3. Theory generation
4. Activities generation
5. HTML formatting
6. Image generation
7. Video recommendations (optional)
8. Bibliography generation (optional)
9. Relevant people generation (optional)
"""

import argparse
from langgraph.graph import StateGraph, START, END

from workflows.state import CourseState, CourseConfig
from workflows.output_manager import OutputManager
from workflows.nodes import (
    generate_index_from_pdf_node,
    generate_theories_node,
    generate_activities_node,
    calculate_metadata_node,
    generate_html_node,
    generate_images_node,
    generate_videos_node,
    generate_bibliography_node,
    generate_people_node,
    generate_mindmap_node,
)


def build_course_generation_graph_from_pdf():
    """Build and return the course generation graph that uses PDF syllabus."""
    graph = StateGraph(CourseState)
    
    # Add nodes for complete course generation pipeline
    graph.add_node("generate_index_from_pdf", generate_index_from_pdf_node)
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
    graph.add_edge(START, "generate_index_from_pdf")
    graph.add_edge("generate_index_from_pdf", "generate_theories")
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
    parser = argparse.ArgumentParser(description="Generate course from PDF syllabus")
    parser.add_argument(
        "--theory-only",
        action="store_true",
        help="No web search; content from theory only (research and reflection disabled).",
    )
    parser.add_argument(
        "--pdf",
        default="example_pdfs/coaching_y_orientacion.pdf",
        help="Path to PDF syllabus (default: example_pdfs/coaching_y_orientacion.pdf)",
    )
    args = parser.parse_args()

    from LLMs.text2text.health_check import validate_provider_keys
    validate_provider_keys("mistral")

    # Build the graph
    app = build_course_generation_graph_from_pdf()
    
    # Create initial CourseState with config and PDF path
    # Note: title will be extracted from the PDF
    config = CourseConfig(
        pdf_syllabus_path=args.pdf,
        theory_only=args.theory_only,
        text_llm_provider="mistral",  # LLM provider: mistral | gemini | groq | openai | deepseek
        web_search_provider="ddg",  # Web search provider: ddg | tavily | wikipedia
        total_pages=200,  # Total pages for the course
        words_per_page=400,  # Target words per page
        language="Español",
        max_retries=8,
        concurrency=10,  # Number of concurrent section theory generations
        use_reflection=True,  # Enable reflection pattern for fact verification
        num_reflection_queries=5,  # Number of verification queries per section
        # Research configuration
        enable_research=True,  # Enable research phase for enrichment
        research_max_queries=5,  # Number of search queries to generate
        research_max_results_per_query=3,  # Results per search query
        # Activities configuration
        activities_concurrency=8,  # Number of concurrent activity generations
        activity_selection_mode="deterministic",  # "random" or "deterministic"
        sections_per_activity=1,  # Generate activities every N sections (1 = every section)
        # HTML configuration
        html_concurrency=8,  # Number of concurrent HTML generations
        select_html="LLM",  # "LLM" | "random"
        html_formats="paragraphs|accordion|tabs|carousel|flip|timeline|conversation",
        include_quotes_in_html=True,  # Include quote elements
        include_tables_in_html=True,  # Include table elements
        # Image generation configuration
        image_search_provider="freepik",  # Image search provider
        use_vision_ranking=False,  # Use vision LLM to rank images
        num_images_to_fetch=8,  # Number of images to fetch for ranking
        vision_llm_provider="pixtral",  # Vision LLM provider
        image_concurrency=5,  # Number of image blocks to process in parallel
        # Video generation configuration
        generate_videos=True,  # Enable video recommendations
        videos_per_module=3,   # Number of videos per module
        # Bibliography generation configuration
        generate_bibliography=True,  # Enable book bibliography
        bibliography_books_per_module=5,  # Number of books per module
        bibliography_articles_per_module=5,  # Number of articles per module
        # People generation configuration
        generate_people=True,  # Enable relevant people
        people_per_module=3,   # Number of people per module
        # Mind map generation configuration
        generate_mindmap=True,  # Enable mind map generation
        mindmap_max_nodes=20,   # Maximum nodes per mind map
    )
    
    initial_state = CourseState(
        config=config,
        title="",  # Will be extracted from PDF
        modules=[]  # Will be populated by PDF skeleton generation
    )
    
    # Create OutputManager for this run
    output_mgr = OutputManager(title="PDF_Course")
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")
    
    # Run the graph with OutputManager passed via configurable
    result = app.invoke(
        initial_state,
        config={
            "run_name": "PDF Course Generation",
            "configurable": {"output_manager": output_mgr}
        }
    )
    
    # Print the final course state
    print("\nWorkflow completed successfully!")
    
    # Extract the final state from LangGraph result
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)
    
    # Save final outputs
    output_mgr.save_final(final_state)
    output_mgr.save_modules(final_state)
    
    # Print summary
    total_sections = sum(len(s.sections) for m in final_state.modules for s in m.submodules)
    print(f"\n📊 Course Summary:")
    print(f"   Title: {final_state.title}")
    print(f"   Modules: {len(final_state.modules)}")
    print(f"   Total Sections: {total_sections}")
    print(f"   Language: {final_state.language}")
    print(f"   HTML Selection: {final_state.config.select_html}")
    print(f"   Source: {final_state.config.pdf_syllabus_path}")
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")
