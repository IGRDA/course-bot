"""
PDF to Podcast Workflow.

Generates a course from a PDF syllabus and creates podcasts for all modules.
Skips activities, HTML generation, and image generation for faster execution.

Pipeline:
1. Extract course structure from PDF
2. Generate section theories
3. Calculate metadata
4. Generate podcasts for all modules
"""

from pathlib import Path

from langgraph.graph import StateGraph, START, END

from workflows.state import CourseState, CourseConfig
from workflows.output_manager import OutputManager
from workflows.nodes import (
    generate_index_from_pdf_node,
    generate_theories_node,
    calculate_metadata_node,
    generate_podcasts_node,
)


def build_pdf2podcast_graph():
    """Build and return the PDF to podcast generation graph.
    
    This graph has a streamlined pipeline:
    1. generate_index_from_pdf - Extracts course structure from PDF syllabus
    2. generate_theories - Populates section content for podcast dialogue
    3. calculate_metadata - Sets IDs and indexes for proper course.json structure
    4. generate_podcasts - Generates podcast audio for ALL modules
    
    Skipped steps (not needed for podcasts):
    - generate_activities
    - generate_html
    - generate_images
    """
    graph = StateGraph(CourseState)
    
    # Add nodes for PDF to podcast pipeline
    graph.add_node("generate_index_from_pdf", generate_index_from_pdf_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_podcasts", generate_podcasts_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index_from_pdf")
    graph.add_edge("generate_index_from_pdf", "generate_theories")
    graph.add_edge("generate_theories", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_podcasts")
    graph.add_edge("generate_podcasts", END)
    
    return graph.compile()


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate podcasts from a PDF syllabus (no activities, HTML, or images)")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF syllabus file")
    parser.add_argument("--total-pages", type=int, default=50, help="Total pages for the course (default: 50)")
    parser.add_argument("--target-words", type=int, default=600, help="Target word count per podcast (default: 600)")
    parser.add_argument("--tts-engine", type=str, choices=["edge", "coqui", "elevenlabs", "chatterbox", "openai_tts", "qwen_tts", "mlx_tts", "qwen_tts_api"], default="qwen_tts_api", help="TTS engine (default: qwen_tts_api)")
    parser.add_argument("--language", type=str, default="Español", help="Language for content generation (default: Español)")
    parser.add_argument("--provider", type=str, default="mistral", help="LLM provider (default: mistral)")
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        exit(1)
    
    from LLMs.text2text.health_check import validate_provider_keys
    validate_provider_keys(args.provider)

    # Build the graph
    app = build_pdf2podcast_graph()
    
    # Create initial CourseState with PDF to podcast config
    course_config = CourseConfig(
        pdf_syllabus_path=str(pdf_path),
        text_llm_provider=args.provider,
        web_search_provider="ddg",
        total_pages=args.total_pages,
        words_per_page=400,
        language=args.language,
        max_retries=8,
        concurrency=20,
        use_reflection=True,
        num_reflection_queries=5,
        # Research configuration
        enable_research=True,
        research_max_queries=5,
        research_max_results_per_query=5,
        # Podcast configuration
        podcast_target_words=args.target_words,
        podcast_tts_engine=args.tts_engine,
        podcast_speaker_map={'host': 'adrian', 'guest': 'teresa'},
        target_audience=None,
    )
    
    initial_state = CourseState(
        config=course_config,
        title="",  # Will be extracted from PDF
        modules=[]
    )
    
    # Create OutputManager - title will be updated after PDF extraction
    output_mgr = OutputManager(title=pdf_path.stem.replace(" ", "_"))
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")
    print(f"📄 PDF Source: {pdf_path}")
    
    # Run the graph
    result = app.invoke(
        initial_state,
        config={
            "run_name": f"PDF2Podcast: {pdf_path.stem}",
            "configurable": {"output_manager": output_mgr}
        }
    )
    
    # Print the final course state
    print("\nWorkflow completed successfully!")
    
    # Extract the final state
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
    print(f"   TTS Engine: {final_state.config.podcast_tts_engine}")
    print(f"   Source PDF: {final_state.config.pdf_syllabus_path}")
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")
