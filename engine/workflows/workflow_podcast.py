"""
Podcast-focused Course Generation Workflow.

Generates a course (index + theories only) and creates podcasts for all modules.
Skips activities, HTML generation, and image generation for faster execution.
"""

from langgraph.graph import StateGraph, START, END

from workflows.state import CourseState, CourseConfig
from workflows.output_manager import OutputManager
from workflows.nodes import (
    generate_index_node,
    generate_theories_node,
    calculate_metadata_node,
    generate_podcasts_node,
)


def build_podcast_generation_graph():
    """Build and return the podcast-focused course generation graph.
    
    This graph has a streamlined pipeline:
    1. generate_index - Creates course structure with sections/summaries
    2. generate_theories - Populates section content for podcast dialogue
    3. calculate_metadata - Sets IDs and indexes for proper course.json structure
    4. generate_podcasts - Generates podcast audio for ALL modules
    
    Skipped steps (not needed for podcasts):
    - generate_activities
    - generate_html
    - generate_images
    """
    graph = StateGraph(CourseState)
    
    # Add nodes for podcast-focused pipeline
    graph.add_node("generate_index", generate_index_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_podcasts", generate_podcasts_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index")
    graph.add_edge("generate_index", "generate_theories")
    graph.add_edge("generate_theories", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_podcasts")
    graph.add_edge("generate_podcasts", END)
    
    return graph.compile()


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a course with podcasts (no activities, HTML, or images)")
    parser.add_argument("--total-pages", type=int, default=2, help="Total pages for the course (default: 2)")
    parser.add_argument("--target-words", type=int, default=600, help="Target word count per podcast (default: 600)")
    parser.add_argument("--tts-engine", type=str, choices=["edge", "coqui", "elevenlabs", "chatterbox", "openai_tts", "qwen_tts", "mlx_tts"], default="edge", help="TTS engine (default: edge)")
    args = parser.parse_args()
    
    from LLMs.text2text.health_check import validate_provider_keys
    validate_provider_keys("mistral")

    # Build the graph
    app = build_podcast_generation_graph()
    
    # Create initial CourseState with podcast-focused config
    course_config = CourseConfig(
        title="Fundamentos del Póker",
        text_llm_provider="mistral",
        web_search_provider="ddg",
        total_pages=args.total_pages,
        words_per_page=400,
        language="Español",
        description="",
        max_retries=8,
        concurrency=20,
        use_reflection=True,
        num_reflection_queries=7,
        # Research configuration
        enable_research=True,
        research_max_queries=5,
        research_max_results_per_query=5,
        # Podcast configuration
        podcast_target_words=args.target_words,
        podcast_tts_engine=args.tts_engine,
        podcast_speaker_map=(
            {'host': 'Luis Moray', 'guest': 'Alma María'}
            if args.tts_engine == 'coqui'
            else None if args.tts_engine in ('chatterbox', 'openai_tts')
            else {'host': 'es-ES-AlvaroNeural', 'guest': 'es-ES-XimenaNeural'}
        ),
        target_audience=None,
    )
    
    initial_state = CourseState(
        config=course_config,
        title=course_config.title,
        modules=[]
    )
    
    # Create OutputManager for this run
    output_mgr = OutputManager(title=course_config.title)
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")
    
    # Run the graph
    result = app.invoke(
        initial_state,
        config={
            "run_name": f"{initial_state.title}",
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
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")
