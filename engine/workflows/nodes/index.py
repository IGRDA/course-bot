"""
Index generation nodes for course structure creation.

Agent imports are deferred to the node functions that use them.
"""

from langchain_core.runnables import RunnableConfig

from workflows.state import CourseState

from .utils import get_output_manager


def generate_index_node(state: CourseState, config: RunnableConfig | None = None) -> CourseState:
    """Generate the course skeleton with empty theories while preserving config.

    This node creates the course structure (modules, submodules, sections) from a topic.
    Optionally includes a research phase if enabled in config.

    Args:
        state: Current CourseState with config and title.
        config: LangGraph RunnableConfig for accessing OutputManager.

    Returns:
        Updated CourseState with populated modules and research data.
    """
    print("Generating course skeleton...")

    from agents.index_generator.agent import generate_course_state

    # Use config from the existing state
    course_config = state.config

    # Generate new course content skeleton (with empty theories)
    # Now includes research phase if enabled
    content_skeleton = generate_course_state(
        title=state.title,
        total_pages=course_config.total_pages,
        description=course_config.description,
        language=course_config.language,
        max_retries=course_config.max_retries,
        words_per_page=course_config.words_per_page,
        provider=course_config.text_llm_provider,
        # Research configuration
        enable_research=course_config.enable_research,
        web_search_provider=course_config.web_search_provider,
        research_max_queries=course_config.research_max_queries,
        research_max_results_per_query=course_config.research_max_results_per_query,
        # Audience configuration
        target_audience=course_config.target_audience,
    )

    # Transfer generated content to state
    state.modules = content_skeleton.modules

    # Preserve research output for downstream agents
    state.research = content_skeleton.research

    print("Course skeleton generated successfully!")
    if state.research:
        print(f"   Research summary: {state.research.course_summary[:100]}...")
        print(f"   Learning objectives: {len(state.research.learning_objectives)}")
        print(f"   Key topics: {len(state.research.key_topics)}")

    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("index", state)

    return state


def generate_index_from_pdf_node(state: CourseState, config: RunnableConfig | None = None) -> CourseState:
    """Generate the course skeleton from PDF syllabus with empty theories while preserving config.

    This node extracts the course structure from a PDF file instead of generating from topic.

    Args:
        state: Current CourseState with config containing pdf_syllabus_path.
        config: LangGraph RunnableConfig for accessing OutputManager.

    Returns:
        Updated CourseState with populated modules, title, and research data.

    Raises:
        ValueError: If pdf_syllabus_path is not specified in CourseConfig.
    """
    print("Generating course skeleton from PDF syllabus...")

    from agents.pdf_index_generator.agent import generate_course_state_from_pdf

    # Use config from the existing state
    course_config = state.config

    # Validate PDF path
    if not course_config.pdf_syllabus_path:
        raise ValueError("pdf_syllabus_path must be specified in CourseConfig")

    # Generate new course content skeleton from PDF (with empty theories)
    content_skeleton = generate_course_state_from_pdf(
        pdf_path=course_config.pdf_syllabus_path,
        total_pages=course_config.total_pages,
        language=course_config.language,
        max_retries=course_config.max_retries,
        words_per_page=course_config.words_per_page,
        provider=course_config.text_llm_provider,
        # Research configuration
        enable_research=course_config.enable_research,
        web_search_provider=course_config.web_search_provider,
        research_max_queries=course_config.research_max_queries,
        research_max_results_per_query=course_config.research_max_results_per_query,
    )

    # Transfer generated content to state
    state.title = content_skeleton.title
    state.modules = content_skeleton.modules

    # Preserve research output for downstream agents
    state.research = content_skeleton.research

    print("Course skeleton generated successfully from PDF!")
    if state.research:
        print(f"   Research summary: {state.research.course_summary[:100]}...")
        print(f"   Learning objectives: {len(state.research.learning_objectives)}")
        print(f"   Key topics: {len(state.research.key_topics)}")

    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("index", state)

    return state
