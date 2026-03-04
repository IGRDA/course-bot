"""
Extra nodes for bibliography, video, people, podcast, and PDF book generation.

Agent imports are deferred to the node functions that use them so that
importing this module does not eagerly pull in every agent and its
transitive dependencies.
"""

import json
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from main.state import CourseState
from .utils import get_output_manager


def generate_bibliography_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate book bibliography for the course (optional, controlled by config).
    
    This node creates book recommendations for each module using Open Library API.
    Only runs if generate_bibliography is enabled in config.
    
    Args:
        state: CourseState with populated course structure.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with bibliography if enabled, unchanged otherwise.
    """
    if not state.config.generate_bibliography:
        print("📚 Bibliography generation disabled, skipping...")
        return state
    
    print("📚 Generating course bibliography...")
    
    from agents.bibliography_generator.agent import generate_course_bibliography

    try:
        bibliography = generate_course_bibliography(state)
        state.bibliography = bibliography

        modules_with_bib = sum(1 for m in state.modules if m.bibliography is not None)
        print(f"   Bibliography: {modules_with_bib}/{len(state.modules)} modules enriched")
        if modules_with_bib == 0:
            print("   WARNING: No modules received bibliography data")
    except Exception as e:
        print(f"   ERROR in bibliography generation: {e}")
    
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("bibliography", state)
    
    return state


def generate_videos_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate video recommendations for each module (optional, controlled by config).
    
    This node searches YouTube for relevant educational videos for each module
    using LLM-generated search queries.
    Only runs if generate_videos is enabled in config.
    
    Args:
        state: CourseState with populated course structure.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with videos if enabled, unchanged otherwise.
    """
    if not state.config.generate_videos:
        print("🎬 Video generation disabled, skipping...")
        return state
    
    print("🎬 Generating video recommendations...")
    
    from agents.video_search.agent import generate_course_videos

    try:
        course_videos = generate_course_videos(state)
        state.videos = course_videos

        modules_with_video = sum(1 for m in state.modules if m.video is not None)
        print(f"   Videos: {modules_with_video}/{len(state.modules)} modules enriched")
        if modules_with_video == 0:
            print("   WARNING: No modules received video data")
    except Exception as e:
        print(f"   ERROR in video generation: {e}")
    
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("videos", state)
    
    return state


def generate_people_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate relevant people for each module (optional, controlled by config).
    
    This node searches for notable people relevant to each module's topic
    using LLM + Wikipedia validation.
    Only runs if generate_people is enabled in config.
    
    Args:
        state: CourseState with populated course structure.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with people embedded in modules if enabled, unchanged otherwise.
    """
    if not state.config.generate_people:
        print("👥 People generation disabled, skipping...")
        return state
    
    print("👥 Generating relevant people...")
    
    from agents.people_search.agent import generate_course_people

    try:
        state = generate_course_people(state)

        modules_with_people = sum(1 for m in state.modules if m.relevant_people)
        print(f"   People: {modules_with_people}/{len(state.modules)} modules enriched")
        if modules_with_people == 0:
            print("   WARNING: No modules received people data")
    except Exception as e:
        print(f"   ERROR in people generation: {e}")
    
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("people", state)
    
    return state


def generate_mindmap_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate mind maps for each module (optional, controlled by config).
    
    This node creates hierarchical concept maps for each module using
    LLM-powered generation following Novak's concept map methodology.
    Only runs if generate_mindmap is enabled in config.
    
    Args:
        state: CourseState with populated course structure.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with mind maps embedded in modules if enabled, unchanged otherwise.
    """
    if not state.config.generate_mindmap:
        print("🧠 Mind map generation disabled, skipping...")
        return state
    
    print("🧠 Generating mind maps...")
    
    from agents.mind_map_generator.agent import generate_course_mindmaps

    try:
        state = generate_course_mindmaps(state)

        modules_with_mindmap = sum(1 for m in state.modules if m.mindmap is not None)
        print(f"   Mindmaps: {modules_with_mindmap}/{len(state.modules)} modules enriched")
        if modules_with_mindmap == 0:
            print("   WARNING: No modules received mindmap data")
    except Exception as e:
        print(f"   ERROR in mindmap generation: {e}")
    
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("mindmap", state)
    
    return state


def generate_podcasts_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate podcasts for all modules in the course.
    
    This node:
    1. Converts the CourseState to course_data dict format
    2. Iterates through all modules
    3. Generates conversation for each module using LLM
    4. Synthesizes audio using configured TTS engine (Edge TTS by default)
    5. Saves conversation JSON and MP3 files to output folder
    
    Args:
        state: CourseState with populated section theories.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Unchanged CourseState (podcasts are saved as files).
    """
    print("Generating podcasts for all modules...")
    
    # Get output manager for saving files
    output_mgr = get_output_manager(config)
    if not output_mgr:
        print("⚠️ Warning: No OutputManager found, skipping podcast generation")
        return state
    
    from agents.podcast_generator.agent import generate_conversation, get_tts_language, TTSEngineType

    # Convert state to course_data dict format expected by podcast generator
    course_data = state.model_dump()
    
    # Get podcast configuration
    tts_engine: TTSEngineType = state.config.podcast_tts_engine
    target_words = state.config.podcast_target_words
    speaker_map = state.config.podcast_speaker_map
    provider = state.config.text_llm_provider
    
    # Get TTS language from course language
    tts_language = get_tts_language(state.config.language)
    
    # Setup podcast output directory
    podcast_dir = Path(output_mgr.get_run_folder()) / "podcast"
    podcast_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the path to background music
    project_root = Path(__file__).parent.parent.parent
    music_path = project_root / "tools" / "podcast" / "background_music.mp3"
    music_path_str = str(music_path) if music_path.exists() else None
    
    num_modules = len(state.modules)
    print(f"   Found {num_modules} modules to process")
    print(f"   TTS Engine: {tts_engine}")
    print(f"   Language: {tts_language}")
    print(f"   Target words per podcast: {target_words}")
    
    # Generate podcast for each module
    for module_idx in range(num_modules):
        module = state.modules[module_idx]
        print(f"\n🎙️ Generating podcast for Module {module_idx + 1}: {module.title}")
        
        # Generate conversation
        print(f"   Generating conversation...")
        conversation = generate_conversation(
            course_data=course_data,
            module_idx=module_idx,
            provider=provider,
            target_words=target_words,
        )
        
        # Save conversation JSON
        conv_filename = f"module_{module_idx + 1}_conversation.json"
        conv_path = podcast_dir / conv_filename
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Conversation saved: {conv_path.name}")
        
        # Generate audio
        audio_filename = f"module_{module_idx + 1}.mp3"
        audio_path = podcast_dir / audio_filename
        
        print(f"   🔊 Synthesizing audio with {tts_engine.upper()} TTS...")
        
        if tts_engine == "edge":
            from tools.podcast import generate_podcast_edge
            
            generate_podcast_edge(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {module.title}",
                artist="Adinhub",
                album=state.title,
                track_number=module_idx + 1,
                music_path=music_path_str,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        elif tts_engine == "elevenlabs":
            from tools.podcast import generate_podcast_elevenlabs
            
            generate_podcast_elevenlabs(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {module.title}",
                artist="Adinhub",
                album=state.title,
                track_number=module_idx + 1,
                music_path=music_path_str,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        elif tts_engine == "chatterbox":
            from tools.podcast import generate_podcast_chatterbox
            
            generate_podcast_chatterbox(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {module.title}",
                artist="Adinhub",
                album=state.title,
                track_number=module_idx + 1,
                music_path=music_path_str,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        elif tts_engine == "openai_tts":
            from tools.podcast import generate_podcast_openai_tts
            
            generate_podcast_openai_tts(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {module.title}",
                artist="Adinhub",
                album=state.title,
                track_number=module_idx + 1,
                music_path=music_path_str,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        else:
            from tools.podcast import generate_podcast
            
            generate_podcast(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                language_code=tts_language,
                title=f"Module {module_idx + 1}: {module.title}",
                artist="Adinhub",
                album=state.title,
                track_number=module_idx + 1,
                music_path=music_path_str,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        
        print(f"   ✅ Audio saved: {audio_path.name}")
    
    print(f"\n✅ All {num_modules} podcasts generated successfully!")
    print(f"   Output folder: {podcast_dir}")
    
    return state


def generate_pdf_book_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate PDF book from course content using LaTeX.
    
    This node:
    1. Saves course.json first (PDF tool reads from disk)
    2. Generates a PDF book via LaTeX compilation
    3. Downloads images and includes them in the PDF
    4. Includes bibliography if available
    
    Args:
        state: CourseState with populated section theories and images.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Unchanged CourseState (PDF is saved as a file).
    """
    print("📗 Generating PDF book...")
    
    # Get output manager for saving files
    output_mgr = get_output_manager(config)
    if not output_mgr:
        print("⚠️ Warning: No OutputManager found, skipping PDF book generation")
        return state
    
    # Save course.json first (PDF tool reads from disk)
    # Note: save_final returns (json_path, html_path)
    json_path, _ = output_mgr.save_final(state)
    
    # Also save bibliography.json if available
    if state.bibliography:
        bibliography_path = Path(output_mgr.get_run_folder()) / "bibliography.json"
        with open(bibliography_path, "w", encoding="utf-8") as f:
            f.write(state.bibliography.model_dump_json(indent=2))
        print(f"   💾 Bibliography saved: bibliography.json")
    
    # Generate PDF book
    from tools.json2book import generate_pdf_book
    
    try:
        pdf_path = generate_pdf_book(
            course_json_path=json_path,
            template="academic",
            download_images=True,
            cleanup=True,
        )
        print(f"   ✅ PDF book generated: {pdf_path.name}")
    except FileNotFoundError as e:
        print(f"   ❌ PDF generation failed: {e}")
        print("   Hint: Make sure TeX Live or MiKTeX is installed (xelatex or pdflatex)")
    except RuntimeError as e:
        print(f"   ❌ PDF compilation error: {e}")
    
    return state

