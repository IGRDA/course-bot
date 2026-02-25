"""
Podcast Generator Agent.

Generates two-speaker educational dialogue conversations from course module content.
Supports both Coqui TTS (offline) and Edge TTS.
"""

import json
from typing import Optional, Literal
from pydantic import BaseModel, Field

from LLMs.text2text import create_text_llm, resolve_text_model_name
from .prompts import conversation_prompt


# TTS Engine types
TTSEngineType = Literal["edge", "coqui", "elevenlabs", "chatterbox", "openai_tts", "qwen_tts"]

# Language mapping from course language to TTS language code
LANGUAGE_MAP = {
    "español": "es",
    "spanish": "es", 
    "english": "en",
    "inglés": "en",
    "french": "fr",
    "français": "fr",
    "german": "de",
    "deutsch": "de",
    "italian": "it",
    "italiano": "it",
    "portuguese": "pt",
    "português": "pt",
}


class Message(BaseModel):
    """A single message in the podcast conversation."""
    role: str = Field(description="Speaker role: 'host' or 'guest'")
    content: str = Field(description="The spoken text content")


class ConversationOutput(BaseModel):
    """Structured output for the conversation."""
    messages: list[Message] = Field(description="List of conversation messages")


def get_tts_language(course_language: str) -> str:
    """Map course language to TTS language code.
    
    Args:
        course_language: Language from course.json (e.g., "Español", "English")
        
    Returns:
        TTS language code (e.g., "es", "en")
    """
    return LANGUAGE_MAP.get(course_language.lower().strip(), "en")


def extract_module_context(course_data: dict, module_idx: int) -> dict:
    """Extract module title, submodules, and section summaries.
    
    Args:
        course_data: Full course.json data
        module_idx: 0-based module index
        
    Returns:
        Dict with course_title, module_title, module_description, language, sections
    """
    if module_idx < 0 or module_idx >= len(course_data.get("modules", [])):
        raise ValueError(f"Module index {module_idx} out of range. "
                        f"Course has {len(course_data.get('modules', []))} modules.")
    
    module = course_data["modules"][module_idx]
    
    # Extract section summaries from all submodules
    sections = []
    for submodule in module.get("submodules", []):
        for section in submodule.get("sections", []):
            sections.append({
                "title": section.get("title", "Untitled"),
                "summary": section.get("summary", ""),
                "submodule_title": submodule.get("title", ""),
            })
    
    return {
        "course_title": course_data.get("title", "Course"),
        "module_title": module.get("title", "Module"),
        "module_description": module.get("description", ""),
        "language": course_data.get("config", {}).get("language", "English"),
        "sections": sections,
    }


def clean_conversation(conversation: list[dict]) -> list[dict]:
    """Clean conversation messages.
    
    Args:
        conversation: List of message dicts with 'role' and 'content' keys
        
    Returns:
        Cleaned conversation
    """
    return [
        {"role": msg["role"], "content": msg["content"].replace("*", "")}
        for msg in conversation
    ]


def format_sections_text(sections: list[dict]) -> str:
    """Format sections into readable text for the prompt.
    
    Args:
        sections: List of section dicts with title, summary, submodule_title
        
    Returns:
        Formatted string for prompt injection
    """
    parts = []
    for i, section in enumerate(sections, 1):
        parts.append(f"{i}. **{section['title']}** (from {section['submodule_title']})")
        if section['summary']:
            parts.append(f"   {section['summary']}")
        parts.append("")
    return "\n".join(parts)


def generate_conversation(
    course_data: dict,
    module_idx: int,
    provider: Optional[str] = None,
    target_words: int = 600,
) -> list[dict]:
    """Generate a podcast conversation for a course module.
    
    Args:
        course_data: Full course.json data
        module_idx: 0-based module index
        provider: LLM provider (default: from course config)
        target_words: Target word count for conversation
        
    Returns:
        List of message dicts [{"role": "host"|"guest", "content": "..."}]
    """
    # Extract context
    context = extract_module_context(course_data, module_idx)
    
    # Determine provider
    if provider is None:
        provider = course_data.get("config", {}).get("text_llm_provider", "openai")
    
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.1}  # Slightly creative for natural dialogue
    if model_name:
        llm_kwargs["model_name"] = model_name
    
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # Format sections for prompt
    sections_text = format_sections_text(context["sections"])
    
    # Estimate number of messages (~30 words per message average)
    num_messages = max(6, target_words // 30)
    
    # Create chain with structured output
    chain = conversation_prompt | llm.with_structured_output(ConversationOutput)
    
    # Generate conversation
    result = chain.invoke({
        "course_title": context["course_title"],
        "module_title": context["module_title"],
        "module_description": context["module_description"],
        "sections_text": sections_text,
        "language": context["language"],
        "target_words": target_words,
        "num_messages": num_messages,
    })
    
    # Convert to list of dicts and clean
    conversation = [{"role": m.role, "content": m.content} for m in result.messages]
    return clean_conversation(conversation)


def generate_module_podcast(
    course_path: str,
    module_idx: int,
    output_dir: Optional[str] = None,
    provider: Optional[str] = None,
    target_words: int = 600,
    skip_tts: bool = False,
    tts_engine: TTSEngineType = "edge",
    speaker_map: Optional[dict[str, str]] = None,
    tts_kwargs: Optional[dict] = None,
) -> dict:
    """Generate podcast conversation and optionally synthesize audio.
    
    Args:
        course_path: Path to course.json
        module_idx: 0-based module index
        output_dir: Output directory (default: podcast/ in course dir)
        provider: LLM provider override
        target_words: Target word count
        skip_tts: If True, only generate conversation JSON
        tts_engine: TTS engine to use ("edge", "coqui", "qwen_tts", etc.)
        speaker_map: Custom speaker mapping for voices.
            For edge: {'host': 'es-ES-AlvaroNeural', 'guest': 'es-ES-XimenaNeural'}
            For qwen_tts voice_clone: {'host': 'path/to/ref.wav', 'guest': 'path/to/ref.wav'}
        tts_kwargs: Extra engine-specific keyword arguments (e.g. task_type, device for qwen_tts).
        
    Returns:
        Dict with conversation_path, audio_path (if not skipped), metadata
    """
    from pathlib import Path
    
    # Load course data
    course_path = Path(course_path)
    with open(course_path, "r", encoding="utf-8") as f:
        course_data = json.load(f)
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = course_path.parent / "podcast"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate conversation
    engine_names = {"edge": "Edge TTS", "coqui": "Coqui TTS", "elevenlabs": "ElevenLabs", "chatterbox": "Chatterbox TTS", "openai_tts": "OpenAI TTS", "qwen_tts": "Qwen3-TTS"}
    engine_name = engine_names.get(tts_engine, tts_engine)
    print(f"🎙️ Generating podcast conversation for module {module_idx + 1} ({engine_name})...")
    
    conversation = generate_conversation(
        course_data=course_data,
        module_idx=module_idx,
        provider=provider,
        target_words=target_words,
    )
    
    # Extract context for metadata
    context = extract_module_context(course_data, module_idx)
    tts_language = get_tts_language(context["language"])
    
    # Save conversation JSON
    conv_filename = f"module_{module_idx + 1}_conversation.json"
    conv_path = output_path / conv_filename
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)
    print(f"✅ Conversation saved to: {conv_path}")
    
    result = {
        "conversation_path": str(conv_path),
        "conversation": conversation,
        "module_title": context["module_title"],
        "course_title": context["course_title"],
        "language": tts_language,
        "tts_engine": tts_engine,
    }
    
    # Generate audio if not skipped
    if not skip_tts:
        audio_filename = f"module_{module_idx + 1}.mp3"
        audio_path = output_path / audio_filename
        
        print(f"🔊 Synthesizing audio with {engine_name} (language={tts_language})...")
        
        # Get the path to background music (relative to project root)
        import os
        project_root = Path(__file__).parent.parent.parent
        music_path = project_root / "tools" / "podcast" / "background_music.mp3"
        
        if tts_engine == "edge":
            from tools.podcast import generate_podcast_edge
            
            generate_podcast_edge(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {context['module_title']}",
                artist="Adinhub",
                album=context["course_title"],
                track_number=module_idx + 1,
                music_path=str(music_path) if music_path.exists() else None,
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
                title=f"Module {module_idx + 1}: {context['module_title']}",
                artist="Adinhub",
                album=context["course_title"],
                track_number=module_idx + 1,
                music_path=str(music_path) if music_path.exists() else None,
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
                title=f"Module {module_idx + 1}: {context['module_title']}",
                artist="Adinhub",
                album=context["course_title"],
                track_number=module_idx + 1,
                music_path=str(music_path) if music_path.exists() else None,
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
                title=f"Module {module_idx + 1}: {context['module_title']}",
                artist="Adinhub",
                album=context["course_title"],
                track_number=module_idx + 1,
                music_path=str(music_path) if music_path.exists() else None,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        elif tts_engine == "qwen_tts":
            from tools.podcast import generate_podcast_qwen_tts
            
            extra = tts_kwargs or {}
            generate_podcast_qwen_tts(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {context['module_title']}",
                artist="Adinhub",
                album=context["course_title"],
                track_number=module_idx + 1,
                music_path=str(music_path) if music_path.exists() else None,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
                **extra,
            )
        else:
            from tools.podcast import generate_podcast
            
            generate_podcast(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {context['module_title']}",
                artist="Adinhub",
                album=context["course_title"],
                track_number=module_idx + 1,
                music_path=str(music_path) if music_path.exists() else None,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        
        print(f"✅ Audio saved to: {audio_path}")
        result["audio_path"] = str(audio_path)
    
    return result
