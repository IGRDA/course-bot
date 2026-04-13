import json
import os
import re
import threading
from operator import add
from typing import Annotated
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send
from LLMs.imagetext2text import create_vision_llm, resolve_vision_model_name
from LLMs.text2text import create_text_llm, resolve_text_model_name
from pydantic import BaseModel, Field

from tools.imagesearch.factory import create_image_search
from workflows.state import CourseState, ParagraphBlock

from .prompts import (
    FALLBACK_PARSE_PROMPT,
    IMAGE_RANKING_SYSTEM_PROMPT,
    ImageRankingResult,
    ImageRankingScore,
    create_image_ranking_prompt,
    image_query_prompt,
)

# Domains that block automated image downloads (403 Forbidden)
# These are filtered out at search time so we select alternate images instead
BLOCKED_IMAGE_DOMAINS = [
    "researchgate.net",
    "rgstatic.net",  # ResearchGate's CDN
]


def _filter_blocked_domains(urls: list[str]) -> list[str]:
    """Filter out URLs from domains known to block automated downloads."""
    filtered = []
    for url in urls:
        domain = urlparse(url).netloc.lower()
        is_blocked = any(blocked in domain for blocked in BLOCKED_IMAGE_DOMAINS)
        if not is_blocked:
            filtered.append(url)
    return filtered


# ---- Global concurrency control ----
# Global semaphore to cap concurrent Pixtral calls across all blocks/sections.
# This prevents concurrency multiplication (sections * blocks * per-block ranking).
_VISION_CALL_SEMAPHORE: threading.Semaphore | None = None


def _set_vision_call_semaphore(max_concurrent_calls: int) -> None:
    global _VISION_CALL_SEMAPHORE
    _VISION_CALL_SEMAPHORE = threading.Semaphore(max(1, int(max_concurrent_calls)))


def _vision_invoke(vision_llm, messages):
    if _VISION_CALL_SEMAPHORE is None:
        return vision_llm.invoke(messages)
    _VISION_CALL_SEMAPHORE.acquire()
    try:
        return vision_llm.invoke(messages)
    finally:
        _VISION_CALL_SEMAPHORE.release()


def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _looks_like_invalid_image_error(exc: BaseException) -> bool:
    """True for Pixtral 400-style invalid image URL errors (permanent, should drop URL)."""
    s = str(exc)
    return ("Error response 400" in s) or ("invalid_request_file" in s) or ('code":"3310' in s) or ("code': '3310" in s)


def _extract_invalid_image_url(exc: BaseException) -> str | None:
    """
    Try to extract the offending image URL from Pixtral 400 errors.

    Examples seen:
    - \"Image, 'https://...png', could not be loaded as a valid image\"
    - \"File could not be fetched from url 'https://...'\"
    - \"Image is an animated GIF. Image URL: https://...gif\"
    """
    s = str(exc)

    m = re.search(r"Image URL:\s*(https?://\S+)", s)
    if m:
        return m.group(1).rstrip("',\"")

    m = re.search(r"from url '([^']+)'", s)
    if m:
        return m.group(1)

    m = re.search(r"Image,\s*'([^']+)'", s)
    if m:
        return m.group(1)

    return None


# ---- State for individual block task ----
class BlockImageTask(BaseModel):
    """State for processing a single block's image"""

    course_title: str
    module_idx: int
    submodule_idx: int
    section_idx: int
    element_idx: int
    block_idx: int
    block_title: str
    content_preview: str
    # Config values needed for processing
    llm_provider: str
    image_provider: str
    k_images: int
    use_vision_ranking: bool
    vision_provider: str
    imagetext2text_concurrency: int
    vision_ranking_batch_size: int


# ---- State for aggregating results ----
class ImageGenerationState(BaseModel):
    """State for the image generation graph"""

    course_state: CourseState
    completed_blocks: Annotated[list[dict], add] = Field(default_factory=list)
    failed_blocks: Annotated[list[dict], add] = Field(default_factory=list)
    total_blocks: int = 0


def rank_images(
    image_urls: list[str],
    course_title: str,
    block_title: str,
    content_preview: str,
    text_llm_provider: str,
    vision_provider: str,
    imagetext2text_concurrency: int,
    batch_size: int,
) -> tuple[str, ImageRankingScore]:
    """
    Rank multiple images using batched vision LLM calls and return the best one.

    This avoids making one Pixtral call per image. Instead, it scores images in batches
    (up to `batch_size` images per call) and selects the best image locally.

    Scoring rubric (max 12 points):
    - Alignment (0,1,2,4,8): How well image matches topic AND block title
    - No Watermark (0 or 2): Clean images score higher
    - No Text (0 or 2): Images without text score higher

    Args:
        image_urls: List of image URLs to rank
        course_title: Course topic for context
        block_title: Block title for context
        content_preview: Content preview for context
        text_llm_provider: Provider for fallback parsing (e.g., 'mistral')
        vision_provider: Vision LLM provider (default: 'pixtral')
        imagetext2text_concurrency: Max parallel Pixtral calls (default: 5)

    Returns:
        Tuple of (best_image_url, score)
    """
    if not image_urls:
        raise ValueError("No images to rank")

    if len(image_urls) == 1:
        # No need to rank a single image, return with default score
        return image_urls[0], ImageRankingScore(alignment=4, no_watermark=2, has_text=2, total=8)

    # If semaphore wasn't initialized at run start, initialize best-effort here.
    if _VISION_CALL_SEMAPHORE is None:
        _set_vision_call_semaphore(imagetext2text_concurrency)

    # Create vision LLM once per block ranking
    vision_model_name = resolve_vision_model_name(vision_provider)
    vision_kwargs = {"temperature": 0.1}
    if vision_model_name:
        vision_kwargs["model_name"] = vision_model_name
    vision_llm = create_vision_llm(provider=vision_provider, **vision_kwargs)

    scored_images: list[tuple[str, ImageRankingScore]] = []
    for batch in _chunk_list(image_urls, batch_size):
        # We may need to drop invalid URLs and retry the same batch
        remaining = list(batch)

        while remaining:
            human_prompt = create_image_ranking_prompt(
                course_title=course_title,
                block_title=block_title,
                content_preview=content_preview,
                num_images=len(remaining),
            )

            content = [{"type": "text", "text": human_prompt}]
            for url in remaining:
                content.append({"type": "image_url", "image_url": {"url": url}})

            messages = [
                {"role": "system", "content": IMAGE_RANKING_SYSTEM_PROMPT},
                HumanMessage(content=content),
            ]

            try:
                response = _vision_invoke(vision_llm, messages)
                raw_response = response.content

                try:
                    json_str = raw_response
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    json_str = json_str.strip()

                    parsed_data = json.loads(json_str)
                    ranking_result = ImageRankingResult.model_validate(parsed_data)
                except (json.JSONDecodeError, Exception) as parse_error:
                    print(f"⚠️ Vision LLM output parsing failed: {parse_error}")
                    print(f"   Raw response: {raw_response[:500]}...")
                    print("   Attempting fallback parsing with text LLM...")
                    ranking_result = _fallback_parse_ranking(raw_response, len(remaining), text_llm_provider)

                # Ensure score count matches batch size
                if len(ranking_result.scores) != len(remaining):
                    while len(ranking_result.scores) < len(remaining):
                        ranking_result.scores.append(
                            ImageRankingScore(alignment=0, no_watermark=0, has_text=0, total=0)
                        )
                    ranking_result.scores = ranking_result.scores[: len(remaining)]

                for url, score in zip(remaining, ranking_result.scores, strict=False):
                    score.total = score.alignment + score.no_watermark + score.has_text
                    scored_images.append((url, score))

                # Batch succeeded; move to next batch
                break

            except Exception as e:
                # For 400 invalid-image style errors, drop the offending URL and retry the batch.
                if _looks_like_invalid_image_error(e):
                    bad_url = _extract_invalid_image_url(e)
                    if bad_url and bad_url in remaining:
                        remaining.remove(bad_url)
                        continue
                    # Heuristic: drop any GIFs (Pixtral rejects animated GIFs)
                    gif_urls = [u for u in remaining if u.lower().endswith(".gif")]
                    if gif_urls:
                        remaining.remove(gif_urls[0])
                        continue

                # Give up on this remaining batch and assign zero scores
                print(f"❌ Vision LLM call failed for batch of {len(remaining)} images: {e!s}")
                for url in remaining:
                    scored_images.append((url, ImageRankingScore(alignment=0, no_watermark=0, has_text=0, total=0)))
                break

    # Find best image by total score
    if not scored_images:
        # Fallback: return first URL with zero score
        return image_urls[0], ImageRankingScore(alignment=0, no_watermark=0, has_text=0, total=0)

    best_url, best_score = scored_images[0]
    for url, score in scored_images[1:]:
        if score.total > best_score.total:
            best_url = url
            best_score = score

    # Find index for logging
    best_idx = image_urls.index(best_url) + 1
    print(f"   🏆 Best image: #{best_idx} with score {best_score.total}/12")

    return best_url, best_score


def _fallback_parse_ranking(raw_text: str, num_images: int, text_llm_provider: str) -> ImageRankingResult:
    """
    Fallback parser using text LLM when vision LLM output is malformed.

    Args:
        raw_text: Raw response from vision LLM
        num_images: Expected number of images/scores
        text_llm_provider: Provider for text LLM (e.g., 'mistral')

    Returns:
        Parsed ImageRankingResult
    """
    # Create text LLM for parsing
    model_name = resolve_text_model_name(text_llm_provider)

    llm_kwargs = {"temperature": 0}  # Deterministic for parsing
    if model_name:
        llm_kwargs["model_name"] = model_name
    text_llm = create_text_llm(provider=text_llm_provider, **llm_kwargs)

    # Create parsing chain
    parse_chain = FALLBACK_PARSE_PROMPT | text_llm | StrOutputParser()

    try:
        parsed_response = parse_chain.invoke({"num_images": num_images, "raw_text": raw_text})

        # Clean up and parse
        json_str = parsed_response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        json_str = json_str.strip()

        parsed_data = json.loads(json_str)
        return ImageRankingResult.model_validate(parsed_data)

    except Exception as e:
        print(f"⚠️ Fallback parsing also failed: {e!s}")
        # Return default scores
        default_scores = [
            ImageRankingScore(alignment=0, no_watermark=0, has_text=0, total=0) for _ in range(num_images)
        ]
        return ImageRankingResult(scores=default_scores)


def plan_blocks(state: ImageGenerationState) -> dict:
    """Count total blocks that need images."""
    block_count = 0
    interactive_formats = ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]

    for module in state.course_state.modules:
        for submodule in module.submodules:
            for section in submodule.sections:
                if section.html:
                    for element in section.html:
                        if element.type in interactive_formats and isinstance(element.content, list):
                            block_count += len(element.content)

    print(f"📋 Planning image generation for {block_count} blocks")

    return {"total_blocks": block_count}


def continue_to_blocks(state: ImageGenerationState) -> list[Send]:
    """Fan-out: Create a Send for each block to process in parallel."""
    sends = []
    config = state.course_state.config
    interactive_formats = ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]

    # Always fetch multiple images to have fallbacks after filtering blocked domains
    # (even when vision ranking is disabled, we need alternatives if first result is blocked)
    k_images = config.num_images_to_fetch

    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                if section.html:
                    for e_idx, element in enumerate(section.html):
                        if element.type in interactive_formats and isinstance(element.content, list):
                            for b_idx, block in enumerate(element.content):
                                # Handle both dict and ParagraphBlock
                                if isinstance(block, dict):
                                    block_title = block.get("title", "")
                                    elements = block.get("elements", [])
                                else:
                                    block_title = block.title
                                    elements = block.elements

                                # Extract content preview from first paragraph
                                content_preview = ""
                                for elem in elements:
                                    elem_type = elem.get("type") if isinstance(elem, dict) else elem.type
                                    elem_content = elem.get("content") if isinstance(elem, dict) else elem.content
                                    if elem_type == "p" and isinstance(elem_content, str):
                                        content_preview = elem_content[:200]
                                        break

                                task = BlockImageTask(
                                    course_title=state.course_state.title,
                                    module_idx=m_idx,
                                    submodule_idx=sm_idx,
                                    section_idx=s_idx,
                                    element_idx=e_idx,
                                    block_idx=b_idx,
                                    block_title=block_title,
                                    content_preview=content_preview,
                                    llm_provider=config.text_llm_provider,
                                    image_provider=config.image_search_provider,
                                    k_images=k_images,
                                    use_vision_ranking=config.use_vision_ranking,
                                    vision_provider=config.vision_llm_provider,
                                    imagetext2text_concurrency=config.imagetext2text_concurrency,
                                    vision_ranking_batch_size=config.vision_ranking_batch_size,
                                )
                                sends.append(Send("generate_block_image", task))

    return sends


def generate_block_image(state: BlockImageTask) -> dict:
    """
    Generate image for a single block.

    This is a LangGraph node that processes ONE block:
    1. Generates an image search query using LLM
    2. Searches for images
    3. Optionally ranks images using vision LLM
    4. Returns the result for aggregation

    Args:
        state: BlockImageTask with block info and config

    Returns:
        Dict with completed_blocks or failed_blocks
    """
    # Create LLM for query generation
    model_name = resolve_text_model_name(state.llm_provider)
    llm_kwargs = {"temperature": 0.1}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=state.llm_provider, **llm_kwargs)

    # Create query generation chain
    query_chain = image_query_prompt | llm | StrOutputParser()

    # Get the image search function
    search_images = create_image_search(state.image_provider)

    try:
        # Generate image query using LLM
        query = query_chain.invoke(
            {
                "course_title": state.course_title,
                "block_title": state.block_title,
                "content_preview": state.content_preview,
            }
        )
        query = query.strip().replace("*", "").replace('"', "").strip()

        # Search for images
        print(f"   🔍 Searching: '{query}'")
        results = search_images(query, max_results=state.k_images)

        # Check for errors
        if not results:
            raise Exception(f"Image search ({state.image_provider}) returned no results for query '{query}'")

        # Filter out error results
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            error_msg = results[0].get("error", "Unknown error") if results else "No results"
            raise Exception(f"Image search ({state.image_provider}) failed for query '{query}': {error_msg}")

        # Extract image URLs
        image_urls = [r["url"] for r in valid_results]

        # Filter out domains that block automated downloads (e.g., ResearchGate)
        image_urls = _filter_blocked_domains(image_urls)
        if not image_urls:
            raise Exception(f"All image results were from blocked domains for query '{query}'")

        # Select image based on config
        if state.use_vision_ranking and len(image_urls) > 1:
            # Rank images and select best one using vision LLM
            print(f"   🎯 Ranking {len(image_urls)} images for '{state.block_title}'...")
            best_url, best_score = rank_images(
                image_urls=image_urls,
                course_title=state.course_title,
                block_title=state.block_title,
                content_preview=state.content_preview,
                text_llm_provider=state.llm_provider,
                vision_provider=state.vision_provider,
                imagetext2text_concurrency=state.imagetext2text_concurrency,
                batch_size=state.vision_ranking_batch_size,
            )

            image_data = {
                "type": "img",
                "query": query,
                "content": best_url,
                "ranking_score": best_score.total,
                "ranking_details": {
                    "alignment": best_score.alignment,
                    "no_watermark": best_score.no_watermark,
                    "has_text": best_score.has_text,
                },
            }
        else:
            # Just use first image (no ranking)
            best_url = image_urls[0]
            image_data = {
                "type": "img",
                "query": query,
                "content": best_url,
            }

        # Return completed block info
        return {
            "completed_blocks": [
                {
                    "module_idx": state.module_idx,
                    "submodule_idx": state.submodule_idx,
                    "section_idx": state.section_idx,
                    "element_idx": state.element_idx,
                    "block_idx": state.block_idx,
                    "image": image_data,
                }
            ]
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error generating image for block '{state.block_title}': {error_msg}")
        return {
            "failed_blocks": [
                {
                    "module_idx": state.module_idx,
                    "submodule_idx": state.submodule_idx,
                    "section_idx": state.section_idx,
                    "element_idx": state.element_idx,
                    "block_idx": state.block_idx,
                    "block_title": state.block_title,
                    "error": error_msg,
                }
            ]
        }


def reduce_blocks(state: ImageGenerationState) -> dict:
    """Fan-in: Merge all block images back into course state."""
    print(f"📦 Reducing {len(state.completed_blocks)} completed blocks")

    # Update course state with all generated images
    for block_info in state.completed_blocks:
        m_idx = block_info["module_idx"]
        sm_idx = block_info["submodule_idx"]
        s_idx = block_info["section_idx"]
        e_idx = block_info["element_idx"]
        b_idx = block_info["block_idx"]

        section = state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx]
        element = section.html[e_idx]
        block = element.content[b_idx]

        # Update the block with the image
        if isinstance(block, dict):
            block["image"] = block_info["image"]
        else:
            block.image = block_info["image"]

    # Log failures
    if state.failed_blocks:
        print(f"⚠️ {len(state.failed_blocks)} blocks failed:")
        for fb in state.failed_blocks:
            print(
                f"   - [{fb['block_title']}] in Module {fb['module_idx'] + 1}.{fb['submodule_idx'] + 1}.{fb['section_idx'] + 1}"
            )
        print("   (Continuing without these images)")

    success_count = len(state.completed_blocks)
    total_count = state.total_blocks
    if state.failed_blocks:
        print(f"✅ Image generation complete! ({success_count}/{total_count} blocks)")
    else:
        print(f"✅ All {total_count} block images generated successfully!")

    return {"course_state": state.course_state}


def build_image_generation_graph(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Build the image generation graph using Send for block-level parallelization.

    Args:
        max_retries: Number of retries for each block generation
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
    """
    graph = StateGraph(ImageGenerationState)

    # Configure retry policy with exponential backoff
    retry_policy = RetryPolicy(
        max_attempts=max_retries, initial_interval=initial_delay, backoff_factor=backoff_factor, max_interval=60.0
    )

    # Add nodes
    graph.add_node("plan_blocks", plan_blocks)
    graph.add_node("generate_block_image", generate_block_image, retry=retry_policy)
    graph.add_node("reduce_blocks", reduce_blocks)

    # Add edges
    graph.add_edge(START, "plan_blocks")
    graph.add_conditional_edges("plan_blocks", continue_to_blocks, ["generate_block_image"])
    graph.add_edge("generate_block_image", "reduce_blocks")
    graph.add_edge("reduce_blocks", END)

    return graph.compile()


def generate_all_section_images(
    course_state: CourseState,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    use_vision_ranking: bool | None = None,
    num_images_to_fetch: int | None = None,
    vision_provider: str | None = None,
) -> CourseState:
    """
    Main function to generate images for all section HTML blocks using LangGraph Send pattern.

    Uses block-level parallelism with LangGraph Send for proper trace context propagation.

    If use_vision_ranking is True, fetches K images for each block, ranks them using
    vision LLM (Pixtral), and selects the best one based on rubric scoring.
    If False, fetches only 1 image and uses it directly (faster, no vision LLM calls).

    Concurrency is controlled at two levels:
    - image_concurrency: Max parallel blocks (LangGraph max_concurrency)
    - imagetext2text_concurrency: Max parallel Pixtral calls (semaphore)

    Args:
        course_state: CourseState with HTML structures filled
        max_retries: Maximum number of retry attempts per block
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
        use_vision_ranking: Whether to use vision LLM for ranking (default from config)
        num_images_to_fetch: Number of images to fetch for ranking (default from config)
        vision_provider: Vision LLM provider for ranking (default from config)

    Returns:
        Updated CourseState with all images added to HTML blocks
    """
    image_provider = course_state.config.image_search_provider

    # Get settings from config (use config defaults if not overridden)
    if use_vision_ranking is None:
        use_vision_ranking = course_state.config.use_vision_ranking
    if num_images_to_fetch is None:
        num_images_to_fetch = course_state.config.num_images_to_fetch
    if vision_provider is None:
        vision_provider = course_state.config.vision_llm_provider

    # Get concurrency settings from config
    image_concurrency = course_state.config.image_concurrency
    imagetext2text_concurrency = course_state.config.imagetext2text_concurrency
    vision_ranking_batch_size = course_state.config.vision_ranking_batch_size

    # Enforce a global cap on concurrent Pixtral calls
    _set_vision_call_semaphore(imagetext2text_concurrency)

    print(
        f"🚀 Starting parallel image generation with provider={image_provider}, "
        f"max_retries={max_retries}, initial_delay={initial_delay}s, backoff_factor={backoff_factor}"
    )
    print(f"   📊 Concurrency: blocks={image_concurrency}, vision_calls={imagetext2text_concurrency}")

    if use_vision_ranking:
        print(
            f"   📸 Fetching {num_images_to_fetch} images per block, ranking with {vision_provider} (batch_size={vision_ranking_batch_size})"
        )
    else:
        print("   📸 Vision ranking disabled: picking first image result")

    # Build the graph with retry configuration
    graph = build_image_generation_graph(
        max_retries=max_retries, initial_delay=initial_delay, backoff_factor=backoff_factor
    )

    # Initialize state
    initial_state = ImageGenerationState(course_state=course_state)

    # Execute the graph with concurrency control
    result = graph.invoke(initial_state, config={"max_concurrency": image_concurrency})

    return result["course_state"]


if __name__ == "__main__":
    import json

    from workflows.state import CourseConfig

    INPUT_FILE = "/Users/inaki/Documents/Personal/course-generator-agent/output/Chess_masterclass_20251215_213954.json"
    # Generate OUTPUT_FILE from INPUT_FILE with _images suffix
    base, ext = os.path.splitext(INPUT_FILE)
    OUTPUT_FILE = f"{base}_images{ext}"

    # Hardcoded defaults - change these to override config values
    DEFAULT_LLM_PROVIDER = "mistral"
    DEFAULT_LANGUAGE = "English"
    DEFAULT_IMAGE_PROVIDER = "google"  # Options: bing | freepik | ddg
    USE_VISION_RANKING = False  # Set to True to rank images with Pixtral, False to pick first result
    NUM_IMAGES_TO_FETCH = 8  # Number of images to fetch for ranking (only used if USE_VISION_RANKING=True)
    VISION_PROVIDER = "pixtral"  # Vision LLM for ranking
    IMAGE_CONCURRENCY = 3  # Number of blocks to process in parallel
    IMAGETEXT2TEXT_CONCURRENCY = 5  # Number of Pixtral calls in parallel for image scoring
    VISION_RANKING_BATCH_SIZE = 8

    print("=" * 60)
    print("Image Search - Standalone Mode")
    print("=" * 60)
    print(f"Reading from: {INPUT_FILE}")

    # Load input JSON
    try:
        with open(INPUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found")
        print(f"   Please create an {INPUT_FILE} file with CourseState JSON structure")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {INPUT_FILE}: {e!s}")
        exit(1)

    # Parse as CourseState
    try:
        course_state = CourseState.model_validate(data)

        # Apply defaults if config is incomplete
        if not course_state.config:
            course_state.config = CourseConfig(
                title=course_state.title or "Untitled Course",
                text_llm_provider=DEFAULT_LLM_PROVIDER,
                image_search_provider=DEFAULT_IMAGE_PROVIDER,
                total_pages=10,
            )
        else:
            # Override config with hardcoded defaults
            course_state.config.image_search_provider = DEFAULT_IMAGE_PROVIDER
            course_state.config.use_vision_ranking = USE_VISION_RANKING
            course_state.config.num_images_to_fetch = NUM_IMAGES_TO_FETCH
            course_state.config.vision_llm_provider = VISION_PROVIDER
            course_state.config.image_concurrency = IMAGE_CONCURRENCY
            course_state.config.imagetext2text_concurrency = IMAGETEXT2TEXT_CONCURRENCY
            course_state.config.vision_ranking_batch_size = VISION_RANKING_BATCH_SIZE

        if not course_state.language:
            course_state.language = DEFAULT_LANGUAGE

    except Exception as e:
        print(f"❌ Error: Failed to parse CourseState: {e!s}")
        exit(1)

    # Count sections with HTML
    html_sections = 0
    total_sections = 0
    for module in course_state.modules:
        for submodule in module.submodules:
            for section in submodule.sections:
                total_sections += 1
                if section.html:
                    html_sections += 1

    print(f"📊 Course: {course_state.title}")
    print(f"   Language: {course_state.language}")
    print(f"   Modules: {len(course_state.modules)}")
    print(f"   Sections: {total_sections}")
    print(f"   Sections with HTML: {html_sections}")

    if html_sections == 0:
        print("⚠️  Warning: No sections have HTML content to add images to")
        print("   Exiting without changes")
        exit(0)

    # Generate images
    print("\n🚀 Starting image generation...")
    print(f"   Using LLM: {course_state.config.text_llm_provider}")
    print(f"   Image provider: {course_state.config.image_search_provider}")
    print(f"   Vision ranking: {'ENABLED' if USE_VISION_RANKING else 'DISABLED (picking first image)'}")
    if USE_VISION_RANKING:
        print(f"   Images to fetch per block: {NUM_IMAGES_TO_FETCH}")
        print(f"   Vision LLM for ranking: {VISION_PROVIDER}")
    print(f"   Concurrency: blocks={IMAGE_CONCURRENCY}, vision_calls={IMAGETEXT2TEXT_CONCURRENCY}")

    try:
        updated_state = generate_all_section_images(
            course_state,
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            use_vision_ranking=USE_VISION_RANKING,
            num_images_to_fetch=NUM_IMAGES_TO_FETCH,
            vision_provider=VISION_PROVIDER,
        )
    except Exception as e:
        print(f"\n❌ Error during image generation: {e!s}")
        print("   No output file created")
        exit(1)

    # Save output
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(updated_state.model_dump_json(indent=2, by_alias=True))
        print(f"✅ Success! Images added to {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error: Failed to save output: {e!s}")
        exit(1)

    # Print all image queries and scores
    print("\n" + "=" * 60)
    print("📝 All Generated Image Queries and Scores:")
    print("=" * 60)

    query_count = 0
    for m_idx, module in enumerate(updated_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                if section.html:
                    section_has_images = False
                    for element in section.html:
                        if element.type in [
                            "paragraphs",
                            "accordion",
                            "tabs",
                            "carousel",
                            "flip",
                            "timeline",
                            "conversation",
                        ] and isinstance(element.content, list):
                            for block in element.content:
                                if isinstance(block, (dict, ParagraphBlock)):
                                    block_obj = block if isinstance(block, ParagraphBlock) else ParagraphBlock(**block)
                                    if block_obj.image and "query" in block_obj.image:
                                        if not section_has_images:
                                            print(f"\n📍 Module {m_idx + 1}.{sm_idx + 1}.{s_idx + 1}: {section.title}")
                                            section_has_images = True
                                        query_count += 1
                                        score = block_obj.image.get("ranking_score")
                                        score_str = f"score: {score}/12" if score is not None else "no ranking"
                                        print(
                                            f'   {query_count}. [{block_obj.title}] → "{block_obj.image["query"]}" ({score_str})'
                                        )

    print(f"\n✨ Total queries generated: {query_count}")
    print("=" * 60)
