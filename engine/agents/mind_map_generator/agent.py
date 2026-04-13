"""
Mind Map Generator Agent.

Generates hierarchical concept maps for course modules using:
1. Extract key concepts from module submodules and sections
2. LLM generates mind map structure following Novak's methodology
3. Structured output parsing with retry on failures
4. Results are embedded directly in each Module
"""

import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from LLMs.text2text import create_text_llm, resolve_text_model_name

from workflows.state import (
    CourseState,
    MindmapNode,
    MindmapNodeData,
    MindmapRelation,
    MindmapRelationData,
    Module,
    ModuleMindmap,
)

from .prompts import mindmap_generation_prompt

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence (```json or ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove closing fence
        return "\n".join(lines)
    return text


def _extract_key_concepts(module: Module) -> list[str]:
    """
    Extract key concepts from module content.

    Collects titles and descriptions from submodules and sections
    to provide context for mind map generation.

    Args:
        module: Module to extract concepts from

    Returns:
        List of concept strings
    """
    concepts = []

    # Add module-level info
    concepts.append(f"Module: {module.title}")
    if module.description:
        concepts.append(f"Description: {module.description}")

    # Extract from submodules and sections
    for submodule in module.submodules:
        concepts.append(f"Submodule: {submodule.title}")
        if submodule.description:
            concepts.append(f"  - {submodule.description}")

        for section in submodule.sections:
            concepts.append(f"  Section: {section.title}")
            if section.description:
                concepts.append(f"    - {section.description}")
            # Include summary if available
            if section.summary:
                concepts.append(f"    Summary: {section.summary[:200]}...")

    return concepts


def _parse_mindmap_response(
    raw_response: str,
    module_idx: int,
    module_title: str,
) -> ModuleMindmap | None:
    """
    Parse the LLM response into a ModuleMindmap object.

    Args:
        raw_response: Raw JSON string from LLM
        module_idx: Module index for the mindmap
        module_title: Module title for the mindmap

    Returns:
        ModuleMindmap object or None if parsing fails
    """
    try:
        # Clean up the response
        clean_response = _strip_markdown_fences(raw_response)
        data = json.loads(clean_response)

        # Parse nodes
        nodes = []
        for node_data in data.get("nodes", []):
            node = MindmapNode(
                id=node_data["id"], level=node_data["level"], data=MindmapNodeData(label=node_data["data"]["label"])
            )
            nodes.append(node)

        # Parse relations
        relations = []
        for rel_data in data.get("relations", []):
            relation = MindmapRelation(
                id=rel_data["id"],
                source=rel_data["source"],
                target=rel_data["target"],
                data=MindmapRelationData(label=rel_data["data"]["label"]),
            )
            relations.append(relation)

        # Create the mindmap
        mindmap = ModuleMindmap(
            moduleIdx=data.get("moduleIdx", module_idx),
            title=data.get("title", module_title),
            nodes=nodes,
            relations=relations,
        )

        return mindmap

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse mindmap JSON: {e}")
        return None
    except KeyError as e:
        logger.error(f"Missing required field in mindmap response: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing mindmap: {e}")
        return None


def generate_module_mindmap(
    module: Module,
    course_title: str,
    language: str,
    provider: str = "mistral",
    max_nodes: int = 20,
    max_retries: int = 3,
) -> ModuleMindmap | None:
    """
    Generate a mind map for a single module.

    Uses LLM to create a hierarchical concept map based on module content,
    following Novak's concept map methodology.

    Args:
        module: Module to generate mind map for
        course_title: Course title for context
        language: Output language for labels
        provider: LLM provider for generation
        max_nodes: Maximum number of nodes in the map
        max_retries: Number of retry attempts on parse failure

    Returns:
        ModuleMindmap or None if generation fails
    """
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.3}  # Some creativity for varied maps
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)

    # Extract key concepts from module
    key_concepts = _extract_key_concepts(module)
    key_concepts_str = "\n".join(key_concepts)

    # Build the chain
    chain = mindmap_generation_prompt | llm | StrOutputParser()

    # Retry loop
    for attempt in range(max_retries):
        try:
            print(f"      🔄 Attempt {attempt + 1}/{max_retries}...")

            raw_response = chain.invoke(
                {
                    "course_title": course_title,
                    "module_title": module.title,
                    "module_description": module.description or "",
                    "module_idx": module.index,
                    "key_concepts": key_concepts_str,
                    "language": language,
                    "max_nodes": max_nodes,
                }
            )

            mindmap = _parse_mindmap_response(
                raw_response,
                module.index,
                module.title,
            )

            if mindmap:
                # Validate basic structure
                if len(mindmap.nodes) > 0 and len(mindmap.relations) > 0:
                    return mindmap
                else:
                    logger.warning("Mindmap has no nodes or relations, retrying...")

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")

    logger.error(f"Failed to generate mindmap for module '{module.title}' after {max_retries} attempts")
    return None


def generate_course_mindmaps(
    state: CourseState,
    provider: str | None = None,
    max_nodes: int | None = None,
) -> CourseState:
    """
    Generate mind maps for all modules in the course.

    Processes modules in parallel using ThreadPoolExecutor.

    Args:
        state: CourseState with modules
        provider: LLM provider (defaults to state.config.text_llm_provider)
        max_nodes: Max nodes per map (defaults to state.config.mindmap_max_nodes)

    Returns:
        Updated CourseState with mind maps embedded in modules
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    provider = provider or state.config.mindmap_llm_provider or state.config.text_llm_provider
    max_nodes = max_nodes or state.config.mindmap_max_nodes

    print(f"🧠 Generating mind maps for {len(state.modules)} modules...")
    print(f"   Max nodes per map: {max_nodes}")
    print(f"   Provider: {provider}")

    def _process_module(idx: int, module):
        print(f"\n   🗺️ Module {idx + 1}/{len(state.modules)}: {module.title}")
        mindmap = generate_module_mindmap(
            module=module,
            course_title=state.title,
            language=state.config.language,
            provider=provider,
            max_nodes=max_nodes,
            max_retries=state.config.max_retries,
        )
        if mindmap:
            print(f"      ✓ Module {idx + 1}: Generated {len(mindmap.nodes)} nodes, {len(mindmap.relations)} relations")
        else:
            print(f"      ✗ Module {idx + 1}: Failed to generate mind map")
        return idx, mindmap

    results: dict[int, object] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_process_module, idx, module): idx for idx, module in enumerate(state.modules)}
        for future in as_completed(futures):
            idx, mindmap = future.result()
            results[idx] = mindmap

    total_nodes = 0
    total_relations = 0
    successful = 0
    for idx in range(len(state.modules)):
        mindmap = results[idx]
        if mindmap:
            state.modules[idx].mindmap = mindmap
            total_nodes += len(mindmap.nodes)
            total_relations += len(mindmap.relations)
            successful += 1
        else:
            state.modules[idx].mindmap = None

    print("\n✅ Mind map generation complete!")
    print(f"   Successful: {successful}/{len(state.modules)} modules")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Total relations: {total_relations}")

    return state


def generate_mindmap_node(
    state: CourseState,
    config: RunnableConfig | None = None,
) -> CourseState:
    """
    LangGraph node for mind map generation.

    Generates mind maps for all modules and embeds in state.
    Only runs if state.config.generate_mindmap is True.

    Args:
        state: CourseState with modules
        config: LangGraph runtime config

    Returns:
        Updated CourseState with mind maps embedded in modules
    """
    if not state.config.generate_mindmap:
        print("🧠 Mind map generation disabled, skipping...")
        return state

    return generate_course_mindmaps(state)
