"""Prompt templates for mind map generation."""

from langchain_core.prompts import ChatPromptTemplate

# System prompt for mind map generation
MINDMAP_SYSTEM_PROMPT = """You are an expert in educational concept mapping, specializing in Novak's Concept Map methodology.

Your task is to generate a HIERARCHICAL CONCEPT MAP (Mapa Conceptual) based on the provided module content.

CRITICAL UNDERSTANDING - CONCEPT MAP vs MIND MAP:
- A Concept Map is HIERARCHICAL and PROPOSITIONAL (Concept-Connector-Concept)
- It is NOT a radial Mind Map - it has clear levels and meaningful connections
- Every connection MUST have a linking phrase that forms a valid proposition

STRUCTURE REQUIREMENTS (Novak's Model):
1. Level 0 (Root): A single central node representing the main topic of the module
2. Level 1: Primary concepts directly related to the root (3-5 nodes recommended)
3. Level 2+: Sub-concepts that elaborate on Level 1 concepts
4. Cross-links between branches are valuable but not required

NODE CONSTRAINTS:
- Maximum {max_nodes} total nodes (including root) for visual clarity
- Node labels should be SHORT (1-4 words), noun-based concepts
- Avoid verbs in node labels - save verbs for relation connectors
- Each node must have a unique ID: "root" for the root, "n1", "n2", etc.

RELATION REQUIREMENTS (CRITICAL):
- EVERY relation MUST have a linking phrase (connector)
- Linking phrases should be VERBS or SHORT PHRASES that form valid propositions
- Examples of good connectors: "includes", "causes", "is part of", "requires", "enables", "affects"
- The proposition "Source + Connector + Target" must make grammatical sense
- Each relation must have a unique ID: "r1", "r2", etc.

LANGUAGE:
- Generate all labels and connectors in {language}
- Use appropriate academic vocabulary for the subject matter

OUTPUT FORMAT:
Return ONLY valid minified JSON matching the exact schema provided. No explanations or markdown."""

# Human prompt for mind map generation
MINDMAP_HUMAN_PROMPT = """Generate a concept map for this educational module:

**Course Title:** {course_title}
**Module Title:** {module_title}
**Module Description:** {module_description}

**Key Concepts from Module Content:**
{key_concepts}

**Language:** {language}
**Maximum Nodes:** {max_nodes}

Generate the concept map following this exact JSON schema:
{{
  "moduleIdx": {module_idx},
  "title": "{module_title}",
  "nodes": [
    {{ "id": "root", "level": 0, "data": {{ "label": "Main Topic" }} }},
    {{ "id": "n1", "level": 1, "data": {{ "label": "Concept 1" }} }},
    {{ "id": "n2", "level": 2, "data": {{ "label": "Sub-concept" }} }}
  ],
  "relations": [
    {{ "id": "r1", "source": "root", "target": "n1", "data": {{ "label": "includes" }} }},
    {{ "id": "r2", "source": "n1", "target": "n2", "data": {{ "label": "contains" }} }}
  ]
}}

IMPORTANT RULES:
1. Root node (level 0) should capture the module's main topic
2. Every node except root must be connected via a relation
3. Every relation must have a meaningful linking phrase
4. Use the provided key concepts to build the hierarchy
5. Prioritize the most important {max_nodes} concepts if there are more

Return ONLY the JSON object, no markdown or explanations:"""

# Create the ChatPromptTemplate
mindmap_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", MINDMAP_SYSTEM_PROMPT),
    ("human", MINDMAP_HUMAN_PROMPT),
])

