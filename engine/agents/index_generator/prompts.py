from langchain.prompts import PromptTemplate

# ============================================================================
# ROLE & EXPERTISE DEFINITION
# ============================================================================

CURRICULUM_ARCHITECT_ROLE = """You are an expert curriculum architect and instructional designer with deep expertise in:
- Learning progression design and scaffolded instruction
- Cognitive load management and optimal content chunking
- Backward design principles (starting from learning outcomes)

Your course structures should enable effective learning by ensuring:
- Concepts build logically upon each other
- Prerequisites are taught before dependent topics
- Complexity increases gradually and appropriately
- Each section has a clear, focused learning objective"""

# ============================================================================
# PEDAGOGICAL SCAFFOLDING PRINCIPLES
# ============================================================================

SCAFFOLDING_PRINCIPLES = """
=====================================
PEDAGOGICAL SCAFFOLDING REQUIREMENTS
=====================================
Your course structure MUST follow these learning design principles:

LOGICAL PROGRESSION:
- Earlier modules/sections should establish foundations for later content
- Complex topics should be broken into digestible components
- Each section should have ONE clear focus (not multiple competing ideas)
- Avoid circular dependencies (don't require knowledge not yet taught)

LEARNING OUTCOMES:
- Module titles should reflect what learners will ACHIEVE (not just topics)
- Section descriptions should hint at the specific skill or knowledge gained
- Structure should support measurable learning objectives

COGNITIVE LOAD MANAGEMENT:
- Balance theoretical and practical content across the course
- Avoid clustering all difficult concepts in one module
- Ensure variety in section types (concepts, examples, applications)

ENGAGEMENT ARCHITECTURE:
- Start with motivating, accessible content to build momentum
- Place challenging content after foundational concepts are secure
- End modules with synthesis or application to reinforce learning

CONTENT UNIQUENESS (CRITICAL):
- Each section must teach ONE unique concept not covered elsewhere in the course
- Before creating a section, ask: "What NEW knowledge does this add?"
- If two sections seem related, ensure they have CLEARLY DIFFERENT learning outcomes
- A learner should never think "I already learned this" when reading a new section

PREREQUISITE AWARENESS:
- For each module beyond the first, explicitly consider what prior knowledge is required
- Build a clear "knowledge graph" where later concepts reference earlier ones
- Use forward momentum: "Building on [previous concept]..." not "As we saw before..."
- Each module should feel like a natural next step, not an isolated topic

ANTI-REPETITION RULES:
- NEVER create multiple "history of X" or "background on Y" sections - consolidate into ONE
- NEVER create generic "Introduction to..." sections - be specific about what's introduced
- NEVER create "Overview" or "Summary" sections - every section must teach NEW content
- If content could fit in multiple sections, pick ONE and make others explicitly different
- Technical terms and definitions belong in EXACTLY one section (their first relevant use)
"""

# ============================================================================
# TEXTBOOK STRUCTURE PRINCIPLES
# ============================================================================

TEXTBOOK_STRUCTURE_PRINCIPLES = """
=====================================
TEXTBOOK-QUALITY STRUCTURE PRINCIPLES
=====================================
Create a course structure that mirrors professional textbook organization:

PHASE-BASED ORGANIZATION:
- Organize modules into logical learning phases:
  * Phase 1: Foundational Principles (core concepts everyone must know first)
  * Phase 2: Core Development (building systematic understanding, applying foundations)
  * Phase 3: Applications & Synthesis (real-world use, combining concepts)
- Early modules should be broader, later modules more specialized

EXPLICIT PREREQUISITE CHAINS:
- Each module MUST build explicitly on previous modules
- Before creating a module, ask: "What must a learner know BEFORE this?"
- No module should require knowledge from a LATER module
- Create a clear "knowledge ladder" where each rung supports the next

TITLE DIVERSITY (Mix These Styles):
- Question titles for engagement: "Why does X happen?" "What determines Y?"
- Problem statements: "The Problem of Z" "Challenges in W"
- Specific declarations: "The [Specific Phenomenon]" (not "Introduction to...")
- Multi-part markers for complex topics: "Topic I: Foundations" → "Topic II: Applications"

TOPIC OWNERSHIP:
- Each key concept belongs to EXACTLY one section (its "home")
- If a concept appears in multiple places, decide which section "owns" it
- Other sections may reference it briefly but not re-teach it
- Technical terms are defined ONCE, in their most relevant section

STRUCTURAL VARIETY:
- Foundational modules: more detailed breakdown (more submodules/sections)
- Application modules: fewer but deeper sections
- Avoid padding simpler topics with artificial subdivisions
- Let complexity determine depth, not arbitrary uniformity
"""

# ============================================================================
# VERIFICATION MANDATE
# ============================================================================

STRUCTURE_VERIFICATION = """
=====================================
FINAL VERIFICATION (MANDATORY)
=====================================
Before outputting, verify that your structure:
✓ Has exactly the required number of modules, submodules, and sections
✓ Follows logical learning progression (foundations before advanced topics)
✓ Uses clear, descriptive titles
✓ Has descriptions that clarify the learning objective of each element
✓ Maintains the EXACT course title as provided (no translation or modification)
✓ All titles and descriptions are in the target language
✓ Conforms exactly to the JSON schema with no extra fields
"""

# ============================================================================
# MAIN GENERATION TEMPLATE
# ============================================================================

GEN_TEMPLATE = """
{curriculum_role}

{scaffolding_principles}

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Optional description/context: "{course_description}"
- Target language: {language}

{audience_guidelines}

=====================================
STRUCTURE CONSTRAINTS (EXACT)
=====================================
- n_modules: {n_modules}
- n_submodules per module: {n_submodules}
- n_sections per submodule: {n_sections}
- Target words per section (for reference only): {n_words}

=====================================
CONTENT RULES
=====================================
1. The COURSE TITLE must remain EXACTLY as provided: "{course_title}" (do not translate or modify it)
2. Every module, submodule, and section must have:
   - A concise, descriptive title (clear learning focus)
   - A brief description (1-2 sentences explaining the learning objective)
3. All titles and descriptions must be written in {language}
4. Only include structural fields: title, index, description
5. Ensure all lists have the EXACT specified lengths
6. Use only the fields defined in the schema (extra fields are forbidden)

=====================================
QUALITY CHECKLIST FOR TITLES
=====================================
Good titles:
✓ Clearly indicate what will be learned
✓ Use accessible language
✓ Are specific and focused
✓ Build logically from previous sections

Avoid:
✗ Vague titles like "Introduction" or "Advanced Topics"
✗ Overly technical jargon without context
✗ Titles that don't indicate learning outcomes
✗ Repetitive or overlapping titles

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON that conforms to the schema below.
- No code fences, no extra commentary
- The top-level object contains the course structure skeleton

IMPORTANT REMINDERS:
- Keep the course title EXACTLY as provided: "{course_title}"
- Write module titles, submodule titles, section titles, and descriptions in {language}
- Ensure logical learning progression throughout

{structure_verification}

{format_instructions}
"""

gen_prompt = PromptTemplate(
    template=GEN_TEMPLATE,
    input_variables=[
        "course_title", "course_description", "language",
        "n_modules", "n_submodules", "n_sections", "n_words",
        "format_instructions", "audience_guidelines"
    ],
    partial_variables={
        "curriculum_role": CURRICULUM_ARCHITECT_ROLE,
        "scaffolding_principles": SCAFFOLDING_PRINCIPLES,
        "structure_verification": STRUCTURE_VERIFICATION
    }
)


# ============================================================================
# RETRY TEMPLATE
# ============================================================================

retry_template = """Your previous response doesn't follow the required schema and fails parsing.
Fix ONLY the formatting to comply with the schema; do not change the meaning or counts.

=====================================
EXPECTED SCHEMA
=====================================
{schema}

=====================================
CONSTRAINTS
=====================================
- Return pure JSON (no code fences, no extra text)
- Use exactly the specified counts and field names
- No extra fields are allowed
- Only include structural fields: title, index, description
- All titles and descriptions must be written in {language}
- Ensure logical learning progression is maintained
- Course title must remain exactly as originally provided
"""

retry_prompt = PromptTemplate.from_template(retry_template)


# ============================================================================
# HIERARCHICAL GENERATION TEMPLATES (3-Step)
# ============================================================================

# Step 1: Generate module titles only (smallest output - easiest to get correct count)
MODULES_ONLY_TEMPLATE = """You are a curriculum architect creating a course outline.

TASK: Generate ONLY the module titles for a course. No submodules, no sections yet.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Target language: {language}

{audience_guidelines}

=====================================
EXACT REQUIREMENT
=====================================
Generate EXACTLY {n_modules} module titles. No more, no less.

=====================================
RULES
=====================================
1. Generate ONLY module titles (no submodules, no sections, no descriptions)
2. Titles must be in {language}
3. Titles should follow logical learning progression
4. Each title should be clear and specific
5. Avoid vague titles like "Introduction" or "Advanced Topics"

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON with exactly {n_modules} modules.
No code fences, no commentary.

{format_instructions}
"""

modules_only_prompt = PromptTemplate(
    template=MODULES_ONLY_TEMPLATE,
    input_variables=["course_title", "language", "n_modules", "format_instructions", "audience_guidelines"]
)

# Step 1 with research context
MODULES_ONLY_WITH_RESEARCH_TEMPLATE = """You are a curriculum architect creating a course outline informed by research.

TASK: Generate ONLY the module titles for a course. No submodules, no sections yet.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Target language: {language}

{audience_guidelines}

=====================================
RESEARCH CONTEXT
=====================================
Key topics to cover (distribute these across modules):
{key_topics}

Learning objectives to achieve:
{learning_objectives}

{textbook_principles}

=====================================
EXACT REQUIREMENT
=====================================
Generate EXACTLY {n_modules} module titles. No more, no less.

=====================================
MODULE GENERATION RULES
=====================================
1. Generate ONLY module titles (no submodules, no sections, no descriptions)
2. Titles must be in {language}
3. Follow a clear learning progression:
   - First modules: foundational concepts (what must everyone know first?)
   - Middle modules: core development (building on foundations)
   - Later modules: applications and synthesis (putting it together)

4. TOPIC OWNERSHIP: Each key topic from research should belong to EXACTLY ONE module
   - Assign related topics to the same module
   - No topic should span multiple modules
   - If a topic is too broad, it becomes its own module

5. PREREQUISITE THINKING: Before adding each module, ask:
   - "What must a learner already know to start this module?"
   - "Does this build on previous modules?"
   - Modules should form a clear "knowledge ladder"

6. TITLE QUALITY - Use varied, specific styles:
   - Questions: "Why does X happen?" "What determines Y?"
   - Problem statements: "The Problem of Z"
   - Specific topics: "The [Phenomenon]" (NOT "Introduction to...")
   - Multi-part for complex topics: "Topic I: Basics" then "Topic II: Applications"

7. AVOID:
   - Generic titles like "Introduction" or "Basics" or "Advanced Topics"
   - Multiple modules covering the same ground
   - Titles that don't clearly indicate what will be learned

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON with exactly {n_modules} modules.
No code fences, no commentary.

{format_instructions}
"""

modules_only_with_research_prompt = PromptTemplate(
    template=MODULES_ONLY_WITH_RESEARCH_TEMPLATE,
    input_variables=["course_title", "language", "key_topics", "learning_objectives", "n_modules", "format_instructions", "audience_guidelines"],
    partial_variables={
        "textbook_principles": TEXTBOOK_STRUCTURE_PRINCIPLES
    }
)

# Step 2: Add submodules to existing modules
ADD_SUBMODULES_TEMPLATE = """You are a curriculum architect expanding a course outline.

TASK: Add submodule titles to each module. No sections yet.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Target language: {language}

=====================================
EXISTING MODULES
=====================================
{modules_structure}

=====================================
REQUIREMENT
=====================================
Add EXACTLY {n_submodules} submodules to EACH module.
Keep all module titles EXACTLY as provided.

=====================================
RULES
=====================================
1. Keep module titles unchanged
2. Add exactly {n_submodules} submodules per module
3. Submodule titles must be in {language}
4. Submodules should break down the module topic logically
5. No sections yet, no descriptions

=====================================
OUTPUT FORMAT
=====================================
Return the COMPLETE structure with submodules added.
No code fences, no commentary.

{format_instructions}
"""

add_submodules_prompt = PromptTemplate(
    template=ADD_SUBMODULES_TEMPLATE,
    input_variables=["course_title", "language", "modules_structure", "n_submodules", "format_instructions"]
)

# Step 3: Add sections to existing submodules
ADD_SECTIONS_TEMPLATE = """You are a curriculum architect completing a course outline.

TASK: Add section titles to each submodule.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Target language: {language}

=====================================
EXISTING STRUCTURE
=====================================
{structure_with_submodules}

=====================================
REQUIREMENT
=====================================
Add EXACTLY {n_sections} sections to EACH submodule.
Keep all module and submodule titles EXACTLY as provided.

=====================================
RULES
=====================================
1. Keep all existing titles unchanged
2. Add exactly {n_sections} sections per submodule
3. Section titles must be in {language}
4. Sections should cover specific topics within the submodule
5. No descriptions yet

=====================================
OUTPUT FORMAT
=====================================
Return the COMPLETE structure with sections added.
No code fences, no commentary.

{format_instructions}
"""

add_sections_prompt = PromptTemplate(
    template=ADD_SECTIONS_TEMPLATE,
    input_variables=["course_title", "language", "structure_with_submodules", "n_sections", "format_instructions"]
)

# Final step: Add descriptions to complete structure
EXPAND_DESCRIPTIONS_TEMPLATE = """You are a curriculum designer adding descriptions to an existing course outline.

TASK: Add descriptions to each element in the provided course structure.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Target language: {language}

{audience_guidelines}

=====================================
EXISTING STRUCTURE (TITLES ONLY)
=====================================
{titles_structure}

=====================================
YOUR TASK
=====================================
Add a description field to each module, submodule, and section.
Keep all titles EXACTLY as provided - do not change them.

Description guidelines:
- Module descriptions: 2-3 sentences explaining the module's learning goals
- Submodule descriptions: 1-2 sentences about the submodule's focus
- Section descriptions: 1-2 sentences about specific content covered
- Adapt the language and complexity to match the target audience

=====================================
OUTPUT FORMAT
=====================================
Return the COMPLETE structure with descriptions added.
Return ONLY valid JSON. No code fences, no commentary.

{format_instructions}
"""

expand_descriptions_prompt = PromptTemplate(
    template=EXPAND_DESCRIPTIONS_TEMPLATE,
    input_variables=[
        "course_title", "language", "titles_structure",
        "format_instructions", "audience_guidelines"
    ]
)


# ============================================================================
# RESEARCH PHASE PROMPTS
# ============================================================================

# ---- Query Generation Prompt ----
QUERY_GENERATION_TEMPLATE = """You are a research assistant helping to gather information for course development.

Given a course title and description, generate {max_queries} diverse search queries that will help understand:
1. The core concepts and canonical topics in this domain
2. Common learning progressions and prerequisites
3. Real-world applications and examples
4. Key terminology and definitions
5. How existing courses and textbooks structure this topic

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Description: "{course_description}"

=====================================
QUERY GENERATION RULES
=====================================
1. ALWAYS generate queries in ENGLISH for best search results (regardless of course language)
2. Make queries specific enough to return useful educational content
3. Cover different aspects: fundamentals, applications, examples, prerequisites
4. Avoid overly broad or vague queries
5. Include at least one query about prerequisites or foundational knowledge
6. Use standard English terminology for the domain
7. IMPORTANT: Include at least 2-3 queries to find existing course structures:
   - Search for textbook table of contents: "[topic] textbook table of contents"
   - Search for course syllabi: "[topic] course syllabus university"
   - Search for curriculum outlines: "[topic] curriculum outline" or "[topic] course outline"
   - Search for online course structures: "[topic] online course modules"
   These queries help discover how experts structure learning progressions for this topic.

=====================================
OUTPUT FORMAT
=====================================
Return ONLY a JSON array of {max_queries} search query strings in ENGLISH.
No code fences, no extra commentary.

Example output format:
["machine learning fundamentals concepts", "machine learning prerequisites mathematics statistics", "machine learning textbook table of contents", "machine learning course syllabus university", "real world applications of ML"]
"""

query_generation_prompt = PromptTemplate(
    template=QUERY_GENERATION_TEMPLATE,
    input_variables=["course_title", "course_description", "max_queries"]
)


# ---- Research Synthesis Prompt ----
RESEARCH_SYNTHESIS_TEMPLATE = """You are a curriculum research specialist who synthesizes raw information into structured course planning documents.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Description: "{course_description}"

=====================================
RAW RESEARCH DATA
=====================================
The following information was gathered from web searches about this topic:

{raw_research}

CRITICAL: Before using any search result, verify it is:
- Directly relevant to the course topic
- From authoritative/educational sources
- Current and factually accurate
- Not an error message or empty result
Discard irrelevant, outdated, or low-quality results entirely.

=====================================
YOUR TASK
=====================================
Analyze the raw research and synthesize it into a structured course research document.
Extract the most relevant educational insights.

IMPORTANT: ALL output must be in ENGLISH. This research document will be used to inform 
course structure generation, which will later be translated to the target language.

=====================================
OUTPUT REQUIREMENTS
=====================================
Generate a JSON object with these fields (ALL IN ENGLISH):

1. "course_summary": A comprehensive 3-5 sentence summary of the topic, explaining:
   - What the subject matter covers
   - Why it's important/relevant
   - The scope and depth appropriate for learners

2. "learning_objectives": Array of 4-6 specific, measurable learning objectives
   - Start with action verbs (Understand, Apply, Analyze, Create, etc.)
   - Be specific about what learners will achieve

3. "assumed_prerequisites": Array of 2-4 prerequisites learners should have
   - List foundational knowledge or skills needed
   - Be specific (e.g., "basic algebra" not just "math")

4. "out_of_scope": Array of 2-4 topics explicitly NOT covered
   - Help define boundaries of the course
   - Mention related but excluded advanced topics

5. "key_topics": Array of 6-10 canonical topics that MUST be covered
   - List in logical learning order
   - Include foundational and advanced topics
   - These will guide the course structure

=====================================
CONSTRAINTS
=====================================
- ALL text must be written in ENGLISH
- Be concise but comprehensive
- Base your synthesis on the research data provided
- If research is limited, use your domain expertise to supplement
- Return ONLY valid JSON (no code fences, no extra text)

{format_instructions}
"""

research_synthesis_prompt = PromptTemplate(
    template=RESEARCH_SYNTHESIS_TEMPLATE,
    input_variables=["course_title", "course_description", "raw_research", "format_instructions"]
)


# ---- Enhanced Index Generation with Research Context ----
GEN_WITH_RESEARCH_TEMPLATE = """
{curriculum_role}

{scaffolding_principles}

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Optional description/context: "{course_description}"
- Target language: {language}

{audience_guidelines}

=====================================
RESEARCH-BASED GUIDANCE
=====================================
The following research has been conducted to inform the course structure:

COURSE SUMMARY:
{course_summary}

LEARNING OBJECTIVES:
{learning_objectives}

ASSUMED PREREQUISITES:
{assumed_prerequisites}

OUT OF SCOPE:
{out_of_scope}

KEY TOPICS TO COVER:
{key_topics}

IMPORTANT: Use this research to create a well-informed, canonical course structure.
Ensure the key topics are appropriately distributed across modules and sections.

=====================================
STRUCTURE CONSTRAINTS (EXACT)
=====================================
- n_modules: {n_modules}
- n_submodules per module: {n_submodules}
- n_sections per submodule: {n_sections}
- Target words per section (for reference only): {n_words}

=====================================
CONTENT RULES
=====================================
1. The COURSE TITLE must remain EXACTLY as provided: "{course_title}" (do not translate or modify it)
2. Every module, submodule, and section must have:
   - A concise, descriptive title (clear learning focus)
   - A brief description (2-3 sentences explaining the learning objective AND key concepts covered)
3. All titles and descriptions must be written in {language}
4. Only include structural fields: title, index, description
5. Ensure all lists have the EXACT specified lengths
6. Use only the fields defined in the schema (extra fields are forbidden)
7. INCORPORATE the key topics from research into appropriate sections
8. Ensure the learning objectives are addressed by the structure

=====================================
QUALITY CHECKLIST FOR TITLES
=====================================
Good titles:
✓ Clearly indicate what will be learned
✓ Use accessible language
✓ Are specific and focused
✓ Build logically from previous sections
✓ Reflect the key topics from research

Avoid:
✗ Vague titles like "Introduction" or "Advanced Topics"
✗ Overly technical jargon without context
✗ Titles that don't indicate learning outcomes
✗ Repetitive or overlapping titles

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON that conforms to the schema below.
- No code fences, no extra commentary
- The top-level object contains the course structure skeleton

IMPORTANT REMINDERS:
- Keep the course title EXACTLY as provided: "{course_title}"
- Write module titles, submodule titles, section titles, and descriptions in {language}
- Ensure logical learning progression throughout
- Descriptions should be more detailed (2-3 sentences) since we have research context

{structure_verification}

{format_instructions}
"""

gen_with_research_prompt = PromptTemplate(
    template=GEN_WITH_RESEARCH_TEMPLATE,
    input_variables=[
        "course_title", "course_description", "language",
        "course_summary", "learning_objectives", "assumed_prerequisites",
        "out_of_scope", "key_topics",
        "n_modules", "n_submodules", "n_sections", "n_words",
        "format_instructions", "audience_guidelines"
    ],
    partial_variables={
        "curriculum_role": CURRICULUM_ARCHITECT_ROLE,
        "scaffolding_principles": SCAFFOLDING_PRINCIPLES,
        "structure_verification": STRUCTURE_VERIFICATION
    }
)


# ============================================================================
# SECTION SUMMARY GENERATION PROMPT
# ============================================================================

SUMMARY_GENERATION_TEMPLATE = """You are an expert curriculum planner creating content summaries for course sections.

=====================================
MODULE CONTEXT
=====================================
- Course Title: "{course_title}"
- Module Title: "{module_title}"
- Module Description: "{module_description}"
- Target Language: {language}

{audience_guidelines}

=====================================
SECTIONS IN THIS MODULE
=====================================
The following sections belong to this module (across all submodules):

{sections_list}

=====================================
YOUR TASK
=====================================
Generate a 3-line summary for EACH section describing what specific content it will cover.

CRITICAL REQUIREMENTS:
1. Each summary must be EXACTLY 3 lines (sentences)
2. Summaries must be DISTINCT - no overlapping content between sections
3. Summaries should be SPECIFIC about what will be taught (not vague)
4. Each summary should complement the others - together they form a complete module
5. Write all summaries in {language}
6. Adapt language complexity and style to match the target audience

ANTI-PATTERNS TO AVOID:
- Generic phrases like "This section covers..." or "Students will learn..."
- Overlapping content between sections
- Vague descriptions that could apply to multiple sections
- Repeating the section title as the summary

=====================================
OUTPUT FORMAT
=====================================
Return a JSON object mapping each section title (exactly as provided) to its 3-line summary.

Example format:
{{
    "Section Title 1": "Line 1 of summary.\\nLine 2 of summary.\\nLine 3 of summary.",
    "Section Title 2": "Line 1 of summary.\\nLine 2 of summary.\\nLine 3 of summary."
}}

IMPORTANT:
- Use the EXACT section titles as keys (copy them precisely)
- Separate the 3 lines with \\n (newline characters)
- Return ONLY valid JSON (no code fences, no extra text)
- All summaries must be in {language}
"""

summary_generation_prompt = PromptTemplate(
    template=SUMMARY_GENERATION_TEMPLATE,
    input_variables=[
        "course_title", "module_title", "module_description",
        "language", "sections_list", "audience_guidelines"
    ]
)


# ============================================================================
# CORRECTION PROMPT FOR JSON RETRY
# ============================================================================

correction_prompt = PromptTemplate.from_template("""The previous JSON output had errors.

ERROR:
{error}

ORIGINAL OUTPUT:
{completion}

Fix the JSON errors and return ONLY valid JSON:
{format_instructions}

Common issues:
1. Missing commas between fields
2. Unescaped quotes inside strings (use \\" not ")
3. No markdown fences (no ```json)
""")
