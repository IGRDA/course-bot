"""
PDF Index Generator Prompts

Prompts for 3-step hierarchical extraction from PDF syllabus:
1. Extract modules (titles + durations)
2. Extract submodules (topic headings)
3. Extract sections (bullet points/learning outcomes)
Plus enrichment and summary generation prompts.
"""

from langchain.prompts import PromptTemplate


# ============================================================================
# STEP 1: MODULE EXTRACTION PROMPT
# ============================================================================

EXTRACT_MODULES_TEMPLATE = """You are analyzing a course syllabus PDF to extract the module structure.

TASK: Extract ALL modules from the PDF with their titles, durations, and descriptions (from OBJETIVO).

=====================================
PDF CONTENT
=====================================
{pdf_markdown}

=====================================
EXTRACTION RULES (CRITICAL)
=====================================
1. Extract ONLY modules that ACTUALLY EXIST in the PDF
2. Do NOT make up or invent any modules
3. Do NOT skip or forget any modules - extract ALL of them
4. Preserve module titles EXACTLY as written in the PDF
5. Extract duration in hours if specified (e.g., "70 horas" → 70.0)
6. If duration is not specified, use 0.0
7. Extract the OBJETIVO text as the module description (EXACTLY as written)

=====================================
WHERE TO FIND MODULES
=====================================
Look for patterns like:
- "Módulo 1:", "Module 1:", "MÓDULO DE FORMACIÓN 1:"
- Tables with module listings
- Numbered sections with module titles
- "Unidad 1:", "Unit 1:", "Tema 1:"

WHERE TO FIND MODULE DESCRIPTIONS:
- The "OBJETIVO" section that follows each module title
- This contains the learning goal/objective for the module
- Extract this text EXACTLY as the description field

Example:
  MÓDULO DE FORMACIÓN 1: ABCDEFG...
  OBJETIVO
  JKLMNOP...
  
  → title: "ABCDEFG..."
  → description: "JKLMNOP..."

=====================================
OUTPUT REQUIREMENTS
=====================================
- course_title: Extract the main course title from the PDF header
- modules: List of ALL modules with:
  - title: Module title exactly as in PDF
  - duration_hours: Duration in hours (0.0 if not specified)
  - description: The OBJETIVO text for this module (empty string if not found)

Target language for output: {language}

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON. No code fences, no commentary.

{format_instructions}
"""

extract_modules_prompt = PromptTemplate(
    template=EXTRACT_MODULES_TEMPLATE,
    input_variables=["pdf_markdown", "language", "format_instructions"]
)


# ============================================================================
# STEP 2: SUBMODULE EXTRACTION PROMPT
# ============================================================================

EXTRACT_SUBMODULES_TEMPLATE = """You are analyzing a course syllabus PDF to extract submodules for each module.

TASK: For each module, extract the submodules (topic headings/main themes).

=====================================
PDF CONTENT
=====================================
{pdf_markdown}

=====================================
ALREADY EXTRACTED MODULES
=====================================
{modules_structure}

=====================================
EXTRACTION RULES (CRITICAL)
=====================================
1. Extract ONLY submodules that ACTUALLY EXIST in the PDF for each module
2. Do NOT make up or invent any submodules
3. Do NOT skip any submodules - extract ALL of them
4. Keep all module information (title, duration_hours, description) EXACTLY unchanged
5. Submodules are the main topic headings WITHIN each module

=====================================
WHERE TO FIND SUBMODULES
=====================================
Look for patterns within each module like:
- Main topic headings under "Conocimientos / Capacidades cognitivas y prácticas"
- Bold or underlined topic headings that have bullet points beneath them
- Lines that end with a colon and have sub-items beneath

Submodule patterns (these are headings with bullets beneath):
- "ABCDEFG..."
- "HIJKLMN..."
- "OPQRSTU..."
- "VWXYZAB..."

=====================================
IGNORE SOFT SKILLS
=====================================
Do NOT create submodules from "Habilidades de gestión, personales y sociales".
Only extract submodules from the technical/knowledge sections.

=====================================
IF NO CLEAR SUBMODULES EXIST
=====================================
If a module doesn't have clear submodule divisions:
- Create 1 submodule with the same title as the module
- This preserves the structure while indicating no subdivision exists

=====================================
OUTPUT REQUIREMENTS
=====================================
- Keep course_title unchanged
- Keep all module titles, durations, AND descriptions EXACTLY unchanged
- Add submodules array to each module

Target language for output: {language}

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON. No code fences, no commentary.

{format_instructions}
"""

extract_submodules_prompt = PromptTemplate(
    template=EXTRACT_SUBMODULES_TEMPLATE,
    input_variables=["pdf_markdown", "modules_structure", "language", "format_instructions"]
)


# ============================================================================
# STEP 3: SECTION EXTRACTION PROMPT
# ============================================================================

EXTRACT_SECTIONS_TEMPLATE = """You are analyzing a course syllabus PDF to extract sections for each submodule.

TASK: For each submodule, extract the sections (specific learning points/bullet items).

=====================================
PDF CONTENT
=====================================
{pdf_markdown}

=====================================
ALREADY EXTRACTED STRUCTURE
=====================================
{structure_with_submodules}

=====================================
EXTRACTION RULES (CRITICAL)
=====================================
1. Extract ONLY sections that ACTUALLY EXIST in the PDF
2. Do NOT make up or invent any sections
3. Do NOT skip any sections - extract ALL of them
4. Keep all module and submodule information unchanged (including description)
5. Sections are the specific learning points WITHIN each submodule

=====================================
CRITICAL: ONE BULLET = ONE SECTION
=====================================
Each bullet point (•, -, *) under a submodule heading = ONE separate section.

WRONG (collapsing bullets into one):
  title: "ABCDEFG. HIJKLMN. OPQRSTU. VWXYZAB."

CORRECT (each bullet is a section):
  sections: [
    {{"title": "ABCDEFG..."}},
    {{"title": "HIJKLMN..."}},
    {{"title": "OPQRSTU..."}},
    {{"title": "VWXYZAB..."}},
  ]

Do NOT combine multiple bullets into a single section title!

=====================================
WHERE TO FIND SECTIONS
=====================================
Look for patterns within each submodule like:
- Bullet points (-, •, *)
- Numbered items (1., 2., a., b.)
- Indented sub-items under topic headings
- Specific skills or knowledge items

Examples of section patterns:
- "• ABCDEFG..."
- "• HIJKLMN..."
- "• OPQRSTU..."
- "• VWXYZAB..."

=====================================
IGNORE SOFT SKILLS SECTION
=====================================
COMPLETELY IGNORE the "Habilidades de gestión, personales y sociales" section.
Do NOT extract any content from this section.
Only extract from "Conocimientos / Capacidades cognitivas y prácticas".

=====================================
IF NO CLEAR SECTIONS EXIST
=====================================
If a submodule doesn't have clear section divisions:
- Create 1 section with the same title as the submodule
- This preserves the structure while indicating no subdivision exists

=====================================
OUTPUT REQUIREMENTS
=====================================
- Keep course_title unchanged
- Keep all module information unchanged (title, duration_hours, description)
- Keep all submodule titles unchanged
- Add sections array to each submodule

Target language for output: {language}

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON. No code fences, no commentary.

{format_instructions}
"""

extract_sections_prompt = PromptTemplate(
    template=EXTRACT_SECTIONS_TEMPLATE,
    input_variables=["pdf_markdown", "structure_with_submodules", "language", "format_instructions"]
)


# ============================================================================
# DESCRIPTION ENRICHMENT PROMPT
# ============================================================================

ENRICH_DESCRIPTIONS_TEMPLATE = """You are enriching a course structure with descriptions.

TASK: Add descriptions ONLY where they are missing (empty strings).

=====================================
COURSE STRUCTURE
=====================================
{structure_json}

=====================================
{research_context}
=====================================

=====================================
CRITICAL: PRESERVE EXISTING DESCRIPTIONS
=====================================
Some modules already have descriptions extracted from the PDF (from OBJETIVO).
These descriptions MUST be preserved EXACTLY as they are.

Rules:
1. If a module has a non-empty "description" field → KEEP IT UNCHANGED
2. If a module has an empty description ("") → ADD a description
3. Submodules and sections always need descriptions added

=====================================
ENRICHMENT RULES
=====================================
1. Keep ALL titles EXACTLY as they are - do NOT change them
2. Keep ALL duration_hours values unchanged
3. PRESERVE existing module descriptions from PDF (OBJETIVO)
4. Add descriptions only where missing:
   - Submodule descriptions: 1-2 sentences about the submodule's focus
   - Section descriptions: 1-2 sentences about what will be learned

=====================================
DESCRIPTION GUIDELINES
=====================================
Good descriptions:
✓ Explain what the learner will achieve
✓ Are specific to the topic
✓ Connect to the broader course goals
✓ Use clear, accessible language

Avoid:
✗ Generic phrases like "This section covers..."
✗ Simply repeating the title
✗ Overly technical jargon

=====================================
OUTPUT REQUIREMENTS
=====================================
- course_title: Keep unchanged
- modules: Keep all structure
  - Module descriptions: PRESERVE if non-empty, add only if empty
  - Submodule descriptions: Add meaningful descriptions
  - Section descriptions: Add meaningful descriptions

All NEW descriptions must be in: {language}
EXISTING descriptions must be preserved exactly.

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON. No code fences, no commentary.

{format_instructions}
"""

enrich_descriptions_prompt = PromptTemplate(
    template=ENRICH_DESCRIPTIONS_TEMPLATE,
    input_variables=["course_title", "structure_json", "research_context", "language", "format_instructions"]
)


# ============================================================================
# RESEARCH PROMPTS (for enrichment)
# ============================================================================

PDF_QUERY_GENERATION_TEMPLATE = """You are a research assistant helping to gather information for course development.

Given a course extracted from a PDF syllabus, generate search queries to research the topic.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Module topics:
{module_topics}

=====================================
QUERY GENERATION RULES
=====================================
1. Generate queries in ENGLISH for better search results
2. Focus on understanding the domain and pedagogical best practices
3. Include queries about:
   - Core concepts in the field
   - Common teaching approaches
   - Real-world applications
   - Prerequisites and foundations

=====================================
OUTPUT FORMAT
=====================================
Return ONLY a JSON array of {max_queries} search query strings in ENGLISH.
No code fences, no extra commentary.
"""

pdf_query_generation_prompt = PromptTemplate(
    template=PDF_QUERY_GENERATION_TEMPLATE,
    input_variables=["course_title", "module_topics", "max_queries"]
)


PDF_RESEARCH_SYNTHESIS_TEMPLATE = """You are a curriculum research specialist synthesizing information for course enrichment.

=====================================
COURSE INFORMATION
=====================================
- Course title: "{course_title}"
- Module topics:
{module_topics}

=====================================
RAW RESEARCH DATA
=====================================
{raw_research}

=====================================
YOUR TASK
=====================================
Synthesize the research into structured course planning information.
ALL output must be in ENGLISH.

Generate:
1. "course_summary": 3-5 sentence summary of the topic and its importance
2. "learning_objectives": 4-6 specific, measurable objectives
3. "assumed_prerequisites": 2-4 foundational requirements
4. "out_of_scope": 2-4 topics NOT covered
5. "key_topics": 6-10 canonical topics that should be covered

=====================================
OUTPUT FORMAT
=====================================
Return ONLY valid JSON. No code fences, no extra commentary.

{format_instructions}
"""

pdf_research_synthesis_prompt = PromptTemplate(
    template=PDF_RESEARCH_SYNTHESIS_TEMPLATE,
    input_variables=["course_title", "module_topics", "raw_research", "format_instructions"]
)


# ============================================================================
# SUMMARY GENERATION PROMPT
# ============================================================================

PDF_SUMMARY_GENERATION_TEMPLATE = """You are an expert curriculum planner creating content summaries for course sections.

=====================================
MODULE CONTEXT
=====================================
- Course Title: "{course_title}"
- Module Title: "{module_title}"
- Module Description: "{module_description}"
- Target Language: {language}

=====================================
SECTIONS IN THIS MODULE
=====================================
{sections_list}

=====================================
YOUR TASK
=====================================
Generate a 3-line summary for EACH section describing what specific content it will cover.

CRITICAL REQUIREMENTS:
1. Each summary must be EXACTLY 3 lines (sentences)
2. Summaries must be DISTINCT - no overlapping content between sections
3. Summaries should be SPECIFIC about what will be taught
4. Each summary should complement the others
5. Write all summaries in {language}

AVOID:
- Generic phrases like "This section covers..."
- Overlapping content between sections
- Vague descriptions
- Repeating the section title

GOOD EXAMPLE:
"Explores the mathematical foundations of wave-particle duality through de Broglie's hypothesis.
Derives the wavelength-momentum relationship and its experimental verification.
Connects quantum behavior to classical limits through the correspondence principle."

=====================================
OUTPUT FORMAT
=====================================
Return a JSON object mapping each section title (exactly as provided) to its 3-line summary.

Example format:
{{
    "Section Title 1": "Line 1.\\nLine 2.\\nLine 3.",
    "Section Title 2": "Line 1.\\nLine 2.\\nLine 3."
}}

Return ONLY valid JSON. No code fences, no extra text.
"""

pdf_summary_generation_prompt = PromptTemplate(
    template=PDF_SUMMARY_GENERATION_TEMPLATE,
    input_variables=["course_title", "module_title", "module_description", "language", "sections_list"]
)


# ============================================================================
# RETRY PROMPT
# ============================================================================

RETRY_TEMPLATE = """Your previous response doesn't follow the required schema and fails parsing.
Fix ONLY the formatting to comply with the schema; do not change the meaning or structure.

=====================================
EXPECTED SCHEMA
=====================================
{schema}

=====================================
CONSTRAINTS
=====================================
- Return pure JSON (no code fences, no extra text)
- Use exactly the specified field names
- No extra fields are allowed
- All titles must be written in {language}
- Do NOT change any extracted content, only fix formatting
"""

retry_prompt = PromptTemplate.from_template(RETRY_TEMPLATE)
