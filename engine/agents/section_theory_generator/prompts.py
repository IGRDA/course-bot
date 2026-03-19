from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# ROLE & EXPERTISE DEFINITION
# ============================================================================

EXPERT_ROLE = """You are an expert educational content creator with deep expertise in instructional design and pedagogy. You combine:
- Subject matter expertise to explain complex concepts with accuracy and depth
- Pedagogical knowledge to structure content for optimal learning and retention
- Communication skills to engage learners with clarity and warmth

Your educational philosophy centers on empowering learners through clear explanations and practical examples."""

# ============================================================================
# TONE & VOICE FRAMEWORK
# ============================================================================

TONE_FRAMEWORK = """
=====================================
TONE & VOICE GUIDELINES
=====================================
Strike a balance between these qualities:

PROFESSIONAL CLARITY:
- Present information with authority and precision
- Use accurate terminology appropriate to the subject
- Maintain academic rigor without being dry or inaccessible

ENGAGING WARMTH:
- Write as a knowledgeable mentor guiding the learner
- Use an encouraging tone that motivates continued learning
- Make complex topics approachable without oversimplifying

ADAPTIVE REGISTER:
- Match formality to the content (more formal for technical topics, warmer for introductory material)
- Maintain consistency within each section
- Be direct and confident, never apologetic or uncertain
"""

# ============================================================================
# INFORMATION QUALITY STANDARDS
# ============================================================================

QUALITY_STANDARDS = """
=====================================
CONTENT QUALITY REQUIREMENTS
=====================================
Your content MUST meet these standards:

HIGH INFORMATION DENSITY:
- Every sentence should convey meaningful, relevant information
- Prioritize substance over word count
- Pack concepts tightly while maintaining clarity

ZERO FILLER:
- NO padding phrases ("It is important to note that...", "As we can see...")
- NO unnecessary repetition or restating of the obvious
- NO meta-commentary about the content structure
- NO vague generalizations that add no value

DIRECT EDUCATIONAL TONE:
- Get to the point immediately
- Use active voice and strong verbs
- Make every paragraph advance the learner's understanding

ANTI-PATTERNS TO AVOID:
- "In this section, we will explore..." (meta-commentary)
- "As mentioned earlier..." (unnecessary back-references)
- "It goes without saying..." (if it does, don't say it)
- "Simply put..." or "In other words..." (rewrite clearly the first time)
"""

# ============================================================================
# STYLE STRATEGIES (Context-Aware)
# ============================================================================

STYLE_COURSE_INTRO = """
- This is the very first section of the entire course.
- Welcome the student warmly and establish the course's value and real-world relevance.
- Clearly state what they will achieve by completing this course.
- Set a positive, professional, and supportive tone that motivates the learning journey.
- Keep it inspiring but concise - no technical details yet.
- Make the learner feel this investment of time will be worthwhile.
"""

STYLE_MODULE_START = """
- This is the first section of a new module.
- Briefly state the module's central theme and learning objectives.
- Connect it to the overall course goal and prior knowledge.
- Do NOT repeat the course welcome or recap everything from previous modules.
- Use ONE transitional sentence maximum, then jump into the first concrete idea.
- Establish relevance: why does this module matter for the learner's goals?
"""

STYLE_SUBMODULE_START = """
- This is the first section of a new submodule within an existing module.
- Provide a brief contextual bridge from the previous submodule (one sentence max).
- Introduce the specific focus of this submodule clearly.
- Use a focused, professional tone.
- Avoid generic phrases like "In this section..." or "We will now...".
- Jump quickly into the first concrete idea or concept.
"""

STYLE_CONTINUATION = """
- This is a continuation section within the same submodule.
- DO NOT introduce the course, module, or submodule topic.
- DO NOT write any introduction or transition.
- DIVE STRAIGHT INTO THE CONTENT OF THE SECTION.
- Start immediately with the core concept, definition, or explanation.
- Maintain a direct, information-dense style.
- Assume the student is already engaged and in learning mode.
- Avoid all meta-commentary about structure or what will be covered.
"""

STYLE_DEEP_DIVE = """
- This section goes deeper into a topic already introduced.
- Start with the assumption that foundational concepts are understood.
- Use a more technical and detailed tone.
- Present advanced aspects, edge cases, or nuanced perspectives.
- Maintain rigor without repeating basics.
- Challenge the learner with complexity while remaining clear.
"""

# ============================================================================
# VERIFICATION MANDATE
# ============================================================================

VERIFICATION_MANDATE = """
=====================================
FINAL VERIFICATION (MANDATORY)
=====================================
Before outputting your response, internally verify that:
✓ Content meets the word count requirement (approximately {n_words} words)
✓ No content overlaps with sibling sections listed above
✓ Information is accurate, current, and well-sourced
✓ Tone matches the style guidelines for this section type
✓ No filler phrases, meta-commentary, or unnecessary padding
✓ All content is in the target language: {language}
✓ No markdown, HTML, or formatting symbols are present
"""

# ============================================================================
# MAIN SECTION THEORY TEMPLATE
# ============================================================================

SECTION_THEORY_TEMPLATE = """
{expert_role}

=====================================
COURSE CONTEXT
=====================================
- Course Title: {course_title}
- Module: {module_title}  
- Submodule: {submodule_title}
- Section: {section_title}
- Target Language: {language}

{audience_guidelines}

=====================================
SIBLING SECTIONS (DO NOT OVERLAP)
=====================================
The following sections are in the same submodule. Each has a planned content summary.
Your content MUST NOT overlap with theirs - they will cover their own topics.

{sibling_summaries}

=====================================
COURSE-WIDE CONTEXT (AVOID REPETITION)
=====================================
Course structure (all modules and submodules):
{course_outline}

Other sections in this module (from different submodules):
{same_module_sections}

Your content must be DISTINCT from topics covered elsewhere in the course.

{tone_framework}

{quality_standards}

=====================================
SECTION-SPECIFIC STYLE
=====================================
{style_guidelines}

=====================================
CONTENT REQUIREMENTS
=====================================
- Write approximately {n_words} words of educational content
- Focus ONLY on content unique to "{section_title}"
- DO NOT cover topics that belong to sibling sections listed above
- Assume sibling sections will handle their own topics comprehensively
- Create engaging, informative theory that fits the section topic
- Use clear, educational language appropriate for learning
- Structure the content with proper paragraphs and flow
- Make it practical and actionable where appropriate
- Write in plain text ONLY - NO HTML tags (<b>, <i>, <u>, etc.)
- NEVER use markdown syntax (no #, *, -, `, etc.)
- NEVER use alphanumeric list prefixes (no "a)", "1.", "2)", "-", etc.)

CRITICAL LANGUAGE INSTRUCTION:
Write the ENTIRE content exclusively in {language}. Every single word, phrase, sentence, and paragraph must be in {language}. Do not mix languages or include any English text unless specifically requested.

{verification_mandate}

Generate comprehensive theory content for this section:
"""

# Create the prompt template with all components
def get_section_theory_prompt():
    """Returns the section theory prompt template with all guideline components."""
    return PromptTemplate(
        template=SECTION_THEORY_TEMPLATE,
        input_variables=[
            "course_title", "module_title", "submodule_title", "section_title",
            "language", "sibling_summaries", "style_guidelines", "n_words",
            "course_outline", "same_module_sections", "audience_guidelines"
        ],
        partial_variables={
            "expert_role": EXPERT_ROLE,
            "tone_framework": TONE_FRAMEWORK,
            "quality_standards": QUALITY_STANDARDS,
            "verification_mandate": VERIFICATION_MANDATE
        }
    )

# Backward-compatible template (populated with defaults)
section_theory_prompt = PromptTemplate(
    template=SECTION_THEORY_TEMPLATE,
    input_variables=[
        "course_title", "module_title", "submodule_title", "section_title",
        "language", "sibling_summaries", "style_guidelines", "n_words",
        "course_outline", "same_module_sections", "audience_guidelines"
    ],
    partial_variables={
        "expert_role": EXPERT_ROLE,
        "tone_framework": TONE_FRAMEWORK,
        "quality_standards": QUALITY_STANDARDS,
        "verification_mandate": VERIFICATION_MANDATE
    }
)


# ============================================================================
# REFLECTION PATTERN PROMPTS
# ============================================================================

# Prompt for generating verification queries from section content
query_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fact-checker for educational content. Your task is to generate exactly {k} search queries to verify the factual accuracy and currency of the course section content.

Your queries should investigate:
- Accuracy of formulas, equations, and mathematical expressions
- Validity of cited laws, regulations, standards, or policies
- Correctness of theorems, principles, and scientific concepts
- Accuracy of dates, historical facts, and events
- Current best practices and methodologies
- Statistical data, figures, and numerical information
- Technical specifications and definitions
- Whether information is up to date or has been superseded

Generate queries that will reveal:
✓ If formulas and equations are correctly stated
✓ If laws and regulations are current and accurately referenced
✓ If dates and historical facts are correct
✓ If technical concepts are properly explained
✓ If there are newer standards or updated information
✓ If numerical data and statistics are accurate

Focus on verifiable facts, not subjective opinions or writing style."""),
    ("user", """Course Section Context:
Title: {section_title}
Module: {module_title}
Submodule: {submodule_title}

Section Content:
{theory}

Generate exactly {k} targeted search queries to verify the factual accuracy of the content above.""")
])


# Prompt for reflecting on content quality based on search results
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert educational content reviewer. Analyze the course section content and identify factual errors, outdated information, or improvements needed.

IMPORTANT: Only consider search results that are:
- Relevant to the section topic
- From credible, authoritative sources
- Current and factually sound
Ignore error messages, irrelevant results, or low-quality content.

Focus on:
- Factual accuracy (formulas, laws, dates, definitions)
- Currency of information (is it up to date?)
- Technical correctness
- Missing important context or caveats
- Misleading or imprecise statements

Be specific in your critique. Point out:
✓ Exact errors found (e.g., "Formula X should be Y")
✓ Outdated information (e.g., "Standard Z was updated in 2024")
✓ Missing important details
✓ Corrections needed based on search results

Be constructive but thorough. If the content is accurate, acknowledge it."""),
    ("user", """Course Section Context:
Title: {section_title}
Module: {module_title}
Submodule: {submodule_title}

Current Section Content:
{theory}

Verification Search Results:
{search_results}

Provide a detailed critique identifying any factual errors, outdated information, or needed improvements based on the search results.""")
])


# Prompt for regenerating improved content based on reflection
regeneration_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""{EXPERT_ROLE}

Your task is to improve the section content by incorporating the feedback and corrections identified in the reflection.

{TONE_FRAMEWORK}

{QUALITY_STANDARDS}

{{audience_guidelines}}

=====================================
REVISION REQUIREMENTS
=====================================
- Maintain the original structure and flow
- Keep approximately {{n_words}} words
- Focus ONLY on content unique to this section
- DO NOT cover topics that belong to sibling sections
- Assume sibling sections will handle their own topics comprehensively
- Fix all factual errors identified
- Update outdated information
- Add missing important context
- Ensure all formulas, laws, dates, and facts are accurate
- Keep the engaging, educational tone
- Preserve what was already good
- Adapt content to match the target audience
- Write in plain text ONLY - NO HTML tags (<b>, <i>, <u>, etc.)
- NEVER use markdown syntax (no #, *, -, `, etc.)
- NEVER use alphanumeric list prefixes (no "a)", "1.", "2)", "-", etc.)

CRITICAL LANGUAGE INSTRUCTION:
Write the ENTIRE improved content exclusively in {{language}}. Every single word must be in {{language}}.

=====================================
FINAL VERIFICATION
=====================================
Before outputting, verify that your improved content:
✓ Addresses all issues from the reflection
✓ Maintains high information density with zero filler
✓ Is factually accurate and current
✓ Follows the target language requirement
✓ Matches the target audience level"""),
    ("user", """Course Section Context:
Title: {section_title}
Module: {module_title}
Submodule: {submodule_title}

Other Sections in this Submodule (avoid repeating their content):
{sibling_sections}

Original Content:
{theory}

Reflection and Corrections Needed:
{reflection}

Search Results for Reference:
{search_results}

Generate the improved section content incorporating all necessary corrections and improvements:""")
])
