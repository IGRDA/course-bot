"""Prompts for video generator agent."""

from langchain_core.prompts import ChatPromptTemplate

# ---- Video Query Generation Prompt (Legacy - single query) ----
video_query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at creating YouTube search queries for educational content.
Generate a concise search query (4-8 words) that will find relevant educational videos.
Focus on finding tutorials, explanations, or educational content that matches the module topic.
Consider the target language when crafting the query - use the appropriate language for better results.""",
        ),
        (
            "human",
            """Generate a YouTube search query for educational videos about this module:

Course Title: {course_title}
Module Title: {module_title}
Module Description: {module_description}
Key Topics: {key_topics}
Target Language: {language}

Generate a search query in {language} that will find educational videos explaining this module's content.
Return ONLY the search query, nothing else.""",
        ),
    ]
)


# ---- Multi-Query Video Generation Prompt ----
video_multi_query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at creating YouTube search queries for educational content.
Your task is to generate multiple short search queries that will find highly relevant educational videos.

You must generate:
1. ONE general query about the overall module theme
2. Multiple specific queries about concrete concepts, terms, or techniques mentioned in the module

CRITICAL RULES:
- Each query MUST be 3-6 words. No exceptions. Think like a student typing into YouTube.
- Use simple, everyday vocabulary — never academic jargon or formal titles.
- Include one educational keyword per query: "tutorial", "explicado", "qué es", "cómo", "introduction", "explained", etc.
- Specific queries must target DISTINCT concrete sub-topics, not rephrase the general theme.
- Use the target language for all queries.

GOOD query examples (short, natural, YouTube-friendly):
- "reanimación neonatal tutorial"
- "método canguro recién nacido"
- "cuidado cordón umbilical bebé"
- "machine learning introduction tutorial"
- "qué es fotosíntesis explicado"

BAD query examples (too long, too academic — NEVER do this):
- "protocolo reanimación neonatal OMS y AAP paso a paso" (too long, too specific)
- "Convención Derechos del Niño UNICEF en neonatología tutorial" (academic jargon)
- "limpieza cordón umbilical con clorhexidina vs alcohol evidencia científica" (research paper title)

Output your response as valid JSON with this exact structure:
{{
  "general_query": "module theme tutorial",
  "specific_queries": ["concept one explicado", "concept two tutorial"]
}}""",
        ),
        (
            "human",
            """Generate YouTube search queries for educational videos about this module:

Course Title: {course_title}
Module Title: {module_title}
Module Description: {module_description}
Key Topics:
{key_topics}

Target Language: {language}
Number of specific queries needed: {num_specific_queries}

Remember: each query MUST be 3-6 words, written like a real YouTube search. No long academic phrases.

Return ONLY the JSON object, no additional text.""",
        ),
    ]
)
