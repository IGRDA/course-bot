"""Prompt templates for bibliography generation."""

from langchain_core.prompts import ChatPromptTemplate

# Prompt for LLM to suggest relevant books for a module
book_suggestion_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert librarian specializing in academic textbooks and educational resources.

Your task is to suggest REAL, WELL-KNOWN books that can be found in major libraries and bookstores.

CRITICAL REQUIREMENTS:
1. ONLY suggest books that are widely recognized and easily verifiable
2. Prefer classic textbooks that have multiple editions (indicates quality)
3. **PRIORITIZE books in the course language** - if the course is in Spanish, suggest Spanish-language books first
4. For Spanish courses, include books from Spanish-language publishers:
   - Fondo de Cultura Económica, Siglo XXI, Alianza Editorial, Ariel, Gedisa
   - Editorial Universitaria (various countries), UNAM, Trotta, Gredos
   - Spanish editions from international publishers (Pearson, McGraw-Hill, etc.)
5. For English courses, include books from: MIT Press, Cambridge, Oxford, Springer, Wiley, Pearson, O'Reilly, Addison-Wesley
6. You may include some English books as supplements, but majority should match course language

BOOK QUALITY INDICATORS (prefer books with these):
- Multiple editions (e.g., "5th edition" / "5ª edición")
- Famous authors in the field
- Published by academic presses
- Widely cited in academic literature
- Used as textbooks in universities

RECENCY PREFERENCE:
- **PRIORITIZE recent publications** (2015 or later preferred, 2010+ acceptable)
- When suggesting classics, prefer the LATEST EDITION available
- Include publication year accurately - newer editions are better
- Mix: ~70% recent (post-2010), ~30% foundational classics (if relevant)

Do NOT:
- Invent fictional books
- Suggest obscure or self-published books
- Mix up author names with book titles
- Suggest books you're not confident exist"""),
    
    ("human", """Suggest {num_books} highly-regarded academic books for this module:

**Course Title:** {course_title}
**Module Title:** {module_title}
**Module Description:** {module_description}

**Topics covered:**
{module_topics}

**Course Language:** {language}

IMPORTANT LANGUAGE INSTRUCTION:
- If course language is Spanish/Español: Suggest books primarily IN SPANISH. Include Spanish editions of classic texts, or original Spanish-language books from Hispanic academic publishers.
- If course language is English: Suggest books in English.
- You may include 1-2 books in other languages if they are essential references.

{exclusion_note}

Return a JSON array. Each book MUST have:
- "title": Exact book title (in the book's original language)
- "authors": Array of full author names
- "year": Most recent/latest edition year (number, REQUIRED - use the latest edition available)
- "publisher": Publisher name (use well-known publishers)
- "language": Language of the book ("es" for Spanish, "en" for English, etc.)
- "relevance": Why this book is relevant (1 sentence)

IMPORTANT: Prefer books published after 2010. For older classic texts, use the latest edition available.

IMPORTANT: Only include books you are 100% certain are real publications.

Example for a Spanish course:
```json
[
  {{
    "title": "Introducción a la mecánica cuántica",
    "authors": ["Luis de la Peña"],
    "year": 2014,
    "publisher": "Fondo de Cultura Económica",
    "language": "es",
    "relevance": "Texto clásico en español sobre mecánica cuántica para estudiantes universitarios."
  }},
  {{
    "title": "Física cuántica",
    "authors": ["Carlos Sánchez del Río"],
    "year": 2005,
    "publisher": "Pirámide",
    "language": "es",
    "relevance": "Manual universitario de física cuántica en español con enfoque pedagógico."
  }}
]
```

Return ONLY the JSON array:""")
])


# Prompt for formatting a single book citation in APA 7
apa_formatting_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in APA 7th edition citation formatting.

Format book citations according to APA 7 guidelines:

**Basic format:**
Author, A. A., & Author, B. B. (Year). Title of work: Capital letter also for subtitle (Edition ed.). Publisher. DOI or URL

**Rules:**
1. Authors: Last name, First initial. Middle initial. Use & before last author.
2. Year: In parentheses. Use (n.d.) if no date.
3. Title: Italicized. Only first word and proper nouns capitalized.
4. Edition: Include only if not first edition, e.g., (2nd ed.)
5. Publisher: Do not include location.
6. DOI/URL: Include if available, no period after URL.

**Examples:**
- Single author: Jackson, L. M. (2019). The psychology of prejudice: From attitudes to social action (2nd ed.). American Psychological Association.
- Multiple authors: Sapolsky, R. M., & Balt, S. (2020). Behave: The biology of humans at our best and worst. Penguin Books.
- With DOI: Brown, L. S. (2018). Feminist therapy (2nd ed.). American Psychological Association. https://doi.org/10.1037/0000092-000"""),
    
    ("human", """Format this book in APA 7 style:

Title: {title}
Authors: {authors}
Year: {year}
Publisher: {publisher}
Edition: {edition}
DOI: {doi}
URL: {url}

Return ONLY the formatted citation, nothing else.""")
])
